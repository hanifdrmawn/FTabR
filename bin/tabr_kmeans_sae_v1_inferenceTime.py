# The model.

# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import delu
import faiss
# import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from tqdm import tqdm
import time
import random

import lib
from lib import KWArgs


@dataclass(frozen=True)
class Config:
    seed: int
    data: Union[lib.Dataset[np.ndarray], KWArgs]  # lib.data.build_dataset
    model: KWArgs  # Model
    context_size: int
    optimizer: KWArgs  # lib.deep.make_optimizer
    batch_size: int
    patience: Optional[int]
    n_epochs: Union[int, float]

class SAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, beta: float = 1e-3):
        super().__init__()
        self.beta = beta  # sparsity penalty weight (bukan KL seperti SAE)
        self.latent_dim = latent_dim

        # Encoder SAE
        self.sae_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),  # ReLU di sini yang memaksa sparsity
        )

        # Decoder SAE
        self.sae_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Proyeksi balik ke d_main
        self.proj = nn.Linear(latent_dim, input_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = self.sae_encoder(x)

        # Reconstruction
        x_recon = self.sae_decoder(z)
        recon_loss = F.mse_loss(x_recon, x)

        # Sparsity loss: L1 penalty pada aktivasi z
        sparsity_loss = z.abs().mean()

        sae_loss = recon_loss + self.beta * sparsity_loss

        out = self.proj(z)
        return out, sae_loss

import random

class PSOOptimizer:
    """
    Particle Swarm Optimization untuk hyperparameter SAE.
    Evaluasi tiap partikel = score validasi dari TabR.
    """
    
    def __init__(
        self,
        n_particles: int = 10,
        n_iterations: int = 5,
        bounds: dict = None,
        w: float = 0.5,   # inertia
        c1: float = 1.5,  # cognitive
        c2: float = 1.5,  # social
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Batas pencarian: [latent_dim, hidden_dim, beta]
        self.bounds = bounds or {
            'latent_dim': (8, 128),
            'hidden_dim': (32, 256),
            'beta':       (0.1, 5.0),
        }

        # Inisialisasi partikel
        self.particles = self._init_particles()
        self.velocities = [{k: 0.0 for k in self.bounds} for _ in range(n_particles)]
        self.pbest = [p.copy() for p in self.particles]
        self.pbest_scores = [-float('inf')] * n_particles
        self.gbest = self.particles[0].copy()
        self.gbest_score = -float('inf')

    def _init_particles(self):
        particles = []
        for _ in range(self.n_particles):
            p = {}
            for k, (lo, hi) in self.bounds.items():
                val = random.uniform(lo, hi)
                # latent_dim dan hidden_dim harus integer
                if k in ('latent_dim', 'hidden_dim'):
                    val = int(val)
                p[k] = val
            particles.append(p)
        return particles

    def step(self, scores: list[float]):
        """Update kecepatan & posisi berdasarkan scores dari evaluasi TabR."""
        for i, score in enumerate(scores):
            if score > self.pbest_scores[i]:
                self.pbest_scores[i] = score
                self.pbest[i] = self.particles[i].copy()
            if score > self.gbest_score:
                self.gbest_score = score
                self.gbest = self.particles[i].copy()

        for i in range(self.n_particles):
            for k, (lo, hi) in self.bounds.items():
                r1, r2 = random.random(), random.random()
                self.velocities[i][k] = (
                    self.w * self.velocities[i][k]
                    + self.c1 * r1 * (self.pbest[i][k] - self.particles[i][k])
                    + self.c2 * r2 * (self.gbest[k] - self.particles[i][k])
                )
                new_val = self.particles[i][k] + self.velocities[i][k]
                new_val = max(lo, min(hi, new_val))
                if k in ('latent_dim', 'hidden_dim'):
                    new_val = int(new_val)
                self.particles[i][k] = new_val

    def get_particles(self) -> list[dict]:
        return self.particles

    def get_best(self) -> dict:
        return self.gbest

class Model(nn.Module):
    def __init__(
        self,
        *,
        #
        num_clusters: int,
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: Optional[int],
        #
        num_embeddings: Optional[dict],  # lib.deep.ModuleSpec
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        predictor_n_blocks: int,
        mixer_normalization: Union[bool, Literal['auto']],
        context_dropout: float,
        dropout0: float,
        dropout1: Union[float, Literal['dropout0']],
        normalization: str,
        activation: str,
        #
        # The following options should be used only when truly needed.
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
        sae_latent_dim: int = 32,
        sae_hidden_dim: int = 128,
        sae_beta: float = 1.0,
    ) -> None:
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        super().__init__()
        if dropout1 == 'dropout0':
            dropout1 = dropout0

        self.num_clusters = num_clusters
        self.kmeans = None
        self.one_hot_encoder = (
            lib.OneHotEncoder(cat_cardinalities) if cat_cardinalities else None
        )
        self.num_embeddings = (
            None
            if num_embeddings is None
            else lib.make_module(num_embeddings, n_features=n_num_features)
        )

        # >>> E
        d_in = (
            n_num_features
            * (1 if num_embeddings is None else num_embeddings['d_embedding'])
            + n_bin_features
            + sum(cat_cardinalities)
        )
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        Activation = getattr(nn, activation)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [make_block(i > 0) for i in range(encoder_n_blocks)]
        )

        # >>> R
        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes is None
            else nn.Sequential(
                nn.Embedding(n_classes, d_main), delu.nn.Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        # >>> P
        self.blocks1 = nn.ModuleList(
            [make_block(True) for _ in range(predictor_n_blocks)]
        )
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(),
            nn.Linear(d_main, lib.get_d_out(n_classes)),
        )

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.sae = SAE(
            input_dim=d_main,
            latent_dim=sae_latent_dim,
            hidden_dim=sae_hidden_dim,
            beta=sae_beta,
        )
        self.sae_loss_weight = 0.1
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]  # noqa: E501
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]  # noqa: E501
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]  # noqa: E501

    def _encode(self, x_: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        x_num = x_.get('num')
        x_bin = x_.get('bin')
        x_cat = x_.get('cat')
        del x_

        x = []
        if x_num is None:
            assert self.num_embeddings is None
        else:
            x.append(
                x_num
                if self.num_embeddings is None
                else self.num_embeddings(x_num).flatten(1)
            )
        if x_bin is not None:
            x.append(x_bin)
        if x_cat is None:
            assert self.one_hot_encoder is None
        else:
            assert self.one_hot_encoder is not None
            x.append(self.one_hot_encoder(x_cat))
        assert x
        x = torch.cat(x, dim=1)

        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        x, sae_loss = self.sae(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k, sae_loss

    def forward(
        self,
        *,
        x_: dict[str, Tensor],
        y: Optional[Tensor],
        candidate_x_: dict[str, Tensor],
        candidate_y: Tensor,
        context_size: int,
        is_train: bool,
    ) -> tuple[Tensor, Tensor]:
        # >>>
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.memory_efficient
        ):
            # NOTE: during evaluation, candidate keys can be computed just once, which
            # looks like an easy opportunity for optimization. However:
            # - if your dataset is small or/and the encoder is just a linear layer
            #   (no embeddings and encoder_n_blocks=0), then encoding candidates
            #   is not a bottleneck.
            # - implementing this optimization makes the code complex and/or unobvious,
            #   because there are many things that should be taken into account:
            #     - is the input coming from the "train" part?
            #     - is self.training True or False?
            #     - is PyTorch autograd enabled?
            #     - is saving and loading checkpoints handled correctly?
            # This is why we do not implement this optimization.

            # When memory_efficient is True, this potentially heavy computation is
            # performed without gradients.
            # Later, it is recomputed with gradients only for the context objects.
            candidate_k = (
                self._encode(candidate_x_)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [
                        self._encode(x)[1]
                        for x in delu.iter_batches(
                            candidate_x_, self.candidate_encoding_batch_size
                        )
                    ]
                )
            )
        x, k, sae_loss = self._encode(x_)
        if is_train:
            # NOTE: here, we add the training batch back to the candidates after the
            # function `apply_model` removed them. The further code relies
            # on the fact that the first batch_size candidates come from the
            # training batch.
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None

        # >>>
        # The search below is optimized for larger datasets and is significantly faster
        # than the naive solution (keep autograd on + manually compute all pairwise
        # squared L2 distances + torch.topk).
        # For smaller datasets, however, the naive solution can actually be faster.
        batch_size, d_main = k.shape
        # Ketika mencari cluster terdekat
        with torch.no_grad():
            if self.kmeans is None:
                self.kmeans = faiss.Kmeans(d_main, self.num_clusters, niter=20)
            self.kmeans.train(candidate_k.cpu().numpy())
            # Cari hanya pusat klaster terdekat untuk setiap titik dalam batch
            distances, cluster_indices = self.kmeans.index.search(k.cpu().numpy(), 1)  # Hanya 1 klaster terdekat
            # Konversi hasil kembali ke tensor untuk digunakan lebih lanjut
            context_idx = torch.tensor(cluster_indices, device=k.device)
            if is_train:
                distances = torch.tensor(distances, device=k.device)
                # NOTE: to avoid leakage, the index i must be removed from the i-th row,
                # (because of how candidate_k is constructed).
                # Hindari kebocoran data, pastikan indeks yang sama dihapus
                distances[
                    context_idx == torch.arange(batch_size, device=k.device)[:, None]
                ] = torch.inf
                # Not the most elegant solution to remove the argmax, but anyway.
                # Hilangkan kandidat dengan jarak terbesar (argmax)
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            # Repeating the same computation,
            # but now only for the context objects and with autograd on.
            context_k = self._encode(
                {
                    ftype: torch.cat([x_[ftype], candidate_x_[ftype]])[
                        context_idx
                    ].flatten(0, 1)
                    for ftype in x_
                }
            )[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]

        # In theory, when autograd is off, the distances obtained during the search
        # can be reused. However, this is not a bottleneck, so let's keep it simple
        # and use the same code to compute `similarities` during both
        # training and evaluation.
        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # >>>
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x, sae_loss


def main(
    config: lib.JSONDict, output: Union[str, Path], *, force: bool = False
) -> Optional[lib.JSONDict]:
    # >>> start
    if not lib.start(output, force=force):
        return None

    output = Path(output)
    report = lib.create_report(config)
    C = lib.make_config(Config, config)

    delu.random.seed(C.seed)
    device = lib.get_device()

    # >>> data
    dataset = (
        C.data if isinstance(C.data, lib.Dataset) else lib.build_dataset(**C.data)
    ).to_torch(device)
    if dataset.is_regression:
        dataset.data['Y'] = {k: v.float() for k, v in dataset.Y.items()}
    Y_train = dataset.Y['train'].to(
        torch.long if dataset.is_multiclass else torch.float
    )

    # >>> model
    num_clusters = C.model.pop('num_clusters')

    # >>> training
    def zero_wd_condition(
        module_name: str,
        module: nn.Module,
        parameter_name: str,
        parameter: nn.parameter.Parameter,
    ):
        return (
            'label_encoder' in module_name
            or 'label_encoder' in parameter_name
            or lib.default_zero_weight_decay_condition(
                module_name, module, parameter_name, parameter
            )
        )

    loss_fn = lib.get_loss_fn(dataset.task_type)

    train_size = dataset.size('train')
    train_indices = torch.arange(train_size, device=device)

    def get_Xy(part: str, idx) -> tuple[dict[str, Tensor], Tensor]:
        batch = (
            {
                key[2:]: dataset.data[key][part]
                for key in dataset.data
                if key.startswith('X_')
            },
            dataset.Y[part],
        )
        return (
            batch
            if idx is None
            else ({k: v[idx] for k, v in batch[0].items()}, batch[1][idx])
        )

    def quick_train_eval(temp_model, n_epochs: int = 3) -> float:
        temp_optimizer = lib.make_optimizer(
            temp_model, **C.optimizer,
            zero_weight_decay_condition=zero_wd_condition
        )
        temp_model.train()
        for _ in range(n_epochs):
            for batch_idx in lib.make_random_batches(train_size, C.batch_size, device):
                def compute_loss(idx):
                    candidate_indices = train_indices
                    candidate_indices = candidate_indices[~torch.isin(candidate_indices, idx)]
                    candidate_x, candidate_y = get_Xy('train', candidate_indices)
                    x, y = get_Xy('train', idx)
                    output, vl = temp_model(
                        x_=x, y=y,
                        candidate_x_=candidate_x,
                        candidate_y=candidate_y,
                        context_size=C.context_size,
                        is_train=True,
                    )
                    return loss_fn(output.squeeze(-1), Y_train[idx]) + temp_model.sae_loss_weight * vl
                lib.train_step(temp_optimizer, compute_loss, batch_idx, C.batch_size)
        # Evaluasi val
        temp_model.eval()
        with torch.inference_mode():
            preds = torch.cat([
                temp_model(
                    x_=get_Xy('val', idx)[0],
                    y=None,
                    candidate_x_=get_Xy('train', None)[0],
                    candidate_y=dataset.Y['train'],
                    context_size=C.context_size,
                    is_train=False,
                )[0].squeeze(-1)
                for idx in torch.arange(dataset.size('val'), device=device).split(1024)
            ]).cpu().numpy()
        prediction_type = None if dataset.is_regression else 'logits'
        metrics = dataset.calculate_metrics({'val': preds}, prediction_type)
        return metrics['val']['score']
    
    epoch = 0
    eval_batch_size = 32768
    chunk_size = None
    progress = delu.ProgressTracker(C.patience)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore[code]

    # >>> PSO untuk optimasi SAE
    pso = PSOOptimizer(n_particles=6, n_iterations=3)

    logger.info("Starting PSO to optimize SAE hyperparameters...")
    for pso_iter in range(pso.n_iterations):
        particles = pso.get_particles()
        scores = []
        for particle in particles:
            temp_model = Model(
                num_clusters=num_clusters,
                n_num_features=dataset.n_num_features,       # ← tambahkan
                n_bin_features=dataset.n_bin_features,       # ← tambahkan
                cat_cardinalities=dataset.cat_cardinalities(), # ← tambahkan
                n_classes=dataset.n_classes(),               # ← tambahkan
                sae_latent_dim=particle['latent_dim'],
                sae_hidden_dim=particle['hidden_dim'],
                sae_beta=particle['beta'],
                **C.model,
            ).to(device)

            # Latih beberapa epoch saja untuk evaluasi cepat (misal 3 epoch)
            val_score = quick_train_eval(temp_model, n_epochs=3)
            scores.append(val_score)
            del temp_model

        pso.step(scores)
        logger.info(f"PSO iter {pso_iter}: best score={pso.gbest_score:.4f}, params={pso.gbest}")

    best_sae_params = pso.get_best()
    logger.info(f"Best SAE params from PSO: {best_sae_params}")

    # Gunakan best SAE params untuk training final
    model = Model(
        num_clusters=num_clusters,
        n_num_features=dataset.n_num_features,      # ← tambahkan ini
        n_bin_features=dataset.n_bin_features,      # ← tambahkan ini
        cat_cardinalities=dataset.cat_cardinalities(),  # ← tambahkan ini
        n_classes=dataset.n_classes(),              # ← tambahkan ini
        sae_latent_dim=best_sae_params['latent_dim'],
        sae_hidden_dim=best_sae_params['hidden_dim'],
        sae_beta=best_sae_params['beta'],
        **C.model,
    ).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    report['n_parameters'] = lib.get_n_parameters(model)   # ← pindah ke sini
    logger.info(f'n_parameters = {report["n_parameters"]}')
    report['prediction_type'] = None if dataset.is_regression else 'logits'

    optimizer = lib.make_optimizer(
        model, **C.optimizer, zero_weight_decay_condition=zero_wd_condition
    )

    def apply_model(part: str, idx: Tensor, training: bool):
        x, y = get_Xy(part, idx)

        candidate_indices = train_indices
        is_train = part == 'train'
        if is_train:
            # NOTE: here, the training batch is removed from the candidates.
            # It will be added back inside the model's forward pass.
            candidate_indices = candidate_indices[~torch.isin(candidate_indices, idx)]
        candidate_x, candidate_y = get_Xy(
            'train',
            # This condition is here for historical reasons, it could be just
            # the unconditional `candidate_indices`.
            None if candidate_indices is train_indices else candidate_indices,
        )

        output, sae_loss = model(          # ← unpack tuple dari forward()
            x_=x, y=y if is_train else None,
            candidate_x_=candidate_x,
            candidate_y=candidate_y,
            context_size=C.context_size,
            is_train=is_train,
        )
        return output.squeeze(-1), sae_loss

    @torch.inference_mode()
    def evaluate(parts: list[str], eval_batch_size: int):
        model.eval()
        predictions = {}
        total_inference_time = 0
        for part in parts:
            while eval_batch_size:
                try:
                    start_time = time.time()
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx, False)[0]
                                for idx in torch.arange(
                                    dataset.size(part), device=device
                                ).split(eval_batch_size)
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                    end_time = time.time()
                    inference_time = (end_time - start_time)
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    logger.warning(f'eval_batch_size = {eval_batch_size}')
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')
        total_inference_time += inference_time
        report['total_inference_time'] = total_inference_time
        metrics = (
            dataset.calculate_metrics(predictions, report['prediction_type'])
            if lib.are_valid_predictions(predictions)
            else {x: {'score': -999999.0} for x in predictions}
        )
        return metrics, predictions, eval_batch_size

    def save_checkpoint():
        lib.dump_checkpoint(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'random_state': delu.random.get_state(),
                'progress': progress,
                'report': report,
                'timer': timer,
                'training_log': training_log,
            },
            output,
        )
        lib.dump_report(report, output)
        lib.backup_output(output)

    print()
    timer = lib.run_timer()
    while epoch < C.n_epochs:
        print(f'[...] {lib.try_get_relative_path(output)} | {timer}')

        model.train()
        epoch_losses = []
        for batch_idx in tqdm(
            lib.make_random_batches(train_size, C.batch_size, device),
            desc=f'Epoch {epoch}',
        ):
            loss, new_chunk_size = lib.train_step(
                optimizer,
                lambda idx: (
                    lambda out, vl:
                        loss_fn(out, Y_train[idx]) + model.sae_loss_weight * vl
                )(*apply_model('train', idx, True)),   # ← gabungkan task_loss + sae_loss
                batch_idx,
                chunk_size or C.batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or C.batch_size):
                chunk_size = new_chunk_size
                logger.warning(f'chunk_size = {chunk_size}')

        epoch_losses, mean_loss = lib.process_epoch_losses(epoch_losses)
        metrics, predictions, eval_batch_size = evaluate(
            ['val', 'test'], eval_batch_size
        )
        lib.print_metrics(mean_loss, metrics)
        training_log.append(
            {'epoch-losses': epoch_losses, 'metrics': metrics, 'time': timer()}
        )
        writer.add_scalars('loss', {'train': mean_loss}, epoch, timer())
        for part in metrics:
            writer.add_scalars('score', {part: metrics[part]['score']}, epoch, timer())

        progress.update(metrics['val']['score'])
        if progress.success:
            lib.celebrate()
            report['best_epoch'] = epoch
            report['metrics'] = metrics
            save_checkpoint()
            lib.dump_predictions(predictions, output)

        elif progress.fail or not lib.are_valid_predictions(predictions):
            break

        epoch += 1
        print()
    report['time'] = str(timer)

    # >>> finish
    model.load_state_dict(lib.load_checkpoint(output)['model'])
    report['metrics'], predictions, _ = evaluate(
        ['train', 'val', 'test'], eval_batch_size
    )
    report['chunk_size'] = chunk_size
    report['eval_batch_size'] = eval_batch_size
    lib.dump_predictions(predictions, output)
    lib.dump_summary(lib.summarize(report), output)
    save_checkpoint()
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_Function_cli(main)
