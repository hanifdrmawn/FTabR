# k-means.py
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import lib
from lib import KWArgs
from sklearn.preprocessing import OneHotEncoder

@dataclass(frozen=True)
class Config:
    seed: int
    data: KWArgs  # lib.data.build_dataset
    k_range: list[int]  # Range of k to search for best clusters


def find_best_k(embeddings: np.ndarray, k_range: list[int]) -> int:
    """Find the best number of clusters (k) using silhouette score."""
    best_k = k_range[0]
    best_score = -1

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings)
        score = silhouette_score(embeddings, kmeans.labels_)
        print(f'k={k}, silhouette_score={score}')
        
        if score > best_score:
            best_score = score
            best_k = k

    return best_k

def save_clusters(labels, output):
    """Save cluster labels to a text file."""
    output_file = Path(output) / "clusters.txt"
    with open(output_file, "w") as f:
        for label in labels:
            f.write(f"{label}\n")


@torch.inference_mode()
def main(
    config: lib.JSONDict, output: Union[str, Path], *, force: bool = False
) -> Optional[lib.JSONDict]:
    if not lib.start(output, force=force):
        return None

    output = Path(output)
    report = lib.create_report(config)
    C = lib.make_config(Config, config)

    # Set random seed
    np.random.seed(C.seed)
    torch.manual_seed(C.seed)

    # Load dataset
    dataset = lib.build_dataset(**C.data)

    # Prepare embeddings
    embeddings = {'train': [], 'val': [], 'test': []}
    if dataset.X_num is not None:
        for part in dataset.parts():
            embeddings[part].append(dataset.X_num[part])
    if dataset.X_bin is not None:
        for part in dataset.parts():
            embeddings[part].append(dataset.X_bin[part])
    if dataset.X_cat is not None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        for part in dataset.parts():
            embeddings[part].append(encoder.fit_transform(dataset.X_cat[part]))
        assert not dataset.is_multiclass, 'Categorical features not supported for multiclass problems'

    # Concatenate embeddings
    embeddings = {
        k: np.concatenate(v, axis=-1) for k, v in embeddings.items()
    }

    # Convert to numpy arrays
    train_embeddings = embeddings['train']

    # Find best k (number of clusters)
    best_k = find_best_k(train_embeddings, C.k_range)
    report['best_k'] = best_k

    # Fit KMeans with the best k
    kmeans = KMeans(n_clusters=best_k, random_state=C.seed).fit(train_embeddings)
    report['inertia'] = kmeans.inertia_

    # Save cluster labels
    save_clusters(kmeans.labels_, output)
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_Function_cli(main)