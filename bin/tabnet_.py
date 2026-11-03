# TabNet

# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import delu
import numpy as np
from loguru import logger
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

import lib
from lib import KWArgs


@dataclass(frozen=True)
class Config:
    seed: int
    data: Union[lib.Dataset[np.ndarray], KWArgs]  # lib.data.build_dataset
    model: KWArgs  # TabNetClassifier/TabNetRegressor
    fit: KWArgs    # .fit()


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

    # >>> data
    dataset = C.data if isinstance(C.data, lib.Dataset) else lib.build_dataset(**C.data)
    dataset = dataset.merge_num_bin()
    assert dataset.X_num is not None
    assert set(dataset.data.keys()) == {'X_num', 'Y'}, set(dataset.data.keys())

    X_train = dataset.X_num['train']
    X_val   = dataset.X_num['val']
    X_test  = dataset.X_num['test']
    Y_train = dataset.Y['train']
    Y_val   = dataset.Y['val']

    # >>> model
    model_extra_kwargs = {'seed': C.seed, 'verbose': 0}
    if dataset.is_regression:
        model = TabNetRegressor(**C.model, **model_extra_kwargs)
        report['prediction_type'] = None
    else:
        model = TabNetClassifier(**C.model, **model_extra_kwargs)
        report['prediction_type'] = 'probs'

    # >>> training
    logger.info('training...')
    with delu.Timer() as timer:
        if dataset.is_regression:
            model.fit(
                X_train, Y_train.reshape(-1, 1),
                eval_set=[(X_val, Y_val.reshape(-1, 1))],
                **C.fit,
            )
        else:
            model.fit(
                X_train, Y_train,
                eval_set=[(X_val, Y_val)],
                **C.fit,
            )
    report['time'] = str(timer)

    # >>> finish
    def predict(X):
        if dataset.is_regression:
            return model.predict(X).flatten()
        elif dataset.is_multiclass:
            return model.predict_proba(X)
        else:
            return model.predict_proba(X)[:, 1]

    # Inference time — ukur hanya pada test set
    start_time = time.time()
    _ = predict(X_test)
    report['total_inference_time'] = time.time() - start_time

    predictions = {k: predict(v) for k, v in dataset.X_num.items()}
    report['metrics'] = dataset.calculate_metrics(
        predictions, report['prediction_type']
    )
    report['n_parameters'] = sum(
        p.numel() for p in model.network.parameters() if p.requires_grad
    )

    lib.dump_predictions(predictions, output)
    lib.dump_summary(lib.summarize(report), output)
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_Function_cli(main)