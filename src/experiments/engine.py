# src/experiments/engine.py

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.plotting import plot_nlr_extrapolation_heatmap
from src.training import prepareDataset, burstsMultistepDataset, train_model
from src.evaluate import reconstruct_TDDMD, normalize, rmse
from src.models import TDDMD, TDTF, DelayEmbeddedMLP
from src.experiments.arrays import (
    check_case_data,
    prediction_to_dK,
    require_NdK,
    require_T,
)


@dataclass
class ExperimentConfig:
    case: str
    n_values: list[int]
    h_values: list[int]
    seeds: list[int]

    models: list[str] = field(default_factory=lambda: ["tdtf"])

    m: int = 1
    J: int = 5000
    dataset_seed: int = 0
    replace: bool = False

    batch_size: int = 100
    lr: float = 1e-2
    weight_decay: float = 1e-2
    epochs: int = 500

    pos_encoding: bool = True
    multistep: bool = False

    save_predictions: bool = False
    save_checkpoints: bool = False
    make_plots: bool = False

    eval_traj_indices: list[int] | None = None
    output_root: str | None = None


@dataclass
class CaseData:
    name: str
    train_data: np.ndarray       # [N_train, d, K]
    eval_data: np.ndarray        # [N_eval, d, K]
    T: np.ndarray                # [K]
    metadata: dict[str, Any] = field(default_factory=dict)
    ylim: tuple[float, float] | None = None
    plot_kind: str = "single"    # "single", "multi", or "stack"


def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


MODEL_REGISTRY = {
    "tdtf": TDTF,
    "delay_mlp": DelayEmbeddedMLP
}


def make_training_artifacts(case_data: CaseData, cfg: ExperimentConfig, n: int):
    Xtrain = burstsMultistepDataset(
        case_data.train_data,
        n=n,
        m=cfg.m,
        J=cfg.J,
        replace=cfg.replace,
        seed=cfg.dataset_seed,
    )

    Xtrain_norm, vmin, vmax = normalize(Xtrain)
    train_dataset = prepareDataset(Xtrain_norm, n)
    A_TD = TDDMD(Xtrain_norm)

    return {
        "Xtrain": Xtrain,
        "Xtrain_norm": Xtrain_norm,
        "vmin": vmin,
        "vmax": vmax,
        "train_dataset": train_dataset,
        "A_TD": A_TD,
    }


def predict_tddmd(A_TD, eval_data, n, m, steps, vmin, vmax):
    X_true = require_NdK(eval_data, name="eval_data")

    N_eval, d, K = X_true.shape
    X_pred_TD = np.empty((N_eval, d, K), dtype=float)

    for i in range(N_eval):
        x0_TD = X_true[i, :, : n + m - 1]

        pred_TD_raw = reconstruct_TDDMD(
            A_TD,
            x0_TD,
            vmin=vmin,
            vmax=vmax,
            steps=K,
        )
        X_pred_TD[i] = prediction_to_dK(
            pred_TD_raw,
            d=d,
            K=K,
            name=f"X_pred_TD[{i}]",
        )

    return X_true.copy(), X_pred_TD


def predict_neural_model(model, eval_data, n, steps, vmin, vmax):
    X_true = require_NdK(eval_data, name="eval_data")

    N_eval, d, K = X_true.shape
    X_pred = np.empty((N_eval, d, K), dtype=float)

    for i in range(N_eval):
        x0 = X_true[i, :, :n]

        pred_raw = model.AR_sliding(
            x0,
            vmin=vmin,
            vmax=vmax,
            steps=K,
        )
        X_pred[i] = prediction_to_dK(
            pred_raw,
            d=d,
            K=K,
            name=f"X_pred[{i}]",
        )

    return X_true.copy(), X_pred


def predict_trajectories(model, A_TD, eval_data, n, m, steps, vmin, vmax):
    """
    Returns canonical arrays:
      X_true:    [N_eval, d, K]
      X_pred_TD: [N_eval, d, K]
      X_pred_TF: [N_eval, d, K]
    """
    X_true = require_NdK(eval_data, name="eval_data")

    N_eval, d, K_data = X_true.shape
    K = int(steps)

    if K != K_data:
        raise ValueError(
            f"steps must match eval_data time length. "
            f"Got steps={K}, eval_data.shape[-1]={K_data}."
        )

    X_pred_TD = np.empty((N_eval, d, K), dtype=float)
    X_pred_TF = np.empty((N_eval, d, K), dtype=float)

    for i in range(N_eval):
        x0_TD = X_true[i, :, : n + m - 1]
        x0_TF = X_true[i, :, :n]

        pred_TD_raw = reconstruct_TDDMD(
            A_TD,
            x0_TD,
            vmin=vmin,
            vmax=vmax,
            steps=K,
        )
        pred_TF_raw = model.AR_sliding(
            x0_TF,
            vmin=vmin,
            vmax=vmax,
            steps=K,
        )

        X_pred_TD[i] = prediction_to_dK(
            pred_TD_raw,
            d=d,
            K=K,
            name=f"X_pred_TD[{i}]",
        )
        X_pred_TF[i] = prediction_to_dK(
            pred_TF_raw,
            d=d,
            K=K,
            name=f"X_pred_TF[{i}]",
        )

    return X_true.copy(), X_pred_TD, X_pred_TF


def prediction_path(
    case_data: CaseData,
    cfg: ExperimentConfig,
    model_name: str,
    n: int,
    h,
    seed,
) -> Path:
    output_root = cfg.output_root or case_data.name
    out_dir = Path(output_root) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    h_tag = "none" if pd.isna(h) else str(int(h))
    seed_tag = "none" if pd.isna(seed) else str(int(seed))

    return out_dir / (
        f"{case_data.name}"
        f"_{model_name}"
        f"_n={n}"
        f"_h={h_tag}"
        f"_m={cfg.m}"
        f"_J={cfg.J}"
        f"_seed={seed_tag}.npz"
    )


def save_model_prediction_file(
    path: Path,
    case_data: CaseData,
    cfg: ExperimentConfig,
    model_name: str,
    n: int,
    h,
    seed,
    X_true: np.ndarray,
    X_pred: np.ndarray,
    vmin: np.ndarray,
    vmax: np.ndarray,
) -> None:
    X_true = require_NdK(X_true, name="X_true")
    X_pred = require_NdK(X_pred, name="X_pred")
    T = require_T(case_data.T, K=X_true.shape[-1])

    if X_pred.shape != X_true.shape:
        raise ValueError(f"X_pred shape {X_pred.shape} != X_true shape {X_true.shape}")

    metadata = {
        "format_version": 2,
        "case": case_data.name,
        "model": model_name,
        "plot_kind": case_data.plot_kind,
        "ylim": case_data.ylim,
        "n": int(n),
        "h": None if pd.isna(h) else int(h),
        "seed": None if pd.isna(seed) else int(seed),
        "N_eval": int(X_true.shape[0]),
        "d": int(X_true.shape[1]),
        "K": int(X_true.shape[2]),
        "config": asdict(cfg),
        "case_metadata": case_data.metadata,
        "shapes": {
            "T": list(T.shape),
            "X_true": list(X_true.shape),
            "X_pred": list(X_pred.shape),
        },
    }

    np.savez_compressed(
        path,
        T=T,
        X_true=X_true,
        X_pred=X_pred,
        vmin=np.asarray(vmin),
        vmax=np.asarray(vmax),
        metadata=json.dumps(metadata),
    )


def compute_rmse_rows(case_name, model_name, n, h, seed, X_true, X_pred):
    rows = []

    for i in range(X_true.shape[0]):
        rows.append({
            "case": case_name,
            "model": model_name,
            "n": n,
            "h": h,
            "seed": seed,
            "traj_index": i,
            "rmse": rmse(X_true[i], X_pred[i]),
        })

    return rows


def run_experiment(case_data: CaseData, cfg: ExperimentConfig) -> pd.DataFrame:
    case_data.train_data, case_data.eval_data, case_data.T = check_case_data(
        case_data.train_data,
        case_data.eval_data,
        case_data.T,
    )

    output_root = Path(cfg.output_root or case_data.name)
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for n in cfg.n_values:
        artifacts = make_training_artifacts(case_data, cfg, n)
        K = case_data.eval_data.shape[-1]

        # ------------------------------------------------------------
        # TD-DMD baseline: deterministic, computed once per n.
        # ------------------------------------------------------------
        X_true, X_pred_TD = predict_tddmd(
            A_TD=artifacts["A_TD"],
            eval_data=case_data.eval_data,
            n=n,
            m=cfg.m,
            steps=K,
            vmin=artifacts["vmin"],
            vmax=artifacts["vmax"],
        )

        all_rows.extend(
            compute_rmse_rows(
                case_data.name,
                "tddmd",
                n,
                h=np.nan,
                seed=np.nan,
                X_true=X_true,
                X_pred=X_pred_TD,
            )
        )

        if cfg.save_predictions:
            path = prediction_path(
                case_data=case_data,
                cfg=cfg,
                model_name="tddmd",
                n=n,
                h=np.nan,
                seed=np.nan,
            )
            save_model_prediction_file(
                path=path,
                case_data=case_data,
                cfg=cfg,
                model_name="tddmd",
                n=n,
                h=np.nan,
                seed=np.nan,
                X_true=X_true,
                X_pred=X_pred_TD,
                vmin=artifacts["vmin"],
                vmax=artifacts["vmax"],
            )
            print(f"[saved predictions] {path}")

        # ------------------------------------------------------------
        # Neural baselines: run every requested model / h / seed.
        # ------------------------------------------------------------
        for model_key in cfg.models:
            if model_key not in MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown model {model_key!r}. "
                    f"Available models are: {list(MODEL_REGISTRY)}"
                )

            model_class = MODEL_REGISTRY[model_key]

            for h in cfg.h_values:
                for seed in cfg.seeds:
                    set_all_seeds(seed)

                    model_kwargs = {
                        "d": case_data.train_data.shape[1],
                        "h": h,
                        "pos_encoding": cfg.pos_encoding,
                    }

                    # Models with fixed-length delay readouts need n.
                    if model_key == "delay_mlp":
                        model_kwargs["n"] = n

                    model_name = f"{case_data.name}_{model_key}_n={n}_h={h}_seed={seed}"

                    model = train_model(
                        artifacts["train_dataset"],
                        model_class=model_class,
                        model_kwargs=model_kwargs,
                        multistep=cfg.multistep,
                        lr=cfg.lr,
                        weight_decay=cfg.weight_decay,
                        batch_size=cfg.batch_size,
                        epochs=cfg.epochs,
                        model_name=model_name,
                    )

                    _, X_pred = predict_neural_model(
                        model=model,
                        eval_data=case_data.eval_data,
                        n=n,
                        steps=K,
                        vmin=artifacts["vmin"],
                        vmax=artifacts["vmax"],
                    )

                    all_rows.extend(
                        compute_rmse_rows(
                            case_data.name,
                            model_key,
                            n,
                            h,
                            seed,
                            X_true,
                            X_pred,
                        )
                    )

                    if cfg.save_predictions:
                        path = prediction_path(
                            case_data=case_data,
                            cfg=cfg,
                            model_name=model_key,
                            n=n,
                            h=h,
                            seed=seed,
                        )
                        save_model_prediction_file(
                            path=path,
                            case_data=case_data,
                            cfg=cfg,
                            model_name=model_key,
                            n=n,
                            h=h,
                            seed=seed,
                            X_true=X_true,
                            X_pred=X_pred,
                            vmin=artifacts["vmin"],
                            vmax=artifacts["vmax"],
                        )
                        print(f"[saved predictions] {path}")

                    if cfg.save_checkpoints:
                        ckpt_dir = output_root / "checkpoints"
                        ckpt_dir.mkdir(parents=True, exist_ok=True)

                        model.save_state_dict(
                            str(ckpt_dir / f"{model_name}.pth")
                        )

    df = pd.DataFrame(all_rows)
    raw_path = results_dir / f"{case_data.name}_raw_metrics.csv"
    df.to_csv(raw_path, index=False)

    agg = (
        df.groupby(["case", "model", "n", "h"], dropna=False, as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
        )
    )

    agg_path = results_dir / f"{case_data.name}_agg_metrics.csv"
    agg.to_csv(agg_path, index=False)

    # Optional aggregate heatmap for the NLR extrapolation variant.
    case_metadata = case_data.metadata or {}
    if (
        cfg.make_plots
        and case_metadata.get("case") == "nlr"
        and case_metadata.get("variant") == "extrapolation"
    ):
        for model_key in cfg.models:
            plot_nlr_extrapolation_heatmap(
                agg_path,
                qoi=case_metadata["qoi"],
                model=model_key,
                metric="rmse_mean",
                save_dir=output_root / "plots",
                show=False,
            )

    print(f"[saved raw metrics] {raw_path}")
    print(f"[saved aggregated metrics] {agg_path}")

    return agg
