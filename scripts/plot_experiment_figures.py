from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from src.experiments.arrays import require_NdK, require_T

MODEL_LABELS = {
    "tddmd": "TD-DMD",
    "tdtf": "TD-TF",
    "delay_mlp": "Delay-MLP",
}
MODEL_ALIASES = {
    "td-dmd": "tddmd", "td_dmd": "tddmd", "tddmd": "tddmd",
    "tdtf": "tdtf", "td-tf": "tdtf", "td_tf": "tdtf",
    "delay_mlp": "delay_mlp", "delay-mlp": "delay_mlp", "delay mlp": "delay_mlp",
}

MODEL_ALIASES = {
    "td-dmd": "tddmd",
    "td_dmd": "tddmd",
    "tddmd": "tddmd",
    "tdtf": "tdtf",
    "td-tf": "tdtf",
    "td_tf": "tdtf",
    "delay_mlp": "delay_mlp",
    "delay-mlp": "delay_mlp",
    "delay mlp": "delay_mlp",
}

MODEL_STYLES = {
    "tddmd": dict(color="tab:orange", linestyle="--", lw=2.5),
    "tdtf": dict(color="tab:blue", linestyle="--", lw=2.5),
    "delay_mlp": dict(color="tab:red", linestyle="--", lw=2.5),
}

MODEL_ALPHAS = {
    "tddmd": 1,
    "tdtf": 1,
    "delay_mlp": 1, #0.2
}


def normalize_model_name(model: str) -> str:
    key = model.strip().lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    raise ValueError(
        f"Unknown model={model!r}. Expected one of: "
        f"{', '.join(sorted(MODEL_ALIASES))}."
    )


def load_model_prediction(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    z = np.load(path, allow_pickle=False)
    metadata = json.loads(str(z["metadata"]))

    T = require_T(z["T"], name="T")
    X_true = require_NdK(z["X_true"], name="X_true")
    X_pred = require_NdK(z["X_pred"], name="X_pred")

    if X_pred.shape != X_true.shape:
        raise ValueError(
            f"X_pred shape {X_pred.shape} does not match X_true shape {X_true.shape} in {path}."
        )

    model_name = normalize_model_name(metadata["model"])
    return metadata, T, X_true, model_name, X_pred


def default_ylim(metadata: dict):
    case_metadata = metadata.get("case_metadata", {}) or {}
    ylim = case_metadata.get("ylim", None)
    if ylim is None:
        ylim = metadata.get("ylim", None)
    if ylim is not None:
        return ylim

    case = metadata.get("case", "")
    if case == "sinus":
        return (-1.25, 1.35)
    if case == "lorenz63":
        return (-20, 20)
    if case == "nlr_drag":
        return (-0.06, 0.02)
    if case == "nlr_lift":
        return (-0.25, 1.25)
    return None


def is_reaction_diffusion_case(metadata: dict) -> bool:
    """Return True for reaction-diffusion/reacDiff metadata variants."""
    case_metadata = metadata.get("case_metadata", {}) or {}
    fields = [
        metadata.get("case", ""),
        metadata.get("dataset", ""),
        metadata.get("system", ""),
        metadata.get("name", ""),
        metadata.get("qoi", ""),
        case_metadata.get("case", ""),
        case_metadata.get("dataset", ""),
        case_metadata.get("system", ""),
        case_metadata.get("name", ""),
        case_metadata.get("qoi", ""),
    ]
    text = " ".join(str(value) for value in fields if value is not None).lower()
    text = text.replace("-", "_").replace(" ", "_")

    return (
        "reacdiff" in text
        or "reactiondiffusion" in text
        or "reaction_diffusion" in text
        or ("reac" in text and "diff" in text)
    )


def plot_scalar_many_models(
    *,
    T,
    X_true,
    predictions,
    n,
    save_path,
    traj_index=0,
    all_trajectories=False,
    show=False,
    ylim=None,
    xlim=None,
    ground_truth_ylim=False,
):
    T = np.asarray(T).reshape(-1)

    if X_true.shape[1] != 1:
        raise ValueError(f"Expected scalar data with d=1, got X_true shape {X_true.shape}")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        length=7,
        width=2.0,
        labelsize=20,
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(2.0)

    if all_trajectories:
        traj_ids = range(X_true.shape[0])
    else:
        if traj_index < 0 or traj_index >= X_true.shape[0]:
            raise IndexError(f"--traj-index {traj_index} invalid for N_eval={X_true.shape[0]}.")
        traj_ids = [traj_index]

    if ground_truth_ylim and ylim is None:
        ylim = _auto_scalar_ylim(X_true, traj_ids)

    for j in traj_ids:
        ax.plot(T, X_true[j, 0, :], color="k", lw=3.2, alpha=0.8, zorder=1)
        ax.plot(
            T[:n],
            X_true[j, 0, :n],
            "o",
            color="tab:green",
            markersize=7,
            alpha=0.6,
            zorder=4,
        )

    for model_name, X_pred in predictions.items():
        style = MODEL_STYLES.get(model_name, dict(linestyle="--", lw=2.5))
        for j in traj_ids:
            ax.plot(
                T,
                X_pred[j, 0, :],
                alpha=MODEL_ALPHAS.get(model_name, 0.85),
                zorder=3,
                **style,
            )

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel(r"$t$", fontsize=22, labelpad=12)
    ax.set_ylabel(r"$\mathbf{w}$", fontsize=22, labelpad=12)

    handles = [
        Line2D([0], [0], color="k", lw=3.2, label="Ground truth"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="tab:green",
            markersize=7,
            label=rf"$\mathbf{{w}}_0$ $(n={n})$",
        ),
    ]

    for model_name in predictions:
        style = MODEL_STYLES.get(model_name, dict(linestyle="--", lw=2.5))
        handles.append(
            Line2D(
                [0],
                [0],
                label=MODEL_LABELS.get(model_name, model_name),
                alpha=MODEL_ALPHAS.get(model_name, 0.85),
                **style,
            )
        )

    # fig.legend(
    #     handles=handles,
    #     loc="upper center",
    #     fontsize=16,
    #     ncol=len(handles),
    #     frameon=True,
    #     framealpha=0.85,
    #     facecolor="white",
    #     edgecolor="gray",
    #     columnspacing=1.0,
    #     handletextpad=0.5,
    #     borderaxespad=0.2,
    #     bbox_to_anchor=(0.6, 1.15),
    #     bbox_transform=fig.transFigure,
    # )

    plt.tight_layout()
    save_path = Path(save_path).with_suffix(".pdf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"[saved plot] {save_path}")


# def _auto_coordinate_ylims(arrays, ylim_pad=0.1):
#     d = arrays[0].shape[0]
#     ylims = []
#     for k in range(d):
#         vals = np.concatenate([A[k, :][np.isfinite(A[k, :])] for A in arrays])
#         if vals.size == 0:
#             ymin, ymax = -1.0, 1.0
#         else:
#             ymin = np.min(vals)
#             ymax = np.max(vals)
#             if np.isclose(ymin, ymax):
#                 pad = 1.0 if np.isclose(ymin, 0.0) else abs(ymin) * ylim_pad
#             else:
#                 pad = (ymax - ymin) * ylim_pad
#             ymin -= pad
#             ymax += pad
#         ylims.append((ymin, ymax))
#     return ylims


def _auto_coordinate_ylims(X_true, ylim_pad=0.12, min_pad=1e-3):
    """
    Per-coordinate y-limits based only on the ground-truth trajectory.

    X_true shape: [d, K]
    """
    X_true = np.asarray(X_true)
    d = X_true.shape[0]

    ylims = []
    for k in range(d):
        vals = X_true[k, :]
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            ymin, ymax = -1.0, 1.0
        else:
            ymin = float(np.min(vals))
            ymax = float(np.max(vals))

            yrange = ymax - ymin
            if np.isclose(yrange, 0.0):
                pad = max(abs(ymin) * ylim_pad, min_pad)
            else:
                pad = max(yrange * ylim_pad, min_pad)

            ymin -= pad
            ymax += pad

        ylims.append((ymin, ymax))

    return ylims


def _auto_scalar_ylim(X_true, traj_ids, ylim_pad=0.12, min_pad=1e-3):
    """Scalar y-limits based only on the selected ground-truth trajectory/trajectories."""
    vals = []
    for j in traj_ids:
        v = np.asarray(X_true[j, 0, :])
        vals.append(v[np.isfinite(v)])

    vals = np.concatenate(vals) if vals else np.asarray([])
    if vals.size == 0:
        return (-1.0, 1.0)

    ymin = float(np.min(vals))
    ymax = float(np.max(vals))
    yrange = ymax - ymin
    if np.isclose(yrange, 0.0):
        pad = max(abs(ymin) * ylim_pad, min_pad)
    else:
        pad = max(yrange * ylim_pad, min_pad)

    return (ymin - pad, ymax + pad)


def plot_stacked_many_models(
    *,
    T,
    X_true,
    predictions,
    n,
    save_path,
    traj_index=0,
    show=False,
    ylim=None,
    xlim=None,
    coord_labels=None,
    layout="wide",
):
    T = np.asarray(T).reshape(-1)

    if traj_index < 0 or traj_index >= X_true.shape[0]:
        raise IndexError(f"--traj-index {traj_index} invalid for N_eval={X_true.shape[0]}.")

    X0 = np.asarray(X_true[traj_index])
    pred0 = {name: np.asarray(X_pred[traj_index]) for name, X_pred in predictions.items()}

    d, K = X0.shape
    if K != T.shape[0]:
        raise ValueError(f"Time grid length {T.shape[0]} does not match trajectory length {K}.")

    for name, X_pred in pred0.items():
        if X_pred.shape != X0.shape:
            raise ValueError(f"Prediction for {name} has shape {X_pred.shape}; expected {X0.shape}.")

    # if ylim is None:
    #     ylims = _auto_coordinate_ylims(X0)
    # elif (
    #     isinstance(ylim, (list, tuple))
    #     and len(ylim) == 2
    #     and np.isscalar(ylim[0])
    #     and np.isscalar(ylim[1])
    # ):
    #     ylims = [tuple(ylim)] * d
    # else:
    #     if len(ylim) != d:
    #         raise ValueError("If providing per-coordinate ylim, it must have length d.")
    #     ylims = list(ylim)


    if ylim is None:
        ylims = _auto_coordinate_ylims(X0)
    elif (
        isinstance(ylim, (list, tuple))
        and len(ylim) == 2
        and np.isscalar(ylim[0])
        and np.isscalar(ylim[1])
    ):
        ylims = [tuple(ylim)] * d
    else:
        if len(ylim) != d:
            raise ValueError("If providing per-coordinate ylim, it must have length d.")
        ylims = list(ylim)

    if coord_labels is None:
        coord_labels = [rf"$w_{{{j + 1}}}$" for j in range(d)]

    if len(coord_labels) != d:
        raise ValueError("coord_labels must be None or have length d.")

    figsize = (3.5, 1.55 * d + 0.15) if layout == "narrow" else (8.0, 1.35 * d + 0.15)
    fig, axes = plt.subplots(d, 1, sharex=True, figsize=figsize, gridspec_kw=dict(hspace=0.12))
    if d == 1:
        axes = [axes]

    if layout == "wide":
        ylab_x = -0.10
        for ax in axes:
            ax.yaxis.set_label_coords(ylab_x, 0.5)
        fig.align_ylabels(axes)

    for k, ax in enumerate(axes):
        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            length=6,
            width=1.8,
            labelsize=18,
        )

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_linewidth(1.8)

        ax.plot(T, X0[k, :], color="k", lw=3.0, label="Ground truth", zorder=1)
        ax.plot(T[:n], X0[k, :n], "o", color="tab:green", markersize=7, zorder=4)

        for model_name, X_pred in pred0.items():
            style = MODEL_STYLES.get(model_name, dict(linestyle="--", lw=2.5))
            ax.plot(
                T,
                X_pred[k, :],
                alpha=MODEL_ALPHAS.get(model_name, 0.85),
                zorder=3,
                label=MODEL_LABELS.get(model_name, model_name),
                **style,
            )

        ax.set_ylim(ylims[k])
        ax.set_ylabel(coord_labels[k], fontsize=20, labelpad=10)

        if layout == "wide":
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(axis="y", pad=2)

        if k == d - 1:
            ax.set_xlabel(r"$t$", fontsize=20, labelpad=12)

    if xlim is not None:
        axes[-1].set_xlim(xlim)

    handles = [
        Line2D([0], [0], color="k", lw=3.0, label="Ground truth"),
        Line2D([0], [0], marker="o", linestyle="None", color="tab:green", markersize=7, label=rf"$\mathbf{{w}}_0$ $(n={n})$"),
    ]
    for model_name in predictions:
        style = MODEL_STYLES.get(model_name, dict(linestyle="--", lw=2.5))
        handles.append(
            Line2D(
                [0],
                [0],
                label=MODEL_LABELS.get(model_name, model_name),
                alpha=MODEL_ALPHAS.get(model_name, 0.85),
                **style,
            )
        )

    # fig.legend(
    #     handles=handles,
    #     loc="upper center",
    #     fontsize=14,
    #     ncol=len(handles),
    #     frameon=True,
    #     framealpha=0.85,
    #     facecolor="white",
    #     edgecolor="gray",
    #     columnspacing=0.9,
    #     handletextpad=0.5,
    #     bbox_to_anchor=(0.55, 1.04),
    #     bbox_transform=fig.transFigure,
    # )

    save_path = Path(save_path).with_suffix(".pdf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=600)

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"[saved plot] {save_path}")


def plot_trajectory(args):
    T_ref = None
    X_true_ref = None
    n_ref = None
    metadata_ref = None
    predictions = {}

    for path in args.prediction_files:
        metadata, T, X_true, model_name, X_pred = load_model_prediction(path)

        if T_ref is None:
            T_ref = T
            X_true_ref = X_true
            n_ref = int(metadata["n"])
            metadata_ref = metadata
        else:
            if not np.allclose(T_ref, T):
                raise ValueError(f"Time grid mismatch in {path}")
            if X_true_ref.shape != X_true.shape:
                raise ValueError(f"X_true shape mismatch in {path}")
            if n_ref != int(metadata["n"]):
                raise ValueError(f"n mismatch in {path}")

        predictions[model_name] = X_pred

    plot_kind = metadata_ref.get("plot_kind")
    use_ground_truth_ylim = is_reaction_diffusion_case(metadata_ref)
    ylim = None if use_ground_truth_ylim else default_ylim(metadata_ref)
    case_metadata = metadata_ref.get("case_metadata", {}) or {}
    coord_labels = case_metadata.get("coord_labels", None)
    stack_layout = case_metadata.get("stack_layout", "wide")

    if plot_kind in {"scalar_single", "scalar_many"}:
        plot_scalar_many_models(
            T=T_ref,
            X_true=X_true_ref,
            predictions=predictions,
            n=n_ref,
            save_path=args.save_path,
            traj_index=args.traj_index,
            all_trajectories=args.all_trajectories,
            show=args.show,
            ylim=ylim,
            xlim=args.xlim,
            ground_truth_ylim=use_ground_truth_ylim,
        )
    elif plot_kind == "stacked_coordinates":
        plot_stacked_many_models(
            T=T_ref,
            X_true=X_true_ref,
            predictions=predictions,
            n=n_ref,
            save_path=args.save_path,
            traj_index=args.traj_index,
            show=args.show,
            ylim=ylim,
            xlim=args.xlim,
            coord_labels=coord_labels,
            layout=stack_layout,
        )
    else:
        raise ValueError(
            f"Unknown plot_kind={plot_kind!r}. Expected 'scalar_single', 'scalar_many', or 'stacked_coordinates'."
        )


def load_heatmap_data(csv_path: Path, model: str, metric: str, x: str, y: str, agg: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    model_key = normalize_model_name(model)
    model_label = MODEL_LABELS[model_key]

    if {"model", metric}.issubset(df.columns):
        dd = df[df["model"].map(lambda value: normalize_model_name(str(value))) == model_key].copy()
        metric_col = metric

        if dd.empty:
            raise ValueError(f"No rows for model={model!r} in {csv_path}.")

        if y in dd.columns and dd[y].isna().all():
            y_values = sorted(pd.to_numeric(df[y], errors="coerce").dropna().unique())
            if not y_values:
                raise ValueError(f"Cannot build a heatmap for {model_label}: column {y!r} has no finite values.")
            replicated = []
            for y_value in y_values:
                tmp = dd.copy()
                tmp[y] = y_value
                replicated.append(tmp)
            dd = pd.concat(replicated, ignore_index=True)
    else:
        legacy_cols = {
            "tddmd": f"TD_{metric}",
            "tdtf": f"TF_{metric}",
            "delay_mlp": f"DelayMLP_{metric}",
        }
        metric_col = legacy_cols[model_key]
        if metric_col not in df.columns:
            raise ValueError(
                f"Could not find long-format columns ['model', {metric!r}] or legacy column {metric_col!r} in {csv_path}. "
                f"Available columns are: {list(df.columns)}"
            )
        dd = df.copy()

    for col in (x, y, metric_col):
        if col not in dd.columns:
            raise ValueError(f"Column {col!r} not found in {csv_path}. Available columns are: {list(dd.columns)}")

    dd[x] = pd.to_numeric(dd[x], errors="coerce")
    dd[y] = pd.to_numeric(dd[y], errors="coerce")
    dd[metric_col] = pd.to_numeric(dd[metric_col], errors="coerce")
    dd = dd.dropna(subset=[x, y, metric_col])

    if dd.empty:
        raise ValueError(f"No finite heatmap values for model={model_label}, metric={metric} in {csv_path}.")

    if dd.duplicated(subset=[y, x]).any():
        dd = dd.groupby([y, x], as_index=False)[metric_col].agg(agg)

    grid = dd.pivot(index=y, columns=x, values=metric_col)
    grid = grid.sort_index(axis=0).sort_index(axis=1)
    return grid, model_label


def plot_heatmap(args):
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Aggregate metrics CSV not found: {csv_path}")

    save_dir = Path(args.save_dir) if args.save_dir else Path(args.output_root) / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    grid, model_label = load_heatmap_data(csv_path, args.model, args.metric, args.x, args.y, args.agg)

    vals = grid.values
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size == 0:
        raise ValueError(f"No finite heatmap values in {csv_path}")

    norm = Normalize(vmin=float(np.nanmin(finite_vals)), vmax=float(np.nanmax(finite_vals)))
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 1.8), constrained_layout=True)
    im = ax.imshow(
        vals,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=args.cmap,
        norm=norm,
    )

    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))

    xt = [int(v) if float(v).is_integer() else v for v in grid.columns]
    yt = [int(v) if float(v).is_integer() else v for v in grid.index]
    ax.set_xticklabels(xt)
    ax.set_yticklabels(yt)
    ax.set_xlabel(args.x)
    ax.set_ylabel(args.y)

    if args.annotate:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = vals[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)

    safe_model = model_label.replace(" ", "-")
    save_path = save_dir / f"heatmap_{safe_model}_{args.qoi}.pdf"
    fig.savefig(save_path)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print(f"[saved heatmap] {save_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate trajectory and heatmap figures from saved experiment files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    trajectory = subparsers.add_parser("trajectory")
    trajectory.add_argument("prediction_files", nargs="+")
    trajectory.add_argument("--save-path", required=True)
    trajectory.add_argument("--traj-index", type=int, default=0)
    trajectory.add_argument("--all-trajectories", action="store_true")
    trajectory.add_argument("--show", action="store_true")
    trajectory.add_argument("--xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"), default=None)
    trajectory.set_defaults(func=plot_trajectory)

    heatmap = subparsers.add_parser("heatmap")
    heatmap.add_argument("--csv", required=True)
    heatmap.add_argument("--qoi", choices=["lift", "drag"], required=True)
    heatmap.add_argument("--model", default="tdtf")
    heatmap.add_argument("--metric", choices=["rmse_mean", "rmse_std"], default="rmse_mean")
    heatmap.add_argument("--x", default="n")
    heatmap.add_argument("--y", default="h")
    heatmap.add_argument("--agg", default="mean")
    heatmap.add_argument("--cmap", default="viridis")
    heatmap.add_argument("--output-root", default="nlr")
    heatmap.add_argument("--save-dir", default=None)
    heatmap.add_argument("--annotate", action="store_true")
    heatmap.add_argument("--show", action="store_true")
    heatmap.set_defaults(func=plot_heatmap)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
