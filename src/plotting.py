import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from pathlib import Path
import pandas as pd
from matplotlib.colors import Normalize

def plot_predictions_1D_top(
    T, X_all, X_pred_TF, X_pred_TD, n,
    save_path=None, show=True, ylim=(-1.3, 1.3),
    TF_label="TD-TF", TD_label="TD-DMD",
    xlabel=r"$t$", ylabel=r"$\mathbf{w}$",
    tick_fontsize=18, label_fontsize=20, legend_fontsize=16
):
    """
    Plot the predictions of the model in one dimension with the legend on top.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Ticks and spines
    ax.tick_params(axis='both', which='both',
                   direction='in', length=6, width=1.8,
                   labelsize=tick_fontsize)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(1.8)

    # Ground truth
    line_gt = ax.plot(T, X_all, color='k', lw=3.0, label="Ground truth", zorder=1)

    # Initial condition points
    line_w0 = ax.plot(T[:n], X_all[:n], 'o', color='tab:green', markersize=7,
                       label=r"$\mathbf{w}_0$  $(n=" + str(n) + r")$", zorder=4)

    # Predictions
    lines_pred = []
    if X_pred_TF is not None:
        line_tf = ax.plot(T, X_pred_TF, "--", lw=2.5, color='tab:blue',
                          label=TF_label, zorder=3)
        lines_pred.append(line_tf)
    if X_pred_TD is not None:
        line_td = ax.plot(T, X_pred_TD, "--", lw=2.5, color='tab:orange',
                           label=TD_label, zorder=2)
        lines_pred.append(line_td)

    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=12)

    handles = [line_gt[0], line_w0[0]] + [l[0] for l in lines_pred]
    labels = [h.get_label() for h in handles]

    # fig.legend(
    #     handles, labels,
    #     loc='upper center',
    #     ncol=len(handles),
    #     fontsize=legend_fontsize,
    #     frameon=True,
    #     framealpha=0.8,
    #     facecolor='white',
    #     edgecolor='gray',
    #     bbox_to_anchor=(0.5, 1.08),
    #     columnspacing=0.7,
    #     handletextpad=0.6,
    #     borderaxespad=0.3
    # )

    plt.tight_layout()
    if save_path:
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
    if show:
        plt.show()

def _to_KM(arr, K_expected=None):
    """
    Normalize input to shape (K, M). Returns (None, 0) if arr is None.
    Accepts shapes (K,), (K, M), or (M, K).
    """
    if arr is None:
        return None, 0
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a[:, None]  # (K,) -> (K,1)
    elif a.ndim == 2:
        # Heuristic: if first dim != K_expected but second does, transpose
        if K_expected is not None:
            if a.shape[0] != K_expected and a.shape[1] == K_expected:
                a = a.T
        # otherwise keep as-is
    else:
        raise ValueError("Input must be 1D or 2D array.")
    return a, a.shape[1]

def plot_predictions_1D_top_multi(
    T, X_all, X_pred_TF, X_pred_TD, n,
    save_path=None, show=True, ylim=(-1.3, 1.3),
    TF_label="Delayed TF", TD_label="TD-DMD",
    xlabel=r"$t$", ylabel=r"$\mathbf{w}$",
    tick_fontsize=20, label_fontsize=22, legend_fontsize=18,
    lw_gt=3.2, lw_pred=3.0, ms_w0=7,
    alpha_traj=0.8, alpha_w0=0.6
):
    """
    Plot 1D trajectories with optional multiple samples per method.
    - T: (K,)
    - X_all: (K,), (K,M) or (M,K)
    - X_pred_TF: same accepted shapes or None
    - X_pred_TD: same accepted shapes or None
    - n: number of initial points to highlight as w0
    """

    T = np.asarray(T).ravel()
    K = T.shape[0]

    X_all, M_gt = _to_KM(X_all, K_expected=K)
    X_tf,  M_tf = _to_KM(X_pred_TF, K_expected=K) if X_pred_TF is not None else (None, 0)
    X_td,  M_td = _to_KM(X_pred_TD, K_expected=K) if X_pred_TD is not None else (None, 0)

    fig, ax = plt.subplots(figsize=(8, 5))  # narrower/taller for side-by-side in LaTeX

    # Axes styling
    ax.tick_params(axis='both', which='both', direction='in', length=7, width=2.0,
                   labelsize=tick_fontsize)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_linewidth(2.0)

    # === Plot all trajectories ===
    # Ground truth
    if M_gt > 0:
        for j in range(M_gt):
            ax.plot(T, X_all[:, j], color='k', lw=lw_gt, alpha=alpha_traj, zorder=1, label=None)
            # Highlight initial n points for each trajectory
            ax.plot(T[:n], X_all[:n, j], 'o', color='tab:green', markersize=ms_w0,
                    alpha=alpha_w0, zorder=4, label=None)

    # TF predictions
    if X_tf is not None and M_tf > 0:
        for j in range(M_tf):
            ax.plot(T, X_tf[:, j], "--", lw=lw_pred, color='tab:blue', alpha=alpha_traj,
                    zorder=3, label=None)

    # TD-DMD predictions
    if X_td is not None and M_td > 0:
        for j in range(M_td):
            ax.plot(T, X_td[:, j], "--", lw=lw_pred, color='tab:orange', alpha=alpha_traj,
                    zorder=2, label=None)

    # Labels/limits
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=label_fontsize, labelpad=12)

    # === Compact figure-level legend (single row if <=4 items) ===
    # legend_handles = [
    #     Line2D([0], [0], color='k', lw=lw_gt, label="Ground truth"),
    #     Line2D([0], [0], marker='o', linestyle='None', color='tab:green',
    #            markersize=ms_w0, label=rf"$\mathbf{{w}}_0$ $(n={n})$"),
    # ]
    # if X_tf is not None and M_tf > 0:
    #     legend_handles.append(Line2D([0], [0], linestyle='--', color='tab:blue', lw=lw_pred, label=TF_label))
    # if X_td is not None and M_td > 0:
    #     legend_handles.append(Line2D([0], [0], linestyle='--', color='tab:orange', lw=lw_pred, label=TD_label))
    
    # # === Compact figure-level legend (rectangular: 2 top, 2 bottom) ===
    # fig.legend(
    #     handles=legend_handles,
    #     loc="upper center",
    #     fontsize=legend_fontsize,
    #     ncol=len(legend_handles),                        # two columns → rectangular 2x2
    #     frameon=True,
    #     framealpha=0.85,
    #     facecolor="white",
    #     edgecolor="gray",
    #     columnspacing=1.0,             # reduce horizontal space between columns
    #     handletextpad=0.5,             # reduce space between handle and text
    #     borderaxespad=0.2,             # reduce space between legend and axes
    #     bbox_to_anchor=(0.6, 1.15),    # adjust vertical position
    #     bbox_transform=fig.transFigure
    # )

    plt.tight_layout()
    if save_path:
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
    if show:
        plt.show()


# def save_predictions_stack_legend(
#     save_path,
#     n,
#     TF_label="TD-TF",
#     TD_label="TD-DMD",
#     legend_fontsize=16,
#     spine_lw=1.8,
#     ncol=4,
#     figsize=(6.8, 0.55),
#     dpi=600,
# ):
#     """
#     Saves a standalone legend for plot_predictions_stack_top.

#     Produces a clean legend with no surrounding box/frame.
#     """
#     handles = [
#         Line2D([0], [0], color="k", lw=3.0, label="Ground truth"),
#         Line2D(
#             [0], [0],
#             marker="o",
#             linestyle="None",
#             color="tab:green",
#             markersize=7,
#             label=rf"$\mathbf{{w}}_0$", #  $(n={n})$",
#         ),
#         Line2D([0], [0], color="tab:blue", lw=2.5, linestyle="--", label=TF_label),
#         Line2D([0], [0], color="tab:orange", lw=2.5, linestyle="--", label=TD_label),
#     ]

#     labels = [h.get_label() for h in handles]

#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(111)
#     ax.axis("off")

#     fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
#     plt.close(fig)



from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


def save_predictions_stack_legend(
    save_path,
    baseline_models=None,
    legend_fontsize=16,
    spine_lw=1.8,
    ncol=None,
    figsize=None,
    default_alpha=1.0,
    gt_alpha=1.0,
    w0_alpha=1.0,
    dpi=600,
):
    """
    Saves a standalone legend for stacked prediction plots.

    Parameters
    ----------
    save_path : str or Path
        Output path.
    baseline_models : list[dict] or None
        Models to include after the fixed ground-truth entries.

        Example:
            baseline_models=[
                {
                    "label": "TD-TF",
                    "color": "tab:blue",
                    "linestyle": "--",
                    "alpha": 0.85,
                },
                {
                    "label": "TD-DMD",
                    "color": "tab:orange",
                    "linestyle": "--",
                    "alpha": 0.85,
                },
                {
                    "label": "Delay-MLP",
                    "color": "tab:red",
                    "linestyle": "--",
                    "alpha": 0.35,
                },
            ]

    default_alpha : float
        Alpha used for baseline models that do not specify one.
    gt_alpha : float
        Alpha for the ground-truth legend handle.
    w0_alpha : float
        Alpha for the initial-condition marker legend handle.
    dpi : int
        Output resolution.
    """
    if baseline_models is None:
        baseline_models = [
            {
                "label": "TD-TF",
                "color": "tab:blue",
                "linestyle": "--",
                "alpha": 0.85,
            },
            {
                "label": "TD-DMD",
                "color": "tab:orange",
                "linestyle": "--",
                "alpha": 0.85,
            },
        ]

    handles = [
        Line2D(
            [0], [0],
            color="k",
            lw=3.0,
            alpha=gt_alpha,
            label="Ground truth",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            color="tab:green",
            markersize=7,
            alpha=w0_alpha,
            label=rf"$\mathbf{{w}}_0$",
        ),
    ]

    for model in baseline_models:
        handles.append(
            Line2D(
                [0], [0],
                color=model.get("color", "k"),
                lw=model.get("lw", 2.5),
                linestyle=model.get("linestyle", "--"),
                alpha=model.get("alpha", default_alpha),
                label=model["label"],
            )
        )

    labels = [h.get_label() for h in handles]

    if ncol is None:
        ncol = len(handles)

    if figsize is None:
        figsize = (1.55 * len(handles), 0.55)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.legend(
        handles,
        labels,
        loc="center",
        ncol=ncol,
        fontsize=legend_fontsize,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.55,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    plt.close(fig)

def plot_predictions_stack_top(
    T, X_all, X_pred_TF, X_pred_TD, n,
    save_path=None, show=True, ylim=None,
    TF_label="TD-TF", TD_label="TD-DMD",
    xlabel=r"$t$", ylabel=r"$\mathbf{w}$",
    tick_fontsize=18, label_fontsize=20,
    coord_labels=None,
    layout="wide",
    figsize=None,
    hspace=0.1,
    spine_lw=1.8,
    dpi=600,
    ylim_pad=0.08,
):
    """
    Stack d coordinates vertically.

    If ylim is None, y-limits are computed from min/max values of all
    plotted series for each coordinate, with fractional padding ylim_pad.

    Accepts X_* in shape (N,d) or (d,N), auto-transposing if needed.
    """
    T = np.asarray(T).reshape(-1)
    N_T = T.shape[0]

    def _to_Nd(X, name):
        if X is None:
            return None

        X = np.asarray(X)

        if X.ndim == 1:
            if X.shape[0] == N_T:
                return X[:, None]
            raise ValueError(
                f"{name} is 1D with length {X.shape[0]}, but len(T)={N_T}. "
                f"Provide {name} as a time series of length len(T)."
            )

        if X.ndim != 2:
            raise ValueError(f"{name} must be 1D or 2D, got shape {X.shape}.")

        r, c = X.shape

        if r == N_T:
            return X

        if c == N_T:
            return X.T

        raise ValueError(
            f"{name} has shape {X.shape} but len(T)={N_T}. "
            f"One axis must equal len(T)."
        )

    X_all = _to_Nd(X_all, "X_all")
    X_pred_TF = _to_Nd(X_pred_TF, "X_pred_TF")
    X_pred_TD = _to_Nd(X_pred_TD, "X_pred_TD")

    N, d = X_all.shape

    if X_pred_TF is not None and X_pred_TF.shape[1] != d:
        raise ValueError(
            f"X_pred_TF must have the same number of coordinates as X_all (d={d})."
        )

    if X_pred_TD is not None and X_pred_TD.shape[1] != d:
        raise ValueError(
            f"X_pred_TD must have the same number of coordinates as X_all (d={d})."
        )

    def _auto_ylims():
        ylims_auto = []

        arrays = [X_all]
        if X_pred_TF is not None:
            arrays.append(X_pred_TF)
        if X_pred_TD is not None:
            arrays.append(X_pred_TD)

        for k in range(d):
            vals = np.concatenate([
                A[:, k][np.isfinite(A[:, k])]
                for A in arrays
            ])

            if vals.size == 0:
                ymin, ymax = -1.0, 1.0
            else:
                ymin = np.min(vals)
                ymax = np.max(vals)

                if np.isclose(ymin, ymax):
                    pad = 1.0 if np.isclose(ymin, 0.0) else abs(ymin) * ylim_pad
                else:
                    pad = (ymax - ymin) * ylim_pad

                ymin -= pad
                ymax += pad

            ylims_auto.append((ymin, ymax))

        return ylims_auto

    if ylim is None:
        ylims = _auto_ylims()
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
        coord_labels = [rf"{ylabel}_{{{k+1}}}" for k in range(d)]

    if len(coord_labels) != d:
        raise ValueError("coord_labels must be None or have length d.")

    if figsize is None:
        if layout == "narrow":
            figsize = (3.5, 1.55 * d + 0.15)
        else:
            figsize = (8.0, 1.35 * d + 0.15)

    fig, axes = plt.subplots(
        d,
        1,
        sharex=True,
        figsize=figsize,
        gridspec_kw=dict(hspace=hspace),
    )

    if d == 1:
        axes = [axes]

    if layout == "wide":
        if hspace < 0.09:
            fig.subplots_adjust(hspace=0.12)

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
            width=spine_lw,
            labelsize=tick_fontsize,
        )

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        for spine in ["bottom", "left"]:
            ax.spines[spine].set_linewidth(spine_lw)

        ax.plot(
            T,
            X_all[:, k],
            color="k",
            lw=3.0,
            label="Ground truth",
            zorder=1,
        )

        ax.plot(
            T[:n],
            X_all[:n, k],
            "o",
            color="tab:green",
            markersize=7,
            # label=rf"$\mathbf{{w}}_0$ $(n={n})$",
            label=rf"$\mathbf{{w}}_0$",
            zorder=4,
        )

        if X_pred_TF is not None:
            ax.plot(
                T,
                X_pred_TF[:, k],
                "--",
                lw=2.5,
                color="tab:blue",
                label=TF_label,
                zorder=3,
            )

        if X_pred_TD is not None:
            ax.plot(
                T,
                X_pred_TD[:, k],
                "--",
                lw=2.5,
                color="tab:orange",
                label=TD_label,
                zorder=2,
            )

        ax.set_ylim(ylims[k])
        ax.set_ylabel(coord_labels[k], fontsize=label_fontsize, labelpad=10)

        if layout == "wide":
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(axis="y", pad=2)

        if k == d - 1:
            ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=12)

    # fig.tight_layout(pad=0.25)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_nlr_extrapolation_heatmap(
    agg_csv,
    *,
    qoi,
    model="TD-TF",
    metric="rmse_mean",
    x="n",
    y="h",
    agg="mean",
    cmap="viridis",
    annotate=False,
    save_dir="nlr/plots",
    show=False,
):
    """
    Reproduce the heatmap style from nlrExtrapolationPlot.py using the
    new pipeline aggregate CSV.

    Expected aggregate CSV columns:
      case,n,h,TD_rmse_mean,TD_rmse_std,TF_rmse_mean,TF_rmse_std
    """
    agg_csv = Path(agg_csv)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(agg_csv)

    model_to_prefix = {
        "TD-DMD": "TD",
        "TD-TF": "TF",
    }

    if model not in model_to_prefix:
        raise ValueError(f"Unknown model={model!r}. Expected 'TD-DMD' or 'TD-TF'.")

    metric_col = f"{model_to_prefix[model]}_{metric}"

    if metric_col not in df.columns:
        raise ValueError(
            f"Column {metric_col!r} not found in {agg_csv}. "
            f"Available columns are: {list(df.columns)}"
        )

    dd = df.copy()

    if dd.duplicated(subset=[y, x]).any():
        dd = dd.groupby([y, x], as_index=False)[metric_col].agg(agg)

    grid = dd.pivot(index=y, columns=x, values=metric_col)

    try:
        grid.index = grid.index.astype(float)
    except Exception:
        pass

    try:
        grid.columns = grid.columns.astype(float)
    except Exception:
        pass

    grid = grid.sort_index(axis=0).sort_index(axis=1)

    vals = grid.values
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 1.8), constrained_layout=True)

    im = ax.imshow(
        grid.values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )

    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))

    xt = [int(v) if float(v).is_integer() else v for v in grid.columns]
    yt = [int(v) if float(v).is_integer() else v for v in grid.index]

    ax.set_xticklabels(xt)
    ax.set_yticklabels(yt)

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if annotate:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)

    safe_model = model.replace("DMD", "DMD").replace("TF", "TF")
    save_path = save_dir / f"heatmap_{safe_model}_{qoi}.pdf"
    fig.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"[saved heatmap] {save_path}")

    return fig, ax, grid