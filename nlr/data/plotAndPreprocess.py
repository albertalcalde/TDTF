import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import patheffects as pe
import os


filename = "nlr/data/nlrData_allAlphas.npy" 
X_all = np.load(filename)
X_all = X_all.transpose(0,2,1)  #  add first dimension if it's 1

thistime = X_all[0,-1, :]

endAlpha = 16 #12
alphas = X_all[:endAlpha, 0, 0]  # Extract alphas from the first row until alpha < 10

QoIs = [1, 4] # indices for 'C-drag', 'C-lift'

X_all = X_all[:endAlpha,QoIs,:]  # Select only the specified quantities indices and alpha \leq 10
print(f"X_all shape: {X_all.shape}")


def plot_qois_over_time(
    thistime,
    X_all,
    alphas,
    labels=('C-drag', 'C-lift'),
    cmap_names=('tab20', 'tab20'),
    save_dir=None,
    show_colorbar=True,
    dashed_indices=None,
    xlabel=r"$t$",
    tick_fontsize=18,
    label_fontsize=20,
    figsize=(8, 5),
    ylim=None,
    cbar_tick_format="{:.2f}"
):
    """
    One figure per QoI. For each figure:
    - Colored lines for all trajectories.
    - Highlighted dashed_indices get a black-edged dashed style
      (color preserved from colormap).
    """
    if dashed_indices is None:
        dashed_indices = []

    num_alphas = X_all.shape[0]
    num_quant  = X_all.shape[1]
    alpha_labels = [cbar_tick_format.format(a) for a in alphas]

    for quant_idx in range(num_quant):
        fig, ax = plt.subplots(figsize=figsize)

        ax.tick_params(axis='both', which='both',
                       direction='in', length=6, width=1.8,
                       labelsize=tick_fontsize)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1.8)

        base_name = cmap_names[quant_idx % len(cmap_names)]
        cmap = plt.get_cmap(base_name, num_alphas)
        boundaries = np.arange(num_alphas + 1) - 0.5
        norm = colors.BoundaryNorm(boundaries, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for alpha_idx in range(num_alphas):
            color = cmap(alpha_idx / max(1, num_alphas - 1))
            y = X_all[alpha_idx, quant_idx, :]

            # keep one plot call; add a black outline via path effects
            dash_pattern = (6, 4)  # optional: customize as you like

            if alpha_idx in dashed_indices:
                line, = ax.plot(
                    thistime, y,
                    linestyle='--',
                    dashes=dash_pattern,   # ensures consistent dash pattern
                    color=color,           # keep the colormap color
                    lw=2.5,
                    alpha=1.0,
                    zorder=5
                )
                line.set_path_effects([
                    pe.Stroke(linewidth=5.5, foreground='black'),  # the "edgecolor"
                    pe.Normal(),
                ])

            else:
                ax.plot(thistime, y, color=color, lw=2.0, linestyle='-', alpha=0.9, zorder=1)

        ax.set_xlabel(xlabel, fontsize=label_fontsize, labelpad=12)
        ax.set_ylabel(labels[quant_idx], fontsize=label_fontsize, labelpad=12)
        if ylim is not None:
            ax.set_ylim(ylim)

        if show_colorbar:
            cbar = fig.colorbar(sm, ax=ax, pad=0.02,
                                boundaries=boundaries, ticks=np.arange(num_alphas))
            cbar.set_label(r'$\alpha$', fontsize=tick_fontsize)
            cbar.ax.tick_params(labelsize=tick_fontsize-2)
            
                # Show only integer tick labels
            int_ticks = np.arange(num_alphas)
            int_labels = [str(int(a)) for a in alphas]  # force integers
            cbar.set_ticks(int_ticks)
            cbar.set_ticklabels(int_labels)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            safe_label = str(labels[quant_idx]).replace(" ", "_")
            pdf_path = os.path.join(save_dir, f"nlr_{safe_label}.pdf")
            plt.savefig(pdf_path, bbox_inches='tight')

        plt.show()


# Example usage:
plot_qois_over_time(
    thistime=thistime,
    X_all=X_all,
    alphas=alphas,
    labels=[r'$C_d$', r'$C_l$'],
    cmap_names=('turbo', 'turbo'),
    save_dir="nlr/plots",
    show_colorbar=True,
    dashed_indices=[4, 8]  # <--- dashed trajectories
)


save = False
if save:
    np.save('nlr/data/nlrX.npy', X_all)
    np.save('nlr/data/nlrAlphas.npy', alphas)
    np.save('nlr/data/nlrTime.npy', thistime)