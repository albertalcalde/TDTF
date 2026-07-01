from src.plotting import save_predictions_stack_legend


# save_predictions_stack_legend(
#     save_path=f"lorenz63/plots/pred_legend.png"
# )

save_predictions_stack_legend(
    save_path=f"sinus/plots/sinus_legend.pdf",
    baseline_models=[
        {"label": "TD-TF", "color": "tab:blue", "linestyle": "--"},
        {"label": "TD-DMD", "color": "tab:orange", "linestyle": "--"},
        {"label": "TD-MLP", "color": "tab:red", "linestyle": "--"}
        ],
    ncol=6,
    figsize=(8.4, 0.55),
)

save_predictions_stack_legend(
    "reacDiff/plots/reacDiff_legend.pdf",
    baseline_models=[
        {
            "label": "TD-TF",
            "color": "tab:blue",
            "linestyle": "--",
            "alpha": 1,
        },
        {
            "label": "TD-DMD",
            "color": "tab:orange",
            "linestyle": "--",
            "alpha": 1,
        },
        {
            "label": "TD-MLP",
            "color": "tab:red",
            "linestyle": "--",
            "alpha": 0.2,
        },
    ],
)