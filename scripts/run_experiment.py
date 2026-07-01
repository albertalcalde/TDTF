# scripts/run_experiment.py

import argparse
from email import parser

from src.experiments.engine import ExperimentConfig, run_experiment
from src.experiments.cases.sinus import load_case as load_sinus
from src.experiments.cases.lorenz63 import load_case as load_lorenz63
from src.experiments.cases.reacdiff import load_case as load_reacdiff
from src.experiments.cases.nlr import load_case as load_nlr, DEFAULT_TEST_TRAJ_INDICES

def parse_ints(values):
    return [int(v) for v in values]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--case", required=True, choices=["sinus", "lorenz63", "reacdiff", "nlr"])
    parser.add_argument("--qoi", choices=["lift", "drag"], default=None)
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["tdtf", "delay_mlp"],
        default=["tdtf"],
    )    
    parser.add_argument("--tau", type=int, default=30)
    parser.add_argument("--test-traj-ind", nargs="+", type=int, default=None)
    parser.add_argument("--nlr-variant", choices=["interpolation", "extrapolation"], default="interpolation")

    parser.add_argument("--n", nargs="+", type=int, required=True)
    parser.add_argument("--h", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])

    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--J", type=int, default=5000)
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--replace", action="store_true")

    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=500)

    parser.add_argument("--output-root", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    if args.case == "sinus":
        case_data = load_sinus()
    elif args.case == "lorenz63":
        case_data = load_lorenz63()
    elif args.case == "reacdiff":
        case_data = load_reacdiff()
    elif args.case == "nlr":
        if args.qoi is None:
            raise ValueError("--qoi is required for --case nlr")

        test_traj_ind = args.test_traj_ind
        if args.nlr_variant == "extrapolation" and test_traj_ind is None:
            test_traj_ind = [-2, -1]

        case_data = load_nlr(
            qoi=args.qoi,
            tau=args.tau,
            test_traj_ind=test_traj_ind if test_traj_ind is not None else DEFAULT_TEST_TRAJ_INDICES,
            variant=args.nlr_variant,
        )
    else:
        raise ValueError(f"Unknown case: {args.case}")

    cfg = ExperimentConfig(
        case=args.case,
        models=args.models,
        n_values=args.n,
        h_values=args.h,
        seeds=args.seeds,
        m=args.m,
        J=args.J,
        dataset_seed=args.dataset_seed,
        replace=args.replace,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        output_root=args.output_root,
        save_predictions=args.save_predictions,
        save_checkpoints=args.save_checkpoints,
        make_plots=args.plot,
    )

    run_experiment(case_data, cfg)


if __name__ == "__main__":
    main()
