import os
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser("RxnFlow", description="QED-UniDock Optimization with RxnFlow")
    opt_cfg = parser.add_argument_group("Protein Config")
    opt_cfg.add_argument("-p", "--protein", type=str, required=True, help="Protein PDB Path")
    opt_cfg.add_argument("-c", "--center", nargs="+", type=float, help="Pocket Center (--center X Y Z)")
    opt_cfg.add_argument("-l", "--ref_ligand", type=str, help="Reference Ligand Path (required if center is missing)")

    run_cfg = parser.add_argument_group("Operation Config")
    run_cfg.add_argument("-o", "--out_dir", type=str, required=True, help="Output directory")
    run_cfg.add_argument(
        "-n",
        "--num_oracles",
        type=int,
        default=1000,
        help="Number of Oracles (64 molecules per oracle; default: 1000)",
    )
    run_cfg.add_argument("--env_dir", type=str, default="./data/envs/enamine_all", help="Environment Directory Path")
    run_cfg.add_argument(
        "--subsampling_ratio",
        type=float,
        default=0.01,
        help="Action Subsampling Ratio. Memory-variance trade-off (Smaller ratio increase variance; default: 0.01)",
    )
    return parser.parse_args()


def get_center(ligand_path: str) -> tuple[float, float, float]:
    from openbabel import pybel
    import numpy as np

    extension = os.path.splitext(ligand_path)[-1][1:]
    pbmol: pybel.Molecule = next(pybel.readfile(extension, ligand_path))
    x, y, z = np.mean([atom.coords for atom in pbmol.atoms], axis=0).tolist()
    return round(x, 3), round(y, 3), round(z, 3)


def run(args):
    from gflownet.tasks.unidock_moo_synthesis import UniDockMOOSynthesisTrainer
    from gflownet.config import Config, init_empty

    config = init_empty(Config())
    config.desc = "Vina-QED optimization with UniDock"
    config.env_dir = args.env_dir
    config.task.docking.protein_path = args.protein
    config.task.docking.center = tuple(args.center)
    config.num_training_steps = args.num_oracles
    config.algo.action_sampling.sampling_ratio_reactbi = args.subsampling_ratio
    config.cond.temperature.dist_params = [16, 64]  # Different to Paper!
    config.log_dir = args.out_dir
    config.print_every = 1
    trainer = UniDockMOOSynthesisTrainer(config)
    trainer.run()


if __name__ == "__main__":
    args = parse_args()
    assert (args.center is not None) or (args.ref_ligand is not None), "--center or --ref_ligand is required"
    if args.center is None:
        args.center = get_center(args.ref_ligand)
    else:
        assert len(args.center) == 3, "--center need three values: X Y Z"
    run(args)
