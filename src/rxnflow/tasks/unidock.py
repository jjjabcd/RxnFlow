from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import QED

from collections.abc import Callable
from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet import ObjectProperties

from rxnflow.config import Config, init_empty
from rxnflow.base import BaseTask, RxnFlowTrainer, mogfn_trainer
from rxnflow.utils.chem_metrics import mol2qed, mol2vina

aux_tasks = {"qed": mol2qed}


class UniDockTask(BaseTask):
    """Sets up a task where the reward is computed using a UniDock - TacoGFN Task."""

    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        self.protein_path: str = cfg.task.docking.protein_path
        self.center: tuple[float, float, float] = cfg.task.docking.center
        self.best_molecules: list[tuple[float, str]] = []
        self.save_dir: Path = Path(cfg.log_dir) / "unidock"
        self.oracle_idx = 0
        self.search_mode: str = "balance"

    def compute_obj_properties(self, objs: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        is_valid_t = torch.ones((len(objs),), dtype=torch.bool)
        docking_scores = self.run_docking(objs)
        self.update_best_molecules(objs, docking_scores.tolist())
        fr = docking_scores.neg().reshape(-1, 1)
        return ObjectProperties(fr), is_valid_t

    def update_best_molecules(self, mols: list[RDMol], scores: list[float]):
        best_smi = set(smi for score, smi in self.best_molecules)
        score_smiles = [
            (score, Chem.MolToSmiles(mol)) for score, mol in zip(scores, mols, strict=True) if self.constraint(mol)
        ]
        score_smiles = [(score, smi) for score, smi in score_smiles if smi not in best_smi]
        self.best_molecules = sorted(self.best_molecules + score_smiles, reverse=False)[:100]

    def run_docking(self, mols: list[RDMol]) -> Tensor:
        out_dir = self.save_dir / f"oracle{self.oracle_idx}"
        docking_score = mol2vina(mols, self.protein_path, self.center, self.search_mode, out_dir)
        self.oracle_idx += 1
        return docking_score

    def constraint(self, mol: RDMol) -> bool:
        return True


class UniDockMOO_PretrainTask(BaseTask):
    """Sets up a task where the reward is computed using a UniDock, QED."""

    is_moo = True

    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        assert set(self.objectives) <= {"docking", "qed"}

    def compute_obj_properties(self, objs: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        is_valid_t = torch.ones(len(objs), dtype=torch.bool)
        fr: Tensor
        fr_dict: dict[str, Tensor] = {}
        fr_sum: Tensor = torch.zeros(len(objs))
        self.avg_reward_info = []
        for obj in self.objectives:
            if obj == "docking":
                continue
            else:
                fr = aux_tasks[obj](objs)
            fr_dict[obj] = fr
            fr_sum = fr_sum + fr
            self.avg_reward_info.append((obj, fr.mean().item()))
        fr_dict["docking"] = fr_sum / (len(self.objectives) - 1)
        flat_r = [fr_dict[obj] for obj in self.objectives]
        flat_rewards = torch.stack(flat_r, dim=1)
        assert flat_rewards.shape[0] == len(objs)
        return ObjectProperties(flat_rewards), is_valid_t


class UniDockMOOTask(UniDockTask):
    """Sets up a task where the reward is computed using a UniDock, QED."""

    is_moo = True

    def __init__(self, cfg: Config, wrap_model: Callable[[nn.Module], nn.Module]):
        super().__init__(cfg, wrap_model)
        assert set(self.objectives) <= {"docking", "qed"}

    def compute_obj_properties(self, mols: list[RDMol]) -> tuple[ObjectProperties, Tensor]:
        is_valid_t = torch.ones(len(mols), dtype=torch.bool)

        fr: Tensor
        flat_r: list[Tensor] = []
        self.avg_reward_info = []
        for obj in self.objectives:
            if obj == "docking":
                docking_scores = self.run_docking(mols)
                self.update_best_molecules(mols, docking_scores.tolist())
                fr = docking_scores * -0.1
            else:
                fr = aux_tasks[obj](mols)
            flat_r.append(fr)
            self.avg_reward_info.append((obj, fr.mean().item()))
        flat_rewards = torch.stack(flat_r, dim=1)
        assert flat_rewards.shape[0] == len(mols)
        return ObjectProperties(flat_rewards), is_valid_t

    def constraint(self, mol: RDMol) -> bool:
        return QED.qed(mol) > 0.5


@mogfn_trainer
class UniDockMOOTrainer(RxnFlowTrainer):
    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.validate_every = 0
        base.task.moo.objectives = ["docking", "qed"]
        base.num_training_steps = 1000

        # NOTE: Different to paper
        base.cond.temperature.dist_params = [16, 64]  # Different to Paper!
        base.replay.use = True
        base.replay.capacity = 6_400
        base.replay.warmup = 128
        base.cond.weighted_prefs.preference_type = "dirichlet"
        base.cond.focus_region.focus_type = None
        base.algo.train_random_action_prob = 0.05

    def setup_task(self):
        self.task = UniDockMOOTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def log(self, info, index, key):
        for obj, v in self.task.avg_reward_info:
            info[f"sampled_{obj}_avg"] = v
        if len(self.task.best_molecules) > 0:
            info["top100_n"] = len(self.task.best_molecules)
            info["top100_docking"] = np.mean([score for score, _ in self.task.best_molecules])
        super().log(info, index, key)


@mogfn_trainer
class UniDockMOO_Pretrainer(RxnFlowTrainer):
    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.desc = "Vina-QED optimization with UniDock"
        base.validate_every = 0
        base.task.moo.objectives = ["docking", "qed"]
        base.cond.weighted_prefs.preference_type = "dirichlet"
        base.cond.focus_region.focus_type = None

        base.algo.sampling_tau = 0.0
        base.cond.temperature.dist_params = [16, 64]  # Different to Paper!
        base.algo.train_random_action_prob = 0.1

    def setup_task(self):
        self.task = UniDockMOO_PretrainTask(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def log(self, info, index, key):
        for obj, v in self.task.avg_reward_info:
            info[f"sampled_{obj}_avg"] = v
        super().log(info, index, key)


if __name__ == "__main__":
    """Example of how this trainer can be run"""
    config = init_empty(Config())
    config.print_every = 1
    config.num_training_steps = 100
    config.log_dir = "./logs/debug-unidock-moo-syn/"
    config.env_dir = "./data/envs/ablation/subsampled_1k/"
    config.overwrite_existing_exp = True
    config.algo.action_subsampling.sampling_ratio_reactbi = 0.1
    config.algo.action_subsampling.min_sampling_reactbi = 10

    config.task.docking.protein_path = "./data/experiments/LIT-PCBA/ADRB2.pdb"
    config.task.docking.center = (-1.96, -12.27, -48.98)

    trial = UniDockMOOTrainer(config)
    trial.run()
