import torch

from rdkit.Chem import Mol as RDMol
from torch import Tensor

from gflownet.base import SynthesisTrainer, SynthesisGFNSampler, BaseTask
from gflownet.trainer import FlatRewards
from gflownet.misc.chem_metrics import mol2qed


class QEDTask(BaseTask):
    def compute_flat_rewards(self, mols: list[RDMol], batch_idx: list[int]) -> tuple[FlatRewards, Tensor]:
        fr = mol2qed(mols).reshape(-1, 1)
        is_valid_t = torch.ones((len(mols),), dtype=torch.bool)
        return FlatRewards(fr), is_valid_t


class QEDSynthesisTrainer(SynthesisTrainer):
    def setup_task(self):
        self.task: QEDTask = QEDTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)


class QEDSynthesisSampler(SynthesisGFNSampler):
    def setup_task(self):
        self.task: QEDTask = QEDTask(cfg=self.cfg, rng=self.rng, wrap_model=self._wrap_for_mp)
