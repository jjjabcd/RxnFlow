import torch
from torch import Tensor
import torch_geometric.data as gd

from gflownet.utils.misc import get_worker_device
from rxnflow.algo.trajectory_balance import SynthesisTB
from rxnflow.algo.synthetic_path_sampling import SyntheticPathSampler
from rxnflow.policy.action_categorical import RxnActionCategorical

from .model import RxnFlow_MP


class SynthesisTB_MP(SynthesisTB):
    def setup_graph_sampler(self):
        self.graph_sampler = SyntheticPathSampler_MP(
            self.ctx,
            self.env,
            self.global_cfg.algo.min_len,
            self.global_cfg.algo.max_len,
            self.action_subsampler,
            self.global_cfg.algo.action_subsampling.onpolicy_temp,
            self.sample_temp,
            correct_idempotent=self.cfg.do_correct_idempotent,
            pad_with_terminal_state=self.cfg.do_parameterize_p_b,
            num_workers=self.global_cfg.num_workers_retrosynthesis,
        )

    def estimate_policy(
        self,
        model: RxnFlow_MP,
        batch: gd.Batch,
        cond_info: torch.Tensor | None,
        batch_idx: torch.Tensor,
    ) -> tuple[RxnActionCategorical, RxnActionCategorical | None, Tensor]:
        if self.cfg.do_parameterize_p_b:
            raise NotImplementedError
        else:
            if self.model_is_autoregressive:
                raise NotImplementedError
                fwd_cat, per_graph_out = model(batch, cond_info, batch_idx, batched=True)
            else:
                batched_cond_info = cond_info[batch_idx] if cond_info is not None else None
                fwd_cat, per_graph_out = model(batch, batched_cond_info, batch_idx)
            bck_cat = None
        return fwd_cat, bck_cat, per_graph_out


class SyntheticPathSampler_MP(SyntheticPathSampler):
    def estimate_policy(
        self,
        model: RxnFlow_MP,
        torch_graphs: list[gd.Data],
        cond_info: torch.Tensor,
        not_done_mask: list[bool],
    ) -> RxnActionCategorical:
        dev = get_worker_device()
        batch_idcs = [i for i, v in enumerate(not_done_mask) if v]
        fwd_cat, *_ = model(self.ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask], batch_idcs)
        return fwd_cat
