import time

from opt_qed import QEDTask

from rxnflow.base import RxnFlowSampler
from rxnflow.config import Config, init_empty

# NOTE: example setting
NUM_SAMPLES = 200
DEVICE = "cpu"  # or 'cuda'


class QEDSampler(RxnFlowSampler):
    def setup_task(self):
        self.task = QEDTask(cfg=self.cfg)


if __name__ == "__main__":
    # change config from training
    config = init_empty(Config())
    config.algo.num_from_policy = 100  # 64 -> 100
    config.env_dir = "./data/envs/catalog"  # using catalog environment

    ckpt_path = "./weights/qed-unif-0-64_20250512.pt"

    # construct sampler
    sampler = QEDSampler(config, ckpt_path, DEVICE)

    # type1: generate molecules only
    tick_st = time.time()
    res = sampler.sample(NUM_SAMPLES, calc_reward=False)
    tick_end = time.time()
    print(f"Generated Molecules: {len(res)}")
    print(f"Sampling: {tick_end - tick_st:.3f} sec")

    # save molecules
    with open("./example-qed.smi", "w") as w:
        for idx, sample in enumerate(res):
            w.write(f"{sample['smiles']}\tsample{idx}\n")

    # type2: generate molecules with their rewards
    tick_st = time.time()
    res = sampler.sample(NUM_SAMPLES, calc_reward=True)
    tick_end = time.time()
    print(f"Generated Molecules: {len(res)}")
    print(f"Sampling: {tick_end - tick_st:.3f} sec")

    # save molecules
    with open("./example-qed.csv", "w") as w:
        w.write(",SMILES,QED\n")
        for idx, sample in enumerate(res):
            smiles = sample["smiles"]
            qed = sample["info"]["reward"][0]
            w.write(f"{idx},{smiles},{qed:.3f}\n")
