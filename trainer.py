import os
import argparse

from config import get_cfg
from pytorch_lightning import Trainer

from sagan import SAGAN

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="config/default_config.yaml")
args = parser.parse_args()

cfg = get_cfg()
cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()
# default_setup(cfg, args)

model = SAGAN()
trainer = Trainer(gpus=cfg.SOLVER.GPUS, num_nodes=cfg.SOLVER.NUM_NODES)
trainer.fit(model)