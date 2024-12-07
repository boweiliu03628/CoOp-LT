import argparse
import torch
from trainer import Trainer
import os
from utils.config import _C as cfg
from utils.logger import setup_logger
import random
import numpy as np


def main(args):

    cfg.dataset = args.dataset
    cfg.backbone = args.backbone
    cfg_dataset_file = os.path.join("./configs/dataset", cfg.dataset + ".yaml")
    cfg_backbone_file = os.path.join("./configs/model", cfg.backbone + ".yaml")
    cfg.merge_from_file(cfg_dataset_file)
    cfg.merge_from_file(cfg_backbone_file)
    cfg.merge_from_list(args.opts)

    if cfg.output_dir is None:
        cfg_name = "_".join([cfg.dataset, cfg.backbone])
        opts_name = "".join(["_" + item for item in args.opts])
        other_name="_".join(["nctx",str(cfg.n_ctx),cfg.ctp,"epoch",str(cfg.num_epochs)])
        cfg.output_dir = os.path.join("./output", cfg_name,other_name+opts_name)
    else:
        cfg.output_dir = os.path.join("./output", cfg.output_dir)
    print("Output directory: {}".format(cfg.output_dir))
    setup_logger(cfg.output_dir)

    print("** Config **")
    print(cfg)
    print("*************")

    if cfg.seed is not None:
        seed = cfg.seed
        print("Setting fixed seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    trainer = Trainer(cfg)

    if cfg.test_only == True:
        trainer.test_only()
        return

    trainer.train_and_evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="", help="data config file")
    parser.add_argument("--backbone", "-b", type=str, default="", help="model config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")
    args = parser.parse_args()
    main(args)
