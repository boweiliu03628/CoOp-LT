from yacs.config import CfgNode as CN

_C = CN()
_C.dataset=None
_C.backbone=None
_C.output_dir = None  # Directory to save the output files (like log.txt and model weights)
_C.seed = 1  # use manual seed
_C.num_workers = 16
_C.lr = 0.002
_C.weight_decay = 5e-4
_C.momentum = 0.9
_C.test_only=True


_C.train_img_path=''
_C.test_img_path=''
_C.num_epochs =None
_C.train_batch_size=None
_C.test_batch_size=None


_C.n_ctx =None
_C.ctp=None
_C.ctx_init=None




