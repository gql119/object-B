"""Shared actual trainer construction for S5 smoke and formal generation."""
def create_trainer(cfg,role,calibration_path,cache_directory,cache_identity):
    import random
    import numpy as np
    import torch
    from ultralytics import YOLO
    from .s4_tfss import S4Trainer,CleanMapCache
    from .s5_cup_history import load_cup_config
    from .s5_cup_trainer import S5CUPTrainer
    if role not in ('S4_reference','P3_off','P3_CUP'):raise ValueError('Unknown S5 role')
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    frozen=YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0')
    adaptive=YOLO(cfg['surrogate']['ckpt']).model.to('cuda:0')
    args=(cfg,cfg['methods']['legacy_kproto_ret'],torch.device('cuda:0'),frozen,adaptive)
    t=S4Trainer(*args) if role=='S4_reference' else S5CUPTrainer(*args,
        cup_config=load_cup_config(calibration_path),cup_enabled=role=='P3_CUP')
    t.tfss_mode='TFSS-U'
    t.s4_cache=CleanMapCache(cache_directory,cache_identity)
    return t
