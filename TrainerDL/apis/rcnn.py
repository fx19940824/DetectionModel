import os
import re
import torch

from TrainerDL.dataprocess.labelme2coco import generate_dataset
from tools.train_net import train
from tools.train_net import run_test
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.comm import get_rank
from maskrcnn_benchmark.utils.collect_env import collect_env_info


def train_rcnn(cfgs):
    cfg_update_and_freeze(cfgs, True)
    gpus = cfgs["gpus"]
    config_file = cfgs["cfgpath"]

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(gpus)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    logger = setup_logger("maskrcnn_benchmark", cfg.OUTPUT_DIR, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(config_file))

    with open(config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    train(cfg, gpus, distributed)


def test_rcnn(cfgs):
    cfg_update_and_freeze(cfgs, True)
    gpus = cfgs["gpus"]

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(gpus)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    run_test(cfg, distributed)



def auto_gen_datacfg(tcfg, cfgs):
    train_img_dir = cfgs["train_img_dir"]
    train_ann_dir = cfgs.get("train_ann_dir", 0)
    val_img_dir = cfgs.get("val_img_dir", 0)
    val_ann_dir = cfgs.get("val_ann_dir", 0)

    isKeypoints = cfgs["isKeypoints"]
    opts = cfgs.get("opts", [])
    tcfg.merge_from_list(opts)
    if not (train_ann_dir and val_ann_dir):
        ratio = cfgs.get("ratio", 0.05)
        train_img_dir, val_img_dir, train_ann_dir, val_ann_dir = \
            generate_dataset(train_img_dir, '/'.join(train_img_dir.split('/')[:-1]), suffix='Data', ratio=ratio)

    train_dataset_name, test_dataset_name = train_img_dir.replace('/', '') + '_cocostyle', val_img_dir.replace('/','') +'_cocostyle'

    string = """
    #   data
    DATASETS = {
        "%s": {
            "img_dir": "%s",
            "ann_file": "%s"
        },
        "%s": {
            "img_dir": "%s",
            "ann_file": "%s"
        }

    }
    #   data""" % (train_dataset_name, train_img_dir, train_ann_dir, test_dataset_name, val_img_dir, val_ann_dir)

    text = open(tcfg.PATHS_CATALOG, 'r').read()
    text = re.sub(re.compile(r'\s*#\s*data[\s\S]*#\s*data'), string, text)
    open(tcfg.PATHS_CATALOG, 'w').write(text)

    opts = opts + ["DATASETS.TRAIN", (train_dataset_name,), "DATASETS.TEST", (test_dataset_name,)]

    tcfg.merge_from_list(opts)
    return tcfg


def cfg_update_and_freeze(cfgs, load_dataset=False):
    config_file = cfgs["cfgpath"]
    cur_cfg = cfg.clone()
    cur_cfg.merge_from_file(config_file)
    if load_dataset:
        cur_cfg.PATHS_CATALOG = os.path.abspath("./cfgs/paths_catalog.py")
        cur_cfg = auto_gen_datacfg(cur_cfg, cfgs)

    if cfg.is_frozen():
        cfg.defrost()
    cfg.merge_from_other_cfg(cur_cfg)
    cfg.freeze()

