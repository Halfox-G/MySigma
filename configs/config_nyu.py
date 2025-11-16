import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict

C = edict()
config = C
cfg = C

C.seed = 3407

remoteip = os.popen('pwd').read()
C.root_dir = os.path.abspath(os.path.join(os.getcwd(), './'))
C.abs_dir = osp.realpath(".")

# Dataset config
"""Dataset Path"""
C.dataset_name = 'NYUDepthv2'
C.dataset_path = osp.join(C.root_dir, 'datasets', 'NYUDepthv2')
C.rgb_root_folder = osp.join(C.dataset_path, 'RGB')
C.rgb_format = '.jpg'
C.gt_root_folder = osp.join(C.dataset_path, 'Label')
C.gt_format = '.png'
C.gt_transform = True
# True when label 0 is invalid, you can also modify the function _transform_gt in dataloader.RGBXDataset
# True for most dataset valid, Faslse for MFNet(?)
C.x_root_folder = osp.join(C.dataset_path, 'Depth')
C.x_format = '.png'
C.x_is_single_channel = True # True for raw depth, thermal and aolp/dolp(not aolp/dolp tri) input
# training sources removed; keep eval source for inference/evaluation lists
C.eval_source = osp.join(C.dataset_path, "test2.txt")
# Use this config primarily for inference. Set to True to indicate test/inference mode.
C.is_test = True
C.num_eval_imgs = 654
C.num_classes = 40
C.class_names = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floor mat",
    "clothes",
    "ceiling",
    "books",
    "refridgerator",
    "television",
    "paper",
    "towel",
    "shower curtain",
    "box",
    "whiteboard",
    "person",
    "night stand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
    "otherstructure",
    "otherfurniture",
    "otherprop",
]

"""Image Config"""
C.background = 255
C.image_height = 480
C.image_width = 640
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = 'sigma_small' # sigma_tiny / sigma_small / sigma_base
C.pretrained_model = None # do not need to change
C.decoder = 'MambaDecoder' # 'MLPDecoder'
C.decoder_embed_dim = 512

"""Inference / General Config"""
# Turn on test/inference mode and set light-weight dataloader workers for inference.
C.is_test = True
C.inference_batch_size = 1
C.device = 'cuda'  # or 'cpu'
C.model_checkpoint = None  # Path to trained model; set this before running inference
# Keep a reasonable number of workers for data loading during inference
C.num_workers = 4

# Batch-norm and related flags are kept for compatibility with model loading but are
# not used for training here.
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
# C.eval_iter = 1
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [0.75, 1, 1.25] 
C.eval_flip = True
C.eval_crop_size = [480, 640] # [height weight]

"""Store Config"""
# Checkpoint directory is still useful for loading saved models at inference time.
# Training-specific checkpoint scheduling fields removed.

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(osp.join(C.root_dir))

C.log_dir = osp.join(C.root_dir, 'log_final', 'log_nyudepth', 
                     f'log_{C.dataset_name}_{C.backbone}_cromb_conmb_cvssdecoder')
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

if __name__ == '__main__':
    # Simple summary for inference usage
    print("Inference config summary:")
    print(f"  dataset: {C.dataset_name}")
    print(f"  is_test: {C.is_test}")
    print(f"  eval list: {C.eval_source}")
    print(f"  checkpoint_dir: {C.checkpoint_dir}")
    print(f"  model_checkpoint: {C.model_checkpoint}")