#!/usr/bin/env python
import warnings
import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from utils import Constants_AugMethod, add_bool_arg
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False
has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass
from data import create_loader, create_dataset, create_parser, \
    DetectionDatset, SkipSubset, resolve_input_config
from utils import add_bool_arg, get_parameters, seed_everything, throttle_cpu
from engine import SAM
from pycocotools.coco import COCO
from eval import *
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
#------------------------------------------------------------------------------------------------


def create_datasets_and_loaders(args):
    """ Setup datasets, transforms, loaders, evaluator.
    Params
    :args: Model specific configuration dict / struct

    Returns
    :Train loader, validation loader
    """
    # get datasets: labeled, unlabeled (according to a specific percentage),
    # and eval.
    datasets = create_dataset(
        args.dataset, args.root, use_semi_split=args.use_semi_split,
        semi_percentage=args.semi_percentage
    )
    dataset_train_labeled = datasets[0] # labeled
    dataset_train_unlabeled = datasets[1] # unlabeled is not used here
    dataset_eval = datasets[2] # eval

    # create pytorch loaders
    loader_labeled = create_loader(
        dataset_train_labeled,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size,
        is_training=True,
        re_prob=args.reprob,
    )
    loader_eval = create_loader(
        dataset_eval,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size_val,
        is_training=False,
    )
    return loader_labeled, loader_eval

def run_finetuning(args):
    """ Run simple experiment """
    path_output = f"./output/{args.output_folder}/{args.semi_percentage}/{args.run_name}"

    # STEP 1: create data loaders
    loader_labeled, loader_eval = create_datasets_and_loaders(args)

    # STEP 2: create a SAM model
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # STEP 3: fine-tune SAM
    # Resources: 
    # official Github repo: https://github.com/facebookresearch/segment-anything
    # An example of SAM: https://blog.roboflow.com/how-to-use-segment-anything-model-sam/
    # An example of how to fine-tune SAM: https://blog.roboflow.com/how-to-use-segment-anything-model-sam/

    

if __name__ == '__main__':
    args = get_parameters()
    # throttle_cpu(args.numa)
    if not args.seed == None:
        seed_everything(args.seed)
    run_finetuning(args)