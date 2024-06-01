#!/usr/bin/env python
import torch
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
from data import create_loader, create_dataset, create_dataset_ood
from utils import get_parameters, seed_everything, throttle_cpu, save_gt
from engine import SAM
from pycocotools.coco import COCO
from eval import *
from ood_filter import OOD_filter_neg_likelihood
from data import transforms_toNumpy
#------------------------------------------------------------------------------------------------


def create_datasets_and_loaders(args):
    """ Setup datasets, transforms, loaders, evaluator.
    Params
    :args -> Model specific configuration dict / struct

    Returns
    :loaders
    """
    datasets = create_dataset_ood(
        args.dataset, args.root, 
        labeled_samples=args.ood_labeled_samples,
        unlabeled_samples=args.ood_unlabeled_samples
    )
    dataset_train_labeled = datasets[0]
    dataset_train_unlabeled = datasets[1]

    # create data loaders
    trans_f = transforms_toNumpy()
    normalize_imgs = False
    loader_labeled = create_loader(
        dataset_train_labeled,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size_labeled,
        is_training=False,
        transform_fn = trans_f,
        normalize_img=normalize_imgs
    )
    loader_unlabeled = create_loader(
        dataset_train_unlabeled,
        img_resolution=args.img_resolution,
        batch_size=args.batch_size_unlabeled,
        is_training=False,
        transform_fn = trans_f,
        normalize_img=normalize_imgs
    )
    return loader_labeled, loader_unlabeled

def run_experiment(args):
    """ Run simple experiment """

    # STEP 1: create data loaders
    labeled_loader, unlabeled_loader = create_datasets_and_loaders(args)
    # save new gt into a separate json file
    output_root = f"./output/{args.output_folder}/{args.ood_labeled_samples}_{args.ood_unlabeled_samples}_seed{args.seed}"
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    # save gt
    save_gt(unlabeled_loader, output_root, args.method)

    # STEP 2: run the ood filter using inferences from SAM
    # sam instance - default values of the model
    sam = SAM(args)
    mask_gen = sam.get_simple_mask()

    # instance the main class and instance the timm model
    ood_filter_neg_likelihood = OOD_filter_neg_likelihood(
        timm_model=args.timm_model, 
        timm_pretrained=args.load_pretrained,
        sam_model=mask_gen
    )

    # run filter using the backbone, sam, and ood
    ood_filter_neg_likelihood.run_filter(
        labeled_loader, unlabeled_loader, 
        dir_filtered_root=output_root,
        ood_thresh=args.ood_thresh,
        ood_hist_bins=args.ood_histogram_bins
    )

    # STEP 3: evaluate results
    MAX_IMAGES = 100000
    gt_eval_path = f"{output_root}/gt_{args.method}.json"
    coco_gt = COCO(gt_eval_path)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    res_data = f"{output_root}/bbox_results_{args.method}.json"
    eval_sam(coco_gt, image_ids, res_data)
    
if __name__ == '__main__':
    args = get_parameters()
    throttle_cpu(args.numa)
    if not args.seed == None:
        seed_everything(args.seed)
    run_experiment(args)