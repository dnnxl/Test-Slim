import numpy as np
import torch
import torchvision

from PIL import Image
from segment_anything_2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything_2.build_sam import build_sam2, build_sam2_video_predictor
from segment_anything_2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2:

    def __init__(self, args) -> None:
        if args.sam_model == 'vit_t':
            self.checkpoint = "weights/sam2_hiera_tiny.pt"
            self.config = "sam2_hiera_t.yaml"
        elif args.sam_model == 'vit_s':
            self.checkpoint = "weights/sam2_hiera_small.pt"
            self.config = "sam2_hiera_s.yaml"
        elif args.sam_model == 'vit_b':
            self.checkpoint = "weights/sam2_hiera_base_plus.pt"
            self.config = "sam2_hiera_b+.yaml"
        elif args.sam_model == 'vit_l':
            self.checkpoint = "weights/sam2_hiera_large.pt"
            self.config = "sam2_hiera_l.yaml"
        else:
            RuntimeError("No sam config found")
        
        self.model = build_sam2(self.config, self.checkpoint, device=args.device)
        predictor = SAM2ImagePredictor(self.model)

    def load_simple_mask(self):
        #There are several tunable parameters in automatic mask generation that control 
        # how densely points are sampled and what the thresholds are for removing low 
        # quality or duplicate masks. Additionally, generation can be automatically 
        # run on crops of the image to get improved performance on smaller objects, 
        # and post-processing can remove stray pixels and holes. 
        # Here is an example configuration that samples more masks:
        #https://github.com/facebookresearch/segment-anything/blob/9e1eb9fdbc4bca4cd0d948b8ae7fe505d9f4ebc7/segment_anything/automatic_mask_generator.py#L35    

        #Rerun the following with a few settings, ex. 0.86 & 0.9 for iou_thresh
        # and 0.92 and 0.96 for score_thresh

        mask_generator_ = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            # pred_iou_thresh=0.9,
            # stability_score_thresh=0.96,
            # crop_n_layers=1, default:0
            # crop_n_points_downscale_factor=1,default:1
            min_mask_region_area=100,  # Requires open-cv to run post-processing
            output_mode="coco_rle",
        )
        self.mask_generator = mask_generator_

    def get_unlabeled_samples(self, 
            batch, idx, transform, use_sam_embeddings
        ):
        """ From a batch and its index get samples 
        Params
        :batch (<tensor, >)
        """
        imgs = []
        box_coords = []
        scores = []

        # batch[0] has the images    
        img = batch[0][idx].cpu().numpy().transpose(1,2,0)
        img_pil = Image.fromarray(img)

        # run sam to create proposals
        masks = self.mask_generator.generate(img)

        for ann in masks:
            xywh = ann['bbox']
            xyxy = torchvision.ops.box_convert(
                torch.tensor(xywh), in_fmt='xywh', out_fmt='xyxy'
            )
            # get img
            crop = img_pil.crop(np.array(xyxy))  
            if use_sam_embeddings:
                sample = transform.preprocess_sam_embed(crop)
            else:
                sample = transform.preprocess_timm_embed(crop)

            # accumulate
            imgs.append(sample)
            box_coords.append(xywh)
            scores.append(float(ann['predicted_iou']))
        return imgs, box_coords, scores

    def get_embeddings(self, img):
        """
        Receive an image and return the feature embeddings.

        Params
        :img (numpy.array) -> image.
        Return
        :torch of the embeddings from SAM.
        """
        self.mask_generator.predictor.set_image(img)
        embeddings = self.mask_generator.predictor.features

        with torch.no_grad():
            _pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            avg_pooled = _pool(embeddings).view(embeddings.size(0), -1)
        self.mask_generator.predictor.reset_image()
        return avg_pooled

    def get_features(self, img):
        """
        Receive an image and return the feature maps.

        Params
        :img (numpy.array) -> image.
        Return
        :torch of the embeddings from SAM.
        """
        self.mask_generator.predictor.set_image(img)
        embeddings = self.mask_generator.predictor.features
        return embeddings