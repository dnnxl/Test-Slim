from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image

import segment_anything as sam
import numpy as np
import torch
import torchvision

class MobileSAM:

    def __init__(self, args) -> None:
        # Configuration of MobileSAM for object proposals masks
        self.checkpoint = "weights/mobile_sam.pt"
        self.model_type = "vit_t"
        self.use_sam_embeddings = False
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint).to(args.device)
        self.mask_generator = None
        self.model.eval()

        # get the size of the embeddings
        new_pil = Image.new(mode="RGB", size=(200,200))
        predictor = SamPredictor(self.model) 
        predictor.set_image(np.array(new_pil))

        # features shape: ([1, 256, 64, 64]), we keep 256
        self.features_size = predictor.features.shape[1]
        predictor.reset_image()

        # ------------------------------------------------
        # Config for SAM model for the embeddings
        if args.use_sam_embeddings == 1:
            self.use_sam_embeddings = True
            if args.sam_model == 'b':
                self.checkpoint_embeddings = "weights/sam_vit_b_01ec64.pth"
                self.model_type_embeddings = "vit_b"
            elif args.sam_model == 'h':
                self.checkpoint_embeddings = "weights/sam_vit_h_4b8939.pth"
                self.model_type_embeddings = "vit_h"
            else:
                RuntimeError("No sam config found")
            self.model_embeddings = sam.sam_model_registry[self.model_type_embeddings](checkpoint=self.checkpoint_embeddings).to(args.device)

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

        mask_generator_ = SamAutomaticMaskGenerator(
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

        if self.use_sam_embeddings:
            # SAM embeddings
            mask_generator_embeddings = sam.SamAutomaticMaskGenerator(
                model=self.model,
                points_per_side=32,
                # pred_iou_thresh=0.9,
                # stability_score_thresh=0.96,
                # crop_n_layers=1, default:0
                # crop_n_points_downscale_factor=1,default:1
                min_mask_region_area=100,  # Requires open-cv to run post-processing
                output_mode="coco_rle",
            )
            self.mask_generator_embeddings = mask_generator_embeddings

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
        self.mask_generator_embeddings.predictor.set_image(img)
        embeddings = self.mask_generator_embeddings.predictor.features

        with torch.no_grad():
            _pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            avg_pooled = _pool(embeddings).view(embeddings.size(0), -1)
        self.mask_generator_embeddings.predictor.reset_image()
        return avg_pooled

    def get_features(self, img):
        """
        Receive an image and return the feature maps.

        Params
        :img (numpy.array) -> image.
        Return
        :torch of the embeddings from SAM.
        """
        if self.use_sam_embeddings:
            self.mask_generator_embeddings.predictor.set_image(img)
            embeddings = self.mask_generator_embeddings.predictor.features
            return embeddings