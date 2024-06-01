from semantic_sam import build_semantic_sam, SemanticSamAutomaticMaskGenerator
import numpy as np
import torch
import torchvision
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

class SemanticSAM:

    def __init__(self, args) -> None:
        #if args.sam_model == 'T':
        self.checkpoint = "weights/swint_only_sam_many2many.pth"
        self.model_type = "T"
        #elif args.sam_model == 'L':
        #    self.checkpoint = "weights/swinl_only_sam_many2many.pth"
        #    self.model_type = "L"
        #else:
        #    RuntimeError("No Semantic SAM weights found")
        self.model = build_semantic_sam[self.model_type](checkpoint=self.checkpoint)#.to('cuda')
        self.mask_generator = None

        # ------------------------------------------------
        # Config for SAM model for the embeddings
        if args.sam_model == 'b':
            self.checkpoint_embeddings = "weights/sam_vit_b_01ec64.pth"
            self.model_type_embeddings = "vit_b"
        elif args.sam_model == 'h':
            self.checkpoint_embeddings = "weights/sam_vit_h_4b8939.pth"
            self.model_type_embeddings = "vit_h"
        else:
            RuntimeError("No sam config found")
        self.model_embeddings = sam_model_registry[self.model_type_embeddings](checkpoint=self.checkpoint_embeddings).to(args.device)


    def load_simple_mask(self):
        #There are several tunable parameters in automatic mask generation that control 
        # how densely points are sampled and what the thresholds are for removing low 
        # quality or duplicate masks. Additionally, generation can be automatically 
        # run on crops of the image to get improved performance on smaller objects, 
        # and post-processing can remove stray pixels and holes. 
        # Here is an example configuration that samples more masks:
        #https://github.com/UX-Decoder/Semantic-SAM/blob/main/tasks/automatic_mask_generator.py

        #Rerun the following with a few settings, ex. 0.86 & 0.9 for iou_thresh
        # and 0.92 and 0.96 for score_thresh
        # SAM embeddings
        mask_generator_embeddings = SamAutomaticMaskGenerator(
            model=self.model_embeddings,
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

        # The model receive the tensor image
        image_ori = np.asarray(img_pil)
        image_torch = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).to(self.device)
        img = image_torch.to(self.device)

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
        self.mask_generator_embeddings.predictor.set_image(img)
        embeddings = self.mask_generator_embeddings.predictor.features
        return embeddings