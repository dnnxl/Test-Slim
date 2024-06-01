# Run commands:
- python>=3.8
- pip3 install torch torchvision torchaudio 
- pip install timm
- pip install PyYAML
- pip install pycocotools
- pip install omegaconf
- pip install psutil
- pip install scikit-learn

## For Fast SAM 
- pip install git+https://github.com/openai/CLIP.git
- pip install ultralytics == 8.0.120

## For Semantic SAM 
### Dependencies
- pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
#### Detectron
Possible solution for No module named 'MultiScaleDeformableAttention'
https://github.com/fundamentalvision/Deformable-DETR/issues/74.

- git clone https://github.com/facebookresearch/Mask2Former.git
- cd ../Mask2Former/mask2former/modeling/pixel_decoder/ops
- sh make.sh

# Uninstall previous version of Open CV and reinstall the proper one
- pip uninstall opencv-contrib-python opencv-python
- pip install opencv-contrib-python

# To install SAM
- pip install git+https://github.com/facebookresearch/segment-anything.git

# To install Fast SAM
- pip install git+https://github.com/CASIA-IVA-Lab/FastSAM.git

# To install Mobile SAM
- pip install git+https://github.com/ChaoningZhang/MobileSAM.git

# To install Semantic SAM
- pip install git+https://github.com/UX-Decoder/Semantic-SAM
- pip install kornia

# Download Model Weights

## Fast SAM
- https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing

## Mobile SAM
- https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt

## Semantic SAM 
- Swin T: https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swint_only_sam_many2many.pth
- Swin L: https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth

# Other resources
- official Github repo: https://github.com/facebookresearch/segment-anything
- An example of SAM: https://blog.roboflow.com/how-to-use-segment-anything-model-sam/
- An example of how to fine-tune SAM: https://blog.roboflow.com/how-to-use-segment-anything-model-sam/
- An example of FastSAM: https://blog.roboflow.com/what-is-fastsam/
- Paper Semantic SAM:  
- Paper Mobile SAM: