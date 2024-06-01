import os
import cv2
import json
import random
import torch
import numpy as np
from engine.feature_extractor import MyFeatureExtractor
from data import get_foreground, Transform_To_Models, get_background
from tqdm import tqdm
from utils import *
from torch import nn, optim

class Autoencoder(nn.Module):
    """Makes the main denoising auto

    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(in_shape, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, enc_shape),
        )

        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, in_shape)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
def train(model, error, optimizer, n_epochs, x):
    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        output = model(x)
        loss = error(output, x)
        loss.backward()
        optimizer.step()

        if epoch % int(0.1*n_epochs) == 0:
            print(f'epoch {epoch} \t Loss: {loss.item():.4g}')

class AutoencoderFilter:
        def __init__(self,
                timm_model=None,
                timm_pretrained=True,
                num_classes=1,
                sam_model=None,
                use_sam_embeddings=False,
                is_single_class=True,
                device="cpu",
                epochs=10000):
            """
            Raises:
                ValueError: if the backbone is not a feature extractor,
                i.e. if its output for a given image is not a 1-dim tensor.
            """
            self.device = device
            self.num_classes = num_classes
            self.timm_model = timm_model
            self.sam_model = sam_model
            self.is_single_class = is_single_class
            self.epochs = epochs
            self.model = None
            self.error_loss = nn.MSELoss()

            if not use_sam_embeddings:
                # create a model for feature extraction
                feature_extractor = MyFeatureExtractor(
                    timm_model, timm_pretrained, num_classes #128, use_fc=True
                ).to(self.device)
                self.feature_extractor = feature_extractor
            else:
                self.feature_extractor = sam_model
            self.sam_model = sam_model

            # create the default transformation
            if use_sam_embeddings:
                trans_norm = Transform_To_Models()
            else:
                if feature_extractor.is_transformer:
                    trans_norm = Transform_To_Models(
                            size=feature_extractor.input_size,
                            force_resize=True, keep_aspect_ratio=False
                        )
                else:
                    trans_norm = Transform_To_Models(
                            size=33, force_resize=False, keep_aspect_ratio=True
                        )
            self.trans_norm = trans_norm
            self.use_sam_embeddings = use_sam_embeddings

        def fit_autoencoder(self, x_train):
            autoencoder = Autoencoder(in_shape=x_train.shape[1], enc_shape=2).double().to(self.device)
            optimizer = optim.Adam(autoencoder.parameters())
            x_train = x_train.double()
            train(autoencoder, self.error_loss, optimizer, self.epochs, x_train)
            self.model = autoencoder

        def calculate_error(self, autoencoder, x_input):
            with torch.no_grad():
                autoencoder.eval()
                decoded = autoencoder(x_input)
                mse = self.error_loss(decoded, x_input).item()
                return mse 
        
        def predict_error(self, x_training, autoencoder):
            list_errors = []
            for x in x_training:
                list_errors.append(self.calculate_error(autoencoder, x.unsqueeze(dim=0).double()))
            return list_errors
        
        def get_all_features(self, images):
            """
            Extract feature vectors from the images.
            
            Params
            :images (List<tensor>) images to be used to extract features
            """
            features = []
            # get feature maps from the images
            if self.use_sam_embeddings:
                with torch.no_grad():
                    for img in images:
                        t_temp = self.feature_extractor.get_embeddings(img)
                        features.append(t_temp.squeeze().cpu())
            else:
                with torch.no_grad():
                    for img in images:
                        t_temp = self.feature_extractor(img.unsqueeze(dim=0).to(self.device))
                        features.append(t_temp.squeeze().cpu())
            return features

        def run_filter(self,
            labeled_loader,
            unlabeled_loader, validation_loader,
            dir_filtered_root = None, get_background_samples=True,
            num_classes:float=0):

            labeled_imgs = []
            labeled_labels = []

            for (batch_num, batch) in tqdm(
                enumerate(labeled_loader), total= len(labeled_loader), desc="Extract images"
            ):    
                # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
                # ITERATE: IMAGE
                for idx in list(range(batch[1]['img_idx'].numel())):
                    # get foreground samples (from bounding boxes)
                    imgs_f, labels_f = get_foreground(
                        batch, idx, self.trans_norm,
                        self.use_sam_embeddings
                    )
                    labeled_imgs += imgs_f
                    labeled_labels += labels_f
            all_labeled_features = self.get_all_features(labeled_imgs)
            all_labeled_features = torch.stack(all_labeled_features).double()

            self.fit_autoencoder(all_labeled_features.double())
            distances = self.predict_error(all_labeled_features.double(), self.model)

            # Calculate threshold using IQR 
            Q1 = np.percentile(distances, 25)
            Q3 = np.percentile(distances, 75)
            IQR = Q3 - Q1
            threshold = 1.5 * IQR #1.2 * IQR 
            self.threshold = Q3 + threshold 
            print(self.threshold)
            print(distances)
            self.evaluate(unlabeled_loader, dir_filtered_root, "bbox_results")
            self.evaluate(validation_loader, dir_filtered_root, "bbox_results_val")

        def evaluate(self, dataloader, dir_filtered_root, result_name):
            # go through each batch unlabeled
            distances_all = 0

            # keep track of the img id for every sample created by sam
            imgs_ids = []
            imgs_box_coords = []
            imgs_scores = []

            # 3. Get batch of unlabeled // Evaluating the likelihood of unlabeled data
            for (batch_num, batch) in tqdm(
                enumerate(dataloader), total= len(dataloader), desc="Iterate dataloader"
            ):
                unlabeled_imgs = []
                # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
                # ITERATE: IMAGE
                for idx in tqdm(list(range(batch[1]['img_idx'].numel())), desc="Iterate images"):
                    # get foreground samples (from sam)
                    imgs_s, box_coords, scores = self.sam_model.get_unlabeled_samples(
                        batch, idx, self.trans_norm, self.use_sam_embeddings
                    )
                    unlabeled_imgs += imgs_s

                    # accumulate SAM info (inferences)
                    imgs_ids += [batch[1]['img_orig_id'][idx].item()] * len(imgs_s)
                    imgs_box_coords += box_coords
                    imgs_scores += scores

                # get all features maps using: the extractor + the imgs
                featuremaps_list = self.get_all_features(unlabeled_imgs)
                featuremaps = torch.stack(featuremaps_list) # e.g. [387 x 512]

                # init buffer with distances
                support_set_distances = []
                distances = self.predict_error(featuremaps, self.model)
                print("Test distances:", distances)
                support_set_distances = distances
                
                # accumulate
                if (batch_num == 0):
                    distances_all = torch.Tensor(support_set_distances)
                else:
                    distances_all = torch.cat((distances_all, torch.Tensor(support_set_distances)), 0)

            # transform data 
            scores = []
            for j in range(0, distances_all.shape[0]):
                scores += [distances_all[j].item()]
            scores = np.array(scores).reshape((len(scores),1))

            limit = self.threshold 
            # accumulate results
            results = []
            print("Scores: ", len(scores))
            count = 0
            for index, score in enumerate(scores):
                if(score.item() <= limit):
                    image_result = {
                        'image_id': imgs_ids[index],
                        'category_id': 1, # fix this
                        'score': imgs_scores[index],
                        'bbox': imgs_box_coords[index],
                    }
                    results.append(image_result)
                    count=count+1
            print("Count: ", count)

            if len(results) > 0:
                # write output
                results_file = f"{dir_filtered_root}/{result_name}.json"

                if os.path.isfile(results_file):
                    os.remove(results_file)
                json.dump(results, open(results_file, 'w'), indent=4)