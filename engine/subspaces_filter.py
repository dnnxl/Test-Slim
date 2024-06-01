import os
import json
import torch
import numpy as np
from engine.feature_extractor import MyFeatureExtractor
from data import get_foreground, Transform_To_Models, get_background
from tqdm import tqdm
import cv2
from sklearn.preprocessing import PowerTransformer

class SubspacesFilter:

    def __init__(self,
                timm_model=None,
                timm_pretrained=True,
                num_classes=1,
                sam_model=None,
                use_sam_embeddings=False,
                is_single_class=True,
                device="cpu"):
        """
        Raises:
            ValueError: if the backbone is not a feature extractor,
            i.e. if its output for a given image is not a 1-dim tensor.
        """
        self.mean = None
        self.inv_cov = None
        self.device = device
        self.num_samples = None
        self.num_classes = num_classes
        self.timm_model = timm_model
        self.sam_model = sam_model
        self.is_single_class = is_single_class

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

    

    def fit_two_cls(self, embeddings):
        all_hyper_planes, means = self.create_subspace(embeddings)
        return all_hyper_planes, means
    
    def predict_two_cls(self, feature, mean, all_hyper_planes):
        score = self.projection_metric(feature, all_hyper_planes, mean)
        return score
    
    def fit(self, embeddings):
        self.all_hyper_planes, self.means = self.create_subspace(embeddings)
    
    def predict(self, target_features):
        list_scores = []
        for feature in target_features:
            score = self.projection_metric(feature, self.all_hyper_planes, self.means)
            list_scores.append(score)
        return torch.Tensor(list_scores)

    def create_subspace(self, supportset_features, k=1):
        all_support_within_class_t = supportset_features
        meann = torch.mean(all_support_within_class_t, dim=0)
        all_support_within_class = all_support_within_class_t - meann
        all_support_within_class = all_support_within_class
        uu, s, v = torch.svd(all_support_within_class.double(), some=False)

        print("uu: ", uu.shape)
        print("s ", s.shape)
        print("v: ", v.shape)
        all_hyper_planes = v[:k, :].T
        means = meann
        return all_hyper_planes, means


    def projection_metric(self, x, hyper_planes, mean):
        eps = 1e-12
        target_features_expanded = (x - mean).unsqueeze(0).T
        projected_query_j = torch.matmul(hyper_planes.float(), (hyper_planes.float().T @ target_features_expanded.float()))
        projected_query_j = torch.squeeze(projected_query_j).T + mean
        projected_query_dist_inter = x - projected_query_j
        distance = torch.sqrt(torch.sum(projected_query_dist_inter * projected_query_dist_inter, dim=-1) + eps)
        return distance

    """
    def projection_metric(self, target_features, hyperplanes, mu):
        eps = 1e-12

        h_plane_j =  hyperplanes#.unsqueeze(0)
        target_features_expanded = (target_features - mu)
        projected_query_j = torch.mm(h_plane_j.T, torch.mm(h_plane_j, target_features_expanded.unsqueeze(0)))
        projected_query_j = torch.squeeze(projected_query_j) + mu
        projected_query_dist_inter = target_features - projected_query_j

        #Training per epoch is slower but less epochs in total
        distance = torch.sqrt(torch.sum(projected_query_dist_inter * projected_query_dist_inter, dim=-1) + eps) # norm ||.||
        #print("projected_query_dist_inter shape: ", projected_query_dist_inter.shape)
        #print("sum shape: ", torch.sum(projected_query_dist_inter * projected_query_dist_inter, dim=-1))

        #Training per epoch is faster but more epochs in total
        #query_loss = -torch.sum(projected_query_dist_inter * projected_query_dist_inter, dim=-1) # Squared norm ||.||^2
        return distance
    """

    def sample_and_calculate_stats(self, data, num_samples=10):
        mean_list = []
        covariance_list = []

        for _ in range(num_samples):
            # Randomly sample 5 rows with replacement
            sampled_rows = np.random.choice(data.shape[0], int(data.shape[0]/4), replace=True)

            # Calculate mean and covariance for the sampled rows
            sample_mean = np.mean(data[sampled_rows, :], axis=0)
            sample_covariance = np.cov(data[sampled_rows, :], rowvar=False)

            # Append to the lists
            mean_list.append(sample_mean)
            covariance_list.append(sample_covariance)

        return mean_list, covariance_list

    def distribution_calibration(self, query, base_means, base_cov, k,alpha=0.21):
        dist = []
        for i in range(len(base_means)):
            dist.append(np.linalg.norm(query-base_means[i]))
        print(" query[np.newaxis, :]: ",  query[np.newaxis, :].shape)
        print(" np.array(base_means)[index]: ",  np.array(base_means).shape)

        index = np.argpartition(dist, k)[:k]
        mean = np.concatenate([np.array(base_means)[index], query])
        calibrated_mean = np.mean(mean, axis=0)
        calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha

        return calibrated_mean, calibrated_cov

    def run_filter(self,
        labeled_loader,
        unlabeled_loader,
        dir_filtered_root = None, get_background_samples=True,
        num_classes:float=0):

        # 1. Get feature maps from the labeled set
        labeled_imgs = []
        labeled_labels = []

        #all_imgs_context = []
        #all_label_context = []
        
        #back_imgs_context = []
        #back_label_context = []

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

                #all_imgs_context += imgs_f
                #all_label_context += labels_f

                #if get_background_samples:
                #    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

                #    imgs_b, labels_b = get_background(
                #        batch, idx, self.trans_norm, ss, 
                #        num_classes, self.use_sam_embeddings)
                #    back_imgs_context += imgs_b
                #    back_label_context += labels_b
        #if len(labeled_imgs) < 100:
        #    all_imgs_context = labeled_imgs + back_imgs_context
        #else:
        #all_imgs_context = labeled_imgs + back_imgs_context[:len(labeled_imgs)]


        #print("Len labeled imgs: ", len(labeled_imgs))
        # labels start from index 1 to n, translate to start from 0 to n.
        labels = [int(i-1) for i in labeled_labels]

        # get all features maps using: the extractor + the imgs
        feature_maps_list = self.get_all_features(labeled_imgs)
        #all_feature_maps_list = self.get_all_features(all_imgs_context)
        #back_imgs_context = self.get_all_features(back_imgs_context)

        #print("Feature map list: ", feature_maps_list)
        #print("feature_maps_list imgs: ", feature_maps_list)
        #----------------------------------------------------------------
        if self.is_single_class:
            labels = np.zeros(len(feature_maps_list))
        else:
            labels = np.array(labels)

        # 2. Calculating the mean prototype for the labeled data
        #----------------------------------------------------------------
        #support_features_imgs_1 = torch.stack(imgs_1)
        #support_features_imgs_2 = torch.stack(imgs_2)
        support_features = torch.stack(feature_maps_list)
        #all_support_features = torch.stack(all_feature_maps_list)        
        #back_imgs_context = torch.stack(back_imgs_context)

        print("Support features shape: ", support_features.shape)
        #print("all_support_features shape: ", all_support_features.shape)

        # -----------------------------------------------------------
        # NORMALIZE THE TENSORS
        #base_means, base_cov = self.sample_and_calculate_stats(support_features.detach().numpy(), num_samples=10)
        #pt = PowerTransformer(method='yeo-johnson', standardize=False)
        #support_features = pt.fit_transform(support_features.detach().numpy())
        #mean, cov = self.distribution_calibration(support_features, base_means, base_cov, k=2)
        #sampled_data = np.random.multivariate_normal(mean=mean, cov=cov, size=int(support_features.shape[0]*(1/2)))
        #support_features_aug = np.concatenate([support_features, sampled_data])
        #support_features = torch.tensor(support_features_aug)

        # Save the tensor to the specified file
        #torch.save(support_features, './tensor_foreground.pt')
        #torch.save(back_imgs_context, './tensor_background.pt')

        #support_features = torch.sigmoid(support_features)
        #print("Support features mean: ", support_features.mean())
        #print("Features size: ", support_features.shape)

        # 3. Calculating the sigma (covariance matrix), the distances 
        # with respect of the support features and get the threshold
        #----------------------------------------------------------------
        if self.is_single_class:
            
            self.fit(support_features)

            distances = self.predict(support_features)

            # Using IQR
            Q1 = np.percentile(distances.numpy(), 25)
            Q3 = np.percentile(distances.numpy(), 75)
            IQR = Q3 - Q1
            threshold = 1.5 * IQR
            self.threshold = Q3 + threshold 
            print("Q1: ", Q1)
            print("Q3: ", Q3)

            # Using Median Absolute Deviation to set the threshold
            #median = np.median(distances.numpy())
            #absolute_deviations = np.abs(distances.numpy() - median)
            #mad = np.median(absolute_deviations)
            #self.threshold = median + 3 * mad

            #self.threshold = max(distances).item()
            #print("Distances: ", distances)
            mean_value = torch.mean(distances).item()
            std_deviation = torch.std(distances).item()
            print(distances)
            print("Mean predicted:", mean_value)
            print("Standard Deviation predicted:", std_deviation)

            print("Threshold: ", self.threshold)
        #else:
            #self.all_hyper_planes_cls_0, self.means_cls_0 = self.fit_two_cls(support_features)
            #self.all_hyper_planes_cls_1, self.means_cls_1 = self.fit_two_cls(back_imgs_context)
            #print("Two classes Subspaces filter")


            
        # go through each batch unlabeled
        distances_all = 0

        # keep track of the img id for every sample created by sam
        imgs_ids = []
        imgs_box_coords = []
        imgs_scores = []

        # 3. Get batch of unlabeled // Evaluating the likelihood of unlabeled data
        for (batch_num, batch) in tqdm(
            enumerate(unlabeled_loader), total= len(unlabeled_loader), desc="Iterate batch unlabeled"
        ):
            unlabeled_imgs = []
            # every batch is a tuple: (torch.imgs , metadata_and_bboxes)
            # ITERATE: IMAGE
            for idx in tqdm(list(range(batch[1]['img_idx'].numel())), desc="Iterate images"):
            #for idx in list(range(batch[1]['img_idx'].numel())):
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
            #torch.save(featuremaps, './tensor_unlabeled.pt')

            # ----------------------------------------------------
            #featuremaps = torch.Tensor(pt.transform(featuremaps.detach().numpy()))

            # init buffer with distances
            support_set_distances = []
            if self.is_single_class:
                distances = self.predict(featuremaps)
                support_set_distances = distances
            else:
                for fea in featuremaps:
                    score_0 = self.predict_two_cls(fea, self.means_cls_0, self.all_hyper_planes_cls_0)
                    score_1 = self.predict_two_cls(fea, self.means_cls_1, self.all_hyper_planes_cls_1)
                    if  score_0 >= score_1:
                        support_set_distances.append(0)
                    else:
                        support_set_distances.append(1)

            # accumulate
            if (batch_num == 0):
                distances_all = torch.Tensor(support_set_distances)
            else:
                distances_all = torch.cat((distances_all, support_set_distances), 0)

        # transform data 
        scores = []
        for j in range(0, distances_all.shape[0]):
            scores += [distances_all[j].item()]
        scores = np.array(scores).reshape((len(scores),1))

        
        # accumulate results
        results = []
        print("Scores: ", len(scores))
        count = 0
        for index, score in enumerate(scores):
            if self.is_single_class:
                limit = self.threshold 
                if(score.item() <= limit):
                    image_result = {
                        'image_id': imgs_ids[index],
                        'category_id': 1, # fix this
                        'score': imgs_scores[index],
                        'bbox': imgs_box_coords[index],
                    }
                    results.append(image_result)
                    count=count+1
                    print("Score.item(): ", score.item())
            else:
                if(score.item() == 0):
                    image_result = {
                        'image_id': imgs_ids[index],
                        'category_id': 1, # fix this
                        'score': imgs_scores[index],
                        'bbox': imgs_box_coords[index],
                    }
                    results.append(image_result)
                    count=count+1
                    print("Score.item(): ", score.item())
        print("Count: ", count)

        if len(results) > 0:
            # write output
            results_file = f"{dir_filtered_root}/bbox_results.json"
            if os.path.isfile(results_file):
                os.remove(results_file)
            json.dump(results, open(results_file, 'w'), indent=4)

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
                for img in tqdm(images, desc="Extract features"):
                    t_temp = self.feature_extractor.get_embeddings(img)
                    features.append(t_temp.squeeze().cpu())
        else:
            with torch.no_grad():
                for img in tqdm(images, desc="Extract features"):
                    t_temp = self.feature_extractor(img.unsqueeze(dim=0).to(self.device))
                    features.append(t_temp.squeeze().cpu())
        return features