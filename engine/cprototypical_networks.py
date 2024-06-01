"""
See original implementation (quite far from this one)
at https://github.com/jakesnell/prototypical-networks
"""
from typing import List
import torch
import numpy as np
from torch import Tensor
from .fewshot_model import FewShot
from .fewshot_utils import compute_prototypes, compute_prototypes_singleclass
# import statistics
import scipy
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

class CPrototypicalNetworks(FewShot):
    """
    Jake Snell, Kevin Swersky, and Richard S. Zemel.
    "Prototypical networks for few-shot learning." (2017)
    https://arxiv.org/abs/1703.05175

    Prototypical networks extract feature vectors for both support and query images. Then it
    computes the mean of support features for each class (called prototypes), and predict
    classification scores for query images based on their euclidean distance to the prototypes.
    """

    def __init__(self, is_single_class, use_sam_embeddings, device="cpu", *args, **kwargs):
        """
        Raises:
            ValueError: if the backbone is not a feature extractor,
            i.e. if its output for a given image is not a 1-dim tensor.
        """
        super().__init__(*args, **kwargs)
        self.mean = None
        self.std = None
        self.stats_computed = None
        self.device = device
        self.num_samples = None
        self.is_single_class = is_single_class
        self.use_sam_embeddings = use_sam_embeddings

    def get_embeddings_timm(self, img):
        """
        Returns the embeddings from the backbone which is a timm model.
        """
        with torch.no_grad():
            x = self.backbone.forward(img.unsqueeze(dim=0).to(self.device))
        return x
    
    def get_embeddings_sam(self, img):
        """
        Returns the embeddings from the backbone which SAM.
        """
        with torch.no_grad():
            x = self.backbone.get_embeddings(img)
        return x

    def process_support_set(
        self,
        support_images: List,
        support_labels: List = None,
    ):
        """
        Overrides process_support_set of FewShotClassifier.
        Extract feature vectors from the support set and store class prototypes.

        Params
        :support_images (tensor) -> images of the support set
        :support_labels (tensor) <Optional> -> labels of support set images
        """
        support_features = []
        self.num_samples = len(support_images)

        #---------------------------------------
        # split the ids
        if support_labels is None:
            y_labels = np.zeros(len(support_images))
        else:
            y_labels = np.array(support_labels)
        
        
        # Split for validation 
        imgs_1, imgs_2, lbl_1, lbl_2 = train_test_split(
            support_images, y_labels, 
            train_size = 0.6,
            shuffle=True # shuffle the data before splitting
        )
        #---------------------------------------
        

        # get feature maps from the images
        for img in imgs_1:
            if self.use_sam_embeddings:
                t_temp = self.get_embeddings_sam(img)
            else:
                t_temp = self.get_embeddings_timm(img)
            support_features.append(t_temp.squeeze().cpu())
        
        # get prototypes and save them into cuda memory
        support_features = torch.stack(support_features)

        if self.is_single_class:
            support_labels = self.clusterize_embeddings(support_features, 2)
            prototypes = compute_prototypes(support_features, support_labels)
        else:
            support_labels = torch.Tensor(lbl_1)
            prototypes = compute_prototypes(support_features, support_labels)
        self.prototypes = prototypes.to(self.device)

        #---------------------------------------
        if self.is_single_class:
            self.stats_computed = self._calculate_statistics(self.prototypes, support_features, support_labels)
        #---------------------------------------

    def _calculate_statistics(
        self,
        prototypes,
        support_features, support_labels # lista de embedding for prototype

    ) -> Tensor:
        """     
        Get metrics from the embeddings.

        Params
        :imgs (tensor) -> embedding to calculate metrics.
        """
        scores = []
        for index, prototype in enumerate(prototypes):
            support_index = support_features[torch.nonzero(support_labels == float(index))]
            for embedding in support_index:
                score = self.forward_prototype(prototype.unsqueeze(0), embedding.unsqueeze(0))
                scores.append((index, score.cpu().item()))

        scores = torch.tensor(scores)
        unique_prototypes_index = torch.unique(scores[:, 0])

        list_mean_std = []
        # Compute mean and standard deviation for each prototype value
        for prototype_index in unique_prototypes_index:
            filtered_tensor = scores[scores[:, 0] == prototype_index]
            values = filtered_tensor[:, 1]
            mean_value = torch.mean(values.float())
            std_value = torch.std(values.float())
            list_mean_std.append((prototype_index.item(), mean_value.item(), std_value.item()))

        return list_mean_std



    def clusterize_embeddings(self, embeddings: Tensor, n_clusters: int) -> List[Tensor]:
        # Instantiate KMeans with desired number of clusters
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(embeddings)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        #concatenated_tensor = torch.cat((embeddings, torch.Tensor(labels).unsqueeze(1)), dim=1)
        #return concatenated_tensor
        return torch.Tensor(labels)

    def forward_prototype(self, prototype, embedding):
        dist = torch.cdist(embedding, prototype)
        return dist

    def forward(
        self,
        query_image,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Predict query labels based on their distance to class prototypes in the feature space.
        Classification scores are the negative of euclidean distances.

        Params
        :query_image (tensor) -> img to be processed.
        Return
        :a prediction of classification scores for query images
        """
        if self.use_sam_embeddings:
            z_query = self.get_embeddings_sam(query_image)
        else:
            z_query = self.get_embeddings_timm(query_image)

        # Compute the euclidean distance from queries to prototypes
        dist = torch.cdist(z_query, self.prototypes)

        # Use it to compute classification scores
        # FEW SHOT ALWAYS LOOK FOR THE MAX, AND SINCE DISTANCES ARE ALWAYS 
        # POSITIVE, WE NEGATE THE RESULTS IN ORDER TO LOOK FOR THE MAX
        if not self.is_single_class:
            dist = -dist 
        return self.softmax_if_specified(dist)

    @staticmethod
    def is_transductive() -> bool:
        return False


