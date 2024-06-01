from typing import List
from torch import Tensor, nn

from .fewshot_model import FewShot
from .fewshot_utils import compute_prototypes, power_transform

import torch
import numpy as np

MAXIMUM_SINKHORN_ITERATIONS = 1000

class PTMAP(FewShot):
    """
    Yuqing Hu, Vincent Gripon, Stéphane Pateux.
    "Leveraging the Feature Distribution in Transfer-based Few-Shot Learning" (2020)
    https://arxiv.org/abs/2006.03806

    Query soft assignments are computed as the optimal transport plan to class prototypes.
    At each iteration, prototypes are fine-tuned based on the soft assignments.
    This is a transductive method.
    """

    def __init__(
        self,
        *args,
        is_single_class, use_sam_embeddings,
        fine_tuning_steps: int = 10,
        fine_tuning_lr: float = 0.2,
        lambda_regularization: float = 10.0,
        power_factor: float = 0.5,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr
        self.lambda_regularization = lambda_regularization
        self.power_factor = power_factor
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
        #imgs_1, imgs_2, lbl_1, lbl_2 = train_test_split(
        #    support_images, y_labels,
        #    train_size = 0.6, stratify=y_labels,
        #    shuffle=True # shuffle the data before splitting
        #)
        #---------------------------------------

        # get feature maps from the images
        for img in support_images:
            if self.use_sam_embeddings:
                t_temp = self.get_embeddings_sam(img)
            else:
                t_temp = self.get_embeddings_timm(img)
            support_features.append(t_temp.squeeze().cpu())

        # get prototypes and save them into cuda memory
        self.support_features = torch.stack(support_features)
        if self.is_single_class:
            #prototypes = compute_prototypes_singleclass(self.support_features)
            #prototypes = prototypes.unsqueeze(dim=0) # 2D tensor
            print("Not implemented!")
        else:
            self.support_labels = torch.Tensor(y_labels)
            prototypes = compute_prototypes(self.support_features, self.support_labels)
        self.prototypes = prototypes.to(self.device)

        #---------------------------------------
        # get feature maps from the images
        #result = []
        #for img in imgs_2:
        #    if self.use_sam_embeddings:
        #        t_temp = self.get_embeddings_sam(img)
        #    else:
        #        t_temp = self.get_embeddings_timm(img)
        #    result.append(t_temp.squeeze().cpu())

        # get prototypes and save them into cuda memory
        #result = torch.stack(result)
        #if self.is_single_class:
        #    self._calculate_statistics(imgs_2)
        #---------------------------------------

    #def _calculate_statistics(
    #    self,
    #    imgs: List,
    #) -> Tensor:
        """
        Get metrics from the embeddings.

        Params
        :imgs (tensor) -> embedding to calculate metrics.
        """
    #    assert self.is_single_class, "This method can be used just in single class"
    #    scores = []
    #    for img in imgs:
    #        print(img.shape)
    #        score = self.forward(img)
    #        scores.append(score.cpu().item())
    #    self.mean = scipy.mean(scores)
    #    self.std = scipy.std(scores)


    def forward(
        self,
        query_image: Tensor,
    ) -> Tensor:
        """
        Predict query soft assignments following Algorithm 1 of the paper.
        """
        if self.use_sam_embeddings:
            z_query = self.get_embeddings_sam(query_image)
        else:
            z_query = self.get_embeddings_timm(query_image)
        z_query = power_transform(z_query, self.power_factor)
        #query_features = self.compute_features(query_images)

        support_assignments = nn.functional.one_hot(
            self.support_labels.long(), len(self.prototypes)
        ).float()
        for _ in range(self.fine_tuning_steps):
            query_soft_assignments = self.compute_soft_assignments(z_query)

            all_features = torch.cat([self.support_features, z_query], 0)
            all_assignments = torch.cat(
                [support_assignments, query_soft_assignments], dim=0
            )

            self.update_prototypes(all_features, all_assignments)
        return self.compute_soft_assignments(z_query)

    def compute_features(self, images: Tensor) -> Tensor:
        """
        Apply power transform on features following Equation (1) in the paper.
        Args:
            images: images of shape (n_images, **image_shape)
        Returns:
            features of shape (n_images, feature_dimension) with power-transform.
        """
        features = super().compute_features(images)
        return power_transform(features, self.power_factor)

    def compute_soft_assignments(self, query_features: Tensor) -> Tensor:
        """
        Compute soft assignments from queries to prototypes, following Equation (3) of the paper.
        Args:
            query_features: query features, of shape (n_queries, feature_dim)

        Returns:
            soft assignments from queries to prototypes, of shape (n_queries, n_classes)
        """

        distances_to_prototypes = (
            torch.cdist(query_features, self.prototypes) ** 2
        )  # [Nq, K]
        #if torch.isnan(distances_to_prototypes).any().item():
        #    print("distances_to_prototypes", query_features.shape)
        soft_assignments = self.compute_optimal_transport(
            distances_to_prototypes, epsilon=1e-6
        )

        return soft_assignments

    def compute_optimal_transport(
        self, cost_matrix: Tensor, epsilon: float = 1e-6
    ) -> Tensor:
        """
        Compute the optimal transport plan from queries to prototypes using Sinkhorn-Knopp algorithm.
        Args:
            cost_matrix: euclidean distances from queries to prototypes,
                of shape (n_queries, n_classes)
            epsilon: convergence parameter. Stop when the update is smaller than epsilon.
        Returns:
            transport plan from queries to prototypes of shape (n_queries, n_classes)
        """

        instance_multiplication_factor = cost_matrix.shape[0] // cost_matrix.shape[1]
        transport_plan = torch.exp(-self.lambda_regularization * cost_matrix)
        transport_plan /= transport_plan.sum(dim=(0, 1), keepdim=True)

        for _ in range(MAXIMUM_SINKHORN_ITERATIONS):
            per_class_sums = transport_plan.sum(1)
            transport_plan *= (1 / (per_class_sums + 1e-10)).unsqueeze(1)
            transport_plan *= (
                instance_multiplication_factor / (transport_plan.sum(0) + 1e-10)
            ).unsqueeze(0)
            if torch.max(torch.abs(per_class_sums - transport_plan.sum(1))) < epsilon:
                break
        return transport_plan

    def update_prototypes(self, all_features, all_assignments) -> None:
        """
        Update prototypes by weigh-averaging the features with their soft assignments,
            following Equation (6) of the paper.
        Args:
            all_features: concatenation of support and query features,
                of shape (n_support + n_query, feature_dim)
            all_assignments: concatenation of support and query soft assignments,
                of shape (n_support + n_query, n_classes)-
        """
        new_prototypes = (all_assignments.T @ all_features) / all_assignments.sum(
            0
        ).unsqueeze(1)

        delta = new_prototypes - self.prototypes
        self.prototypes += self.fine_tuning_lr * delta

    @staticmethod
    def is_transductive() -> bool:
        return True