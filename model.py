import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class NetVLADModel(nn.Module):
    """
    Your Task) Complete the function in model.py.

    - Use VGG16 CNN to convert the input image into a high-dimensional feature map.
    - Perform soft-assignment to cluster centroids and aggregate residual vectors into a global descriptor.
    - Normalize the final descriptor for retrieval tasks.

    Returns:
        Tensor: Global image descriptor of shape (N, K*D)
    """
    def __init__(self, num_clusters=16, dim=512):
        super().__init__()

        # TODO:
        # 1. Load a pretrained VGG16 backbone.
        #    * Remove the final pooling/classification layers.
        #    * Keep only convolutional feature extractor.
        # 2. Define a NetVLAD pooling layer with `num_clusters` and feature dimension `dim`.
        #    * Cluster centers as learnable parameters.
        #    * 1x1 convolution for soft-assignment.
        #
        # Use exactly these attribute names (base.init_centroids() initializes them with k-means):
        #    self.num_clusters : int
        #    self.backbone     : VGG16 conv layers, image (N,3,H,W) -> feature map (N,dim,h,w)
        #    self.centroids    : nn.Parameter of shape (num_clusters, dim)
        #    self.assign       : nn.Conv2d(dim, num_clusters, kernel_size=1, bias=True)
        self.num_clusters = num_clusters
        self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:-1]
        self.centroids = nn.Parameter(torch.randn(num_clusters, dim)) #c_k
        self.assign = nn.Conv2d(dim, num_clusters, kernel_size=1, bias=True) #a_k 를 만들기 위한 conv layer

        pass

    def forward(self, x):
        """
        Parameters:
            x (Tensor): Input image batch (N, 3, H, W)

        Returns:
            Tensor: Normalized global descriptor (N, K*D)
        """
        # TODO:
        # 1. Extract feature maps using backbone.
        #    (use models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1) as a backbone)
        # 2. L2-normalize each local descriptor (over the channel dimension).
        # 3. Apply NetVLAD pooling to aggregate descriptors (soft-assignment with self.assign,
        #    residuals to self.centroids), then intra-normalization.
        # 4. L2-normalize the final descriptor.

        feature = self.backbone(x)
        norm_feature = F.normalize(feature, p=2, dim=1) #(N, D, h, w)
        a  = F.softmax(self.assign(norm_feature), dim=1) #(N, K, h, w)
        a = a.flatten(2, 3) #(N, K, h*w)
        f = norm_feature.flatten(2, 3) # (N, D, h*w)
        vlad = torch.einsum('nki,ndi->nkd', a, f) - torch.einsum('nki,kd->nkd', a, self.centroids) #(N, K, D)
        vlad = F.normalize(vlad, p=2, dim=2) #intra-normalization
        vlad = vlad.flatten(1) #(N, K*D)
        vlad = F.normalize(vlad, p=2, dim=1) #final normalization

        return vlad
