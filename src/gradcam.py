"""Grad-CAM helpers (prototype-distance target)."""

from typing import List, Sequence, Tuple

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class PrototypeDistanceTarget:
    """Grad-CAM target: negative L2 distance to the label's class centroid.

    Maximising this scalar is equivalent to minimising distance to the prototype,
    matching what the classifier optimises at inference.
    """

    def __init__(self, centroids: torch.Tensor, label: int):
        self.centroids = centroids
        self.label = label

    def __call__(self, embedding):
        centroid = self.centroids[self.label]
        return -torch.norm(embedding - centroid, dim=-1)


def _last_conv(model) -> torch.nn.Module:
    target = None
    for _, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target = module
    if target is None:
        raise RuntimeError("No Conv2d layer found in model.backbone.")
    return target


def compute_gradcam_cams(
    model,
    dataset,
    proto_clf,
    device,
    indices: Sequence[int],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Compute Grad-CAM overlays for the given dataset indices.

    Returns (images, cam_images, labels):
      images: deprocessed RGB arrays in [0,1], shape (H,W,3), float.
      cam_images: uint8 overlays produced by show_cam_on_image.
      labels: integer class labels per sample.
    """
    target_layer = _last_conv(model)
    centroids = proto_clf.prototypes.to(device)
    cam = GradCAM(model=model, target_layers=[target_layer])

    images, cam_images, labels = [], [], []
    for idx in indices:
        img_tensor, label = dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        targets = [PrototypeDistanceTarget(centroids, label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)
        cam_img = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        images.append(img_np)
        cam_images.append(cam_img)
        labels.append(int(label))

    return images, cam_images, labels
