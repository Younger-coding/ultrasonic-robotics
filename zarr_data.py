import os
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# -----------------
# Config (can be overridden externally, e.g., in train_new.py)
# -----------------
LOOKAHEAD_K = 1                 # use t and t+1 by default
TARGET_SIZE = 224               # resize target
MAX_ROTATE_DEG = 0            # rotation range
NOISE_STD = 0.1                 # Gaussian noise std
ENABLE_FLOW = True              # set False if you don't want optical flow
ENABLE_AUG = True               # set False to disable all aug (identity)
ENABLE_FLIP = False             # flips may hurt flow alignment; default off


def _to_tensor_image(np_img: np.ndarray) -> torch.Tensor:
    """(H,W) uint8 -> (1,H,W) float32 in [0,1]."""
    return torch.from_numpy(np_img).float().unsqueeze(0) / 255.0


def _random_affine(img: torch.Tensor,
                   max_rotate: float = MAX_ROTATE_DEG,
                   noise_std: float = NOISE_STD) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    img: (1,H,W) float in [0,1]
    returns: warped_img, affine (2x3 forward), affine_inv (2x3 inverse)
    """
    if (not ENABLE_AUG) or max_rotate == 0.0:
        A = torch.tensor([[1., 0., 0.],
                          [0., 1., 0.]], dtype=img.dtype)
        return img.clone(), A, A

    _, H, W = img.shape
    angle = random.uniform(-max_rotate, max_rotate)
    center = (W * 0.5, H * 0.5)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # 2x3

    img_np = img.squeeze(0).numpy()
    warped = cv2.warpAffine(img_np, M, (W, H),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    warped = torch.from_numpy(warped).unsqueeze(0)

    # optional flips
    if ENABLE_FLIP and random.random() < 0.5:
        warped = torch.flip(warped, dims=[2])  # horizontal
    if ENABLE_FLIP and random.random() < 0.5:
        warped = torch.flip(warped, dims=[1])  # vertical

    # add noise
    warped = warped + torch.randn_like(warped) * noise_std
    warped = warped.clamp(0.0, 1.0)

    # inverse affine (2x3)
    M_3x3 = np.vstack([M, [0, 0, 1]])
    M_inv = np.linalg.inv(M_3x3)[:2, :]
    return warped, torch.from_numpy(M).float(), torch.from_numpy(M_inv).float()


def _compute_flow(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    img1/img2: (1,H,W) float[0,1]
    returns flow: (H,W,2) float32 (dx, dy)
    """
    if not ENABLE_FLOW:
        return torch.zeros(img1.shape[1], img1.shape[2], 2, dtype=torch.float32)

    i1 = (img1.squeeze(0).numpy() * 255).astype("uint8")
    i2 = (img2.squeeze(0).numpy() * 255).astype("uint8")
    flow = cv2.calcOpticalFlowFarneback(
        i1, i2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.1, flags=0
    )
    return torch.from_numpy(flow).float()


class CustomSwinUnetGeometricDataset(Dataset):
    """
    Zarr dataset for geometric/contrastive training.
    Returns two views for t and t+k, with optical flow and affine inverses.
    """
    def __init__(self, zarr_path: str, lookahead_k: int = LOOKAHEAD_K, transform=None):
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr path not found: {zarr_path}")

        import zarr  # local import to keep dependency scoped
        self.root = zarr.open(zarr_path, mode='r')
        self.data_group = self.root['data']
        self.meta_group = self.root['meta']

        self.images = self.data_group['img_current']          # (N,H,W) uint8
        self.poses = self.data_group['current_pose']          # (N,7) float32
        self.episode_ends = self.meta_group['episode_ends'][:]  # array of ends

        self.transform = transform
        self.lookahead_k = lookahead_k
        self.valid_indices = self._preprocess_indices()

        print(f"Raw samples: {self.images.shape[0]}")
        print(f"Valid pairs (t, t+k): {len(self.valid_indices)}")
        print(f"Lookahead k: {self.lookahead_k}")

    def _preprocess_indices(self) -> List[int]:
        valid_indices: List[int] = []
        episode_start = 0
        for end_idx in self.episode_ends:
            max_t_idx = end_idx - self.lookahead_k
            for t_idx in range(episode_start, max_t_idx):
                valid_indices.append(t_idx)
            episode_start = end_idx
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def _resize_if_needed(self, img: torch.Tensor) -> torch.Tensor:
        _, H, W = img.shape
        if H == TARGET_SIZE and W == TARGET_SIZE:
            return img
        img_bchw = img.unsqueeze(0)  # (1,1,H,W)
        img_resized = F.interpolate(img_bchw, size=(TARGET_SIZE, TARGET_SIZE),
                                    mode='bilinear', align_corners=False)
        return img_resized.squeeze(0)  # (1,H,W)

    def __getitem__(self, idx: int):
        t_idx = self.valid_indices[idx]
        tk_idx = t_idx + self.lookahead_k

        np_image_t = self.images[t_idx]    # (H,W) uint8
        np_image_tk = self.images[tk_idx]  # (H,W) uint8
        np_pose_t = self.poses[t_idx]      # (7,)
        np_pose_tk = self.poses[tk_idx]    # (7,)

        image_t = self._resize_if_needed(_to_tensor_image(np_image_t))
        image_tk = self._resize_if_needed(_to_tensor_image(np_image_tk))
        pose_t = torch.from_numpy(np_pose_t).float()
        pose_tk = torch.from_numpy(np_pose_tk).float()

        # two views for each frame
        t_v1, A_t1, Ainv_t1 = _random_affine(image_t)
        t_v2, A_t2, Ainv_t2 = _random_affine(image_t)
        tk_v1, A_tk1, Ainv_tk1 = _random_affine(image_tk)
        tk_v2, A_tk2, Ainv_tk2 = _random_affine(image_tk)

        # optical flow between raw (un-augmented) frames
        flow = _compute_flow(image_t, image_tk)  # (H,W,2)

        sample = {
            'image_t_v1': t_v1, 'affine_t_v1': A_t1, 'affine_t_v1_inv': Ainv_t1,
            'image_t_v2': t_v2, 'affine_t_v2': A_t2, 'affine_t_v2_inv': Ainv_t2,
            'image_tk_v1': tk_v1, 'affine_tk_v1': A_tk1, 'affine_tk_v1_inv': Ainv_tk1,
            'image_tk_v2': tk_v2, 'affine_tk_v2': A_tk2, 'affine_tk_v2_inv': Ainv_tk2,
            'flow': flow, 'pose_t': pose_t, 'pose_tk': pose_tk
        }
        return sample
