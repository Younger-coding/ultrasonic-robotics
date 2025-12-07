import torch
import torch.nn.functional as F

__all__ = [
    "quat_pos_to_matrix",
    "calculate_relative_transform",
    "affine_to_grid",
    "flow_to_grid",
    "project_feature_map",
    "_invert_affine_2x3",
    "pixel_affine_to_normalized",
    "warp_v2_to_v1",
]


def quat_pos_to_matrix(pose_7d: torch.Tensor) -> torch.Tensor:
    """(x,y,z,qx,qy,qz,qw) -> [B,4,4] homogeneous matrix."""
    B = pose_7d.shape[0]
    device = pose_7d.device
    dtype = pose_7d.dtype

    t = pose_7d[:, :3]
    q = pose_7d[:, 3:]
    q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-8)
    qx, qy, qz, qw = q.unbind(dim=1)

    R_mat = torch.stack([
        1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw),
        2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
        2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)
    ], dim=1).view(B, 3, 3)

    M = torch.zeros(B, 4, 4, device=device, dtype=dtype)
    M[:, :3, :3] = R_mat
    M[:, :3, 3] = t
    M[:, 3, 3] = 1.0
    return M


def calculate_relative_transform(pose_t: torch.Tensor, pose_tk: torch.Tensor) -> torch.Tensor:
    """Relative transform M_rel = M_tk @ inv(M_t)."""
    M_t = quat_pos_to_matrix(pose_t)
    M_tk = quat_pos_to_matrix(pose_tk)
    M_t_inv = torch.inverse(M_t)
    return torch.bmm(M_tk, M_t_inv)


def affine_to_grid(affine_2x3: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """[B,2,3] or [2,3] (already normalized) -> grid [B,H,W,2] in [-1,1]."""
    if affine_2x3.dim() == 2:
        affine_2x3 = affine_2x3.unsqueeze(0)
    B = affine_2x3.size(0)
    return F.affine_grid(affine_2x3, [B, 1, H, W], align_corners=True)


def flow_to_grid(flow: torch.Tensor) -> torch.Tensor:
    """flow: [B,H,W,2] or [H,W,2] (dx,dy in pixels) -> grid [B,H,W,2] in [-1,1]."""
    if flow.dim() == 3:
        flow = flow.unsqueeze(0)
    B, H, W, _ = flow.shape
    device = flow.device
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    xx = xx.float().unsqueeze(0).expand(B, -1, -1)
    yy = yy.float().unsqueeze(0).expand(B, -1, -1)
    x_new = xx + flow[..., 0]
    y_new = yy + flow[..., 1]
    x_norm = 2.0 * (x_new / max(W - 1, 1)) - 1.0
    y_norm = 2.0 * (y_new / max(H - 1, 1)) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def pixel_affine_to_normalized(affine_2x3: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Convert pixel-based affine (as from cv2.getRotationMatrix2D) to normalized theta for affine_grid (align_corners=True).
    """
    if affine_2x3.dim() == 2:
        affine_2x3 = affine_2x3.unsqueeze(0)
    sx = (W - 1) / 2.0
    sy = (H - 1) / 2.0
    out = torch.zeros_like(affine_2x3)
    out[:, 0, 0] = affine_2x3[:, 0, 0]
    out[:, 0, 1] = affine_2x3[:, 0, 1] * (sy / sx)
    out[:, 0, 2] = affine_2x3[:, 0, 0] + affine_2x3[:, 0, 1] * (sy / sx) + affine_2x3[:, 0, 2] / sx - 1.0
    out[:, 1, 0] = affine_2x3[:, 1, 0] * (sx / sy)
    out[:, 1, 1] = affine_2x3[:, 1, 1]
    out[:, 1, 2] = affine_2x3[:, 1, 0] * (sx / sy) + affine_2x3[:, 1, 1] + affine_2x3[:, 1, 2] / sy - 1.0
    return out


def project_feature_map(S: torch.Tensor,
                        flow: torch.Tensor = None,
                        affine: torch.Tensor = None,
                        M_rel: torch.Tensor = None) -> torch.Tensor:
    """Warp features by flow or affine; M_rel placeholder."""
    B, C, H, W = S.shape
    if flow is not None:
        grid = flow_to_grid(flow.to(S.device))
        return F.grid_sample(S, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    if affine is not None:
        theta = pixel_affine_to_normalized(affine.to(S.device), H, W)
        grid = affine_to_grid(theta, H, W)
        return F.grid_sample(S, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    if M_rel is not None:
        return S.clone()
    return S.clone()


def _invert_affine_2x3(affine_2x3: torch.Tensor) -> torch.Tensor:
    """Invert 2x3 affine (batch aware)."""
    if affine_2x3.dim() == 2:
        affine_2x3 = affine_2x3.unsqueeze(0)
    B = affine_2x3.size(0)
    device = affine_2x3.device
    inv_list = []
    for i in range(B):
        A = affine_2x3[i]
        A3 = torch.eye(3, device=device, dtype=A.dtype)
        A3[:2, :] = A
        A3_inv = torch.inverse(A3)
        inv_list.append(A3_inv[:2, :])
    return torch.stack(inv_list, dim=0)


def warp_v2_to_v1(feat_v2: torch.Tensor,
                  affine_v2_inv: torch.Tensor,
                  flow: torch.Tensor,
                  affine_v1_inv: torch.Tensor) -> torch.Tensor:
    """
    Chain warp: remove view2 aug (view->orig) -> flow t+k->t -> apply view1 aug (orig->view1).
    """
    B, C, H, W = feat_v2.shape
    device = feat_v2.device
    # remove view2 aug: need forward matrix
    affine_v2 = _invert_affine_2x3(affine_v2_inv)
    theta_v2 = pixel_affine_to_normalized(affine_v2, H, W)
    grid_unaug = affine_to_grid(theta_v2.to(device), H, W)
    feat_tk_orig = F.grid_sample(feat_v2, grid_unaug, mode='bilinear', padding_mode='zeros', align_corners=True)

    # flow warp t+k -> t
    grid_flow = flow_to_grid(flow.to(device))
    feat_t_orig = F.grid_sample(feat_tk_orig, grid_flow, mode='bilinear', padding_mode='zeros', align_corners=True)

    # apply view1 aug: need forward matrix
    affine_v1 = _invert_affine_2x3(affine_v1_inv)
    theta_v1 = pixel_affine_to_normalized(affine_v1, H, W)
    grid_v1 = affine_to_grid(theta_v1.to(device), H, W)
    feat_v1 = F.grid_sample(feat_t_orig, grid_v1, mode='bilinear', padding_mode='zeros', align_corners=True)
    return feat_v1
