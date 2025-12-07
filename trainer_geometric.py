import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from geometric_utils import calculate_relative_transform, project_feature_map, affine_to_grid, flow_to_grid


def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)


def _iic_mutual_information(prob1: torch.Tensor, prob2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    prob1/prob2: [B, K, H, W], already softmaxed
    returns: scalar MI (batch averaged)
    """
    B, K, H, W = prob1.shape
    p1 = prob1.flatten(2)  # [B, K, N]
    p2 = prob2.flatten(2)  # [B, K, N]
    joint = torch.bmm(p1, p2.transpose(1, 2))  # [B, K, K]
    joint = joint / joint.sum(dim=(1, 2), keepdim=True).clamp_min(eps)
    pi = joint.sum(dim=2, keepdim=True)
    pj = joint.sum(dim=1, keepdim=True)
    mi = joint * (torch.log(joint + eps) - torch.log(pi + eps) - torch.log(pj + eps))
    mi = mi.sum(dim=(1, 2))  # [B]
    return mi.mean()  # scalar


def _invert_affine_2x3(affine_2x3: torch.Tensor) -> torch.Tensor:
    """
    将 2x3 仿射矩阵求逆（先补成3x3再求逆），返回 2x3。
    """
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
    将 view2 特征对齐到 view1 坐标：
    1) 去除 view2 增广：使用 forward 矩阵 (逆的逆)
    2) 用光流从 t+k 原坐标 warp 到 t 原坐标
    3) 施加 view1 增广（用 view1 的 inverse 矩阵生成 grid）
    """
    B, C, H, W = feat_v2.shape
    device = feat_v2.device

    # step1: 去除 view2 增广（aug -> 原 t+k），需要正向矩阵
    affine_v2 = _invert_affine_2x3(affine_v2_inv)  # forward
    grid_unaug = affine_to_grid(affine_v2.to(device), H, W)
    feat_tk_orig = F.grid_sample(feat_v2, grid_unaug, mode='bilinear', padding_mode='zeros', align_corners=True)

    # step2: 光流 warp t+k 原 -> t 原
    grid_flow = flow_to_grid(flow.to(device))
    feat_t_orig = F.grid_sample(feat_tk_orig, grid_flow, mode='bilinear', padding_mode='zeros', align_corners=True)

    # step3: 施加 view1 增广（原 -> view1）
    grid_v1 = affine_to_grid(affine_v1_inv.to(device), H, W)
    feat_v1 = F.grid_sample(feat_t_orig, grid_v1, mode='bilinear', padding_mode='zeros', align_corners=True)
    return feat_v1


def trainer_geometric_consistency(args, model, snapshot_path, train_loader):
    """
    Training loop with pixel-wise contrastive + IIC, using two augmented views and flow/affine warping.
    """
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    batch_size = args.batch_size

    # shared dataset for val (same as train, no shuffle)
    db_train = train_loader.dataset
    print("The length of train set (valid pairs) is: {}".format(len(db_train)))

    val_loader = DataLoader(db_train, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)

    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10

    for epoch_num in iterator:
        model.train()
        epoch_loss = 0.0

        for i_batch, data in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}",
                                  total=len(train_loader), leave=False):
            # unpack two views and warping info
            x1 = data['image_t_v1'].cuda()
            x2 = data['image_tk_v2'].cuda()
            flow = data['flow'].cuda() if 'flow' in data else None
            affine_v2_inv = data.get('affine_tk_v2', None)
            affine_v1_inv = data.get('affine_t_v1', None)
            if affine_v2_inv is not None:
                affine_v2_inv = affine_v2_inv.cuda()
            if affine_v1_inv is not None:
                affine_v1_inv = affine_v1_inv.cuda()

            # forward
            _, proj1, prob1 = model(x1)
            _, proj2, prob2 = model(x2)

            # warp view2 to view1 coord: remove view2 aug -> flow -> apply view1 aug
            if flow is not None and affine_v2_inv is not None and affine_v1_inv is not None:
                proj2_aligned = warp_v2_to_v1(proj2, affine_v2_inv, flow, affine_v1_inv)
                prob2_aligned = warp_v2_to_v1(prob2, affine_v2_inv, flow, affine_v1_inv)
            else:
                # fallback: simple warp by affine only
                proj2_aligned = project_feature_map(proj2, affine=affine_v2_inv)
                prob2_aligned = project_feature_map(prob2, affine=affine_v2_inv)

            # contrastive (positive only, cosine)
            cos_sim = F.cosine_similarity(proj1, proj2_aligned, dim=1)  # [B,H,W]
            loss_contrast = (1 - cos_sim).mean()

            # IIC mutual information
            prob1_sm = prob1  # already softmaxed in model
            prob2_sm = prob2_aligned
            mi = _iic_mutual_information(prob1_sm, prob2_sm)
            loss_iic = -mi

            loss = args.lambda_contrast * loss_contrast + args.lambda_iic * loss_iic

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('loss/total', loss.item(), iter_num)
            writer.add_scalar('loss/contrast', loss_contrast.item(), iter_num)
            writer.add_scalar('loss/iic', loss_iic.item(), iter_num)

            epoch_loss += loss.item()

            if iter_num % 50 == 0:
                diff_map = torch.mean(torch.abs(proj1 - proj2_aligned), dim=1, keepdim=True)
                diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-6)
                writer.add_image('train/Image_t_v1', x1[0, 0:1, :, :], iter_num)
                writer.add_image('train/Diff_proj', diff_map[0, ...], iter_num)

        epoch_loss /= len(train_loader)
        logging.info('Train epoch: %d : loss : %f' % (epoch_num, epoch_loss))

        # Validation loop
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i_batch, data in tqdm(enumerate(val_loader), desc=f"Val: {epoch_num}",
                                          total=len(val_loader), leave=False):
                    x1 = data['image_t_v1'].cuda()
                    x2 = data['image_tk_v2'].cuda()
                    flow = data['flow'].cuda() if 'flow' in data else None
                    affine_v2_inv = data.get('affine_tk_v2', None)
                    affine_v1_inv = data.get('affine_t_v1', None)
                    if affine_v2_inv is not None:
                        affine_v2_inv = affine_v2_inv.cuda()
                    if affine_v1_inv is not None:
                        affine_v1_inv = affine_v1_inv.cuda()

                    _, proj1, prob1 = model(x1)
                    _, proj2, prob2 = model(x2)

                    if flow is not None and affine_v2_inv is not None and affine_v1_inv is not None:
                        proj2_aligned = warp_v2_to_v1(proj2, affine_v2_inv, flow, affine_v1_inv)
                        prob2_aligned = warp_v2_to_v1(prob2, affine_v2_inv, flow, affine_v1_inv)
                    else:
                        proj2_aligned = project_feature_map(proj2, affine=affine_v2_inv)
                        prob2_aligned = project_feature_map(prob2, affine=affine_v2_inv)

                    cos_sim = F.cosine_similarity(proj1, proj2_aligned, dim=1)
                    loss_contrast = (1 - cos_sim).mean()
                    mi = _iic_mutual_information(prob1, prob2_aligned)
                    loss_iic = -mi
                    loss = args.lambda_contrast * loss_contrast + args.lambda_iic * loss_iic
                    val_loss += loss.item()

                val_loss /= len(val_loader)
                logging.info('Val epoch: %d : loss : %f' % (epoch_num, val_loss))

                # save model
                if val_loss < best_loss:
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    best_loss = val_loss
                else:
                    save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

        model.train()

    writer.close()
    return "Training Finished!"
