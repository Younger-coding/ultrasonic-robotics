import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader # DataLoader 仍用于创建 val_loader
from tqdm import tqdm
import torch.nn.functional as F

# 导入几何辅助函数 (假设在 geometric_utils 模块)
from geometric_utils import calculate_relative_transform, project_feature_map 
# ----------------------------------------------------------------------


def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)

# ⚠️ 修正函数签名：接受 train_loader 作为第四个参数
def trainer_geometric_consistency(args, model, snapshot_path, train_loader): 
    """
    基于几何一致性约束的自监督训练器。
    """
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    base_lr = args.base_lr
    # 注意: batch_size 已经是 args.batch_size * args.n_gpu
    batch_size = args.batch_size 
    
    # ----------------------------------------------------------------
    # 修正数据加载逻辑：使用传入的 train_loader
    # ----------------------------------------------------------------
    
    # 1. 从传入的 train_loader 中提取数据集对象
    db_train = train_loader.dataset 
    
    print("The length of train set (valid pairs) is: {}".format(len(db_train)))

    # 2. 使用相同的 Dataset 对象创建 Val Loader (仅改变 shuffle)
    # ⚠️ 注意：这里使用 args.batch_size，与 train_loader 保持一致
    val_loader = DataLoader(db_train, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    
    # ----------------------------------------------------------------

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    
    # 2. 替换损失函数：使用 L1 Loss 或 MSE Loss (用于特征图的一致性衡量)
    consistency_loss_fn = nn.L1Loss() 
    
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    
    # 3. 使用传入 train_loader 的长度计算 max_iterations
    max_iterations = args.max_epochs * len(train_loader)
    
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10
    
    for epoch_num in iterator:
        model.train()
        batch_consistency_loss = 0
        
        # 4. 使用传入的 train_loader 进行迭代
        for i_batch, data in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}", 
                                 total=len(train_loader), leave=False):
            
            # 3. 解包定制数据集返回的四项数据
            image_t = data['image_t'].cuda()      # I_t
            pose_t = data['pose_t'].cuda()        # T_t (7D 位姿)
            image_tk = data['image_tk'].cuda()    # I_{t+k}
            pose_tk = data['pose_tk'].cuda()      # T_{t+k} (7D 位姿)
            
            # 4. Swin-Unet 前向传播 (输出语义特征图 S)
            S_t = model(image_t)      
            S_tk = model(image_tk)
            
            # 5. 几何计算与投影 (核心自监督逻辑)
            # 5.1. 计算相对变换矩阵
            M_rel = calculate_relative_transform(pose_t, pose_tk) 
            
            # 5.2. 投影 S_t 到 I_{t+k} 视角，得到 S'_tk
            S_prime_tk = project_feature_map(S_t, M_rel)
            
            # 5.3. 计算几何一致性损失
            consistency_loss = consistency_loss_fn(S_tk, S_prime_tk)
            
            loss = consistency_loss 
            
            # 6. 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 学习率调整
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/consistency_loss', consistency_loss.item(), iter_num)

            batch_consistency_loss += consistency_loss.item()
            
            if iter_num % 20 == 0:
                # 可视化: 现在可视化 S_tk 和 S'_tk 的差异更有意义
                diff_map = torch.mean(torch.abs(S_tk - S_prime_tk), dim=1, keepdim=True)
                diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min())
                writer.add_image('train/Image_t', image_t[1, 0:1, :, :], iter_num)
                writer.add_image('train/Consistency_Diff', diff_map[1, ...] * 255, iter_num)
        
        # 7. 记录训练周期损失
        batch_consistency_loss /= len(train_loader)
        batch_loss = batch_consistency_loss
        logging.info('Train epoch: %d : loss : %f' % (epoch_num, batch_loss))
        
        # 8. 评估循环 (Val Loop) - 保持自监督逻辑
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i_batch, data in tqdm(enumerate(val_loader), desc=f"Val: {epoch_num}",
                                         total=len(val_loader), leave=False):
                    image_t = data['image_t'].cuda()
                    pose_t = data['pose_t'].cuda()
                    image_tk = data['image_tk'].cuda()
                    pose_tk = data['pose_tk'].cuda()

                    S_t = model(image_t)      
                    S_tk = model(image_tk)
                    M_rel = calculate_relative_transform(pose_t, pose_tk) 
                    S_prime_tk = project_feature_map(S_t, M_rel)
                    
                    consistency_loss = consistency_loss_fn(S_tk, S_prime_tk)
                    val_loss += consistency_loss.item()

                val_loss /= len(val_loader)
                logging.info('Val epoch: %d : loss : %f' % (epoch_num, val_loss))
                
                # 保存模型逻辑
                if val_loss < best_loss:
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    best_loss = val_loss
                else:
                    save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"