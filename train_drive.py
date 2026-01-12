import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # 用于保存日志
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# --- 引入模型 ---
from networks.UNet.unet_PreCM_fixed import unet_gconv as PreCM_unet_fixed
from networks.UNet.unet import Unet as Standard_unet

# --- 配置 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4 
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
EPOCHS = 250
IMG_SIZE = (448, 448)
NUM_REPEATS = 5
OUTPUT_DIR = './output_drive_full_metrics'

# --- 1. 数据集 (保持不变) ---
class DriveDataset(Dataset):
    def __init__(self, root_dir, split='training', is_train=False, file_list=None):
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.masks_dir = os.path.join(root_dir, split, '1st_manual')
        self.roi_dir = os.path.join(root_dir, split, 'mask')
        
        if file_list:
            self.image_files = file_list
        else:
            self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.tif')])
            
        self.is_train = is_train
        self.split = split
        self.normalize = transforms.Normalize([0.5]*3, [0.5]*3)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    def __len__(self): return len(self.image_files)

    def enhance(self, img_pil):
        img_np = np.array(img_pil)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        return Image.fromarray(cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2RGB))

    def __getitem__(self, idx):
        name = self.image_files[idx]
        prefix = name.split('_')[0]
        roi_name = f"{prefix}_{self.split}_mask.gif"
        
        img = Image.open(os.path.join(self.images_dir, name)).convert('RGB')
        mask = Image.open(os.path.join(self.masks_dir, f"{prefix}_manual1.gif")).convert('L')
        roi_path = os.path.join(self.roi_dir, roi_name)
        
        if os.path.exists(roi_path):
            roi = Image.open(roi_path).convert('L')
        else:
            roi = Image.new('L', img.size, 255)

        img = img.resize(IMG_SIZE, Image.BICUBIC)
        mask = mask.resize(IMG_SIZE, Image.NEAREST)
        roi = roi.resize(IMG_SIZE, Image.NEAREST)
        
        img = self.enhance(img)
        
        if self.is_train:
            if random.random() > 0.5:
                img, mask, roi = TF.hflip(img), TF.hflip(mask), TF.hflip(roi)
            if random.random() > 0.5:
                img, mask, roi = TF.vflip(img), TF.vflip(mask), TF.vflip(roi)

        img = self.normalize(transforms.ToTensor()(img))
        mask = (torch.from_numpy(np.array(mask)).float() / 255.0 > 0.5).float().unsqueeze(0)
        roi = (torch.from_numpy(np.array(roi)).float() / 255.0 > 0.5).float().unsqueeze(0)
        return img, mask, roi, name

# --- 2. Loss & Metrics ---
class SafeDiceBCELoss(nn.Module):
    def forward(self, logits, targets, smooth=1.0):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
        bce = F.binary_cross_entropy(probs, targets, reduction='mean')
        inputs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        return 0.5 * bce + 0.5 * dice

def calculate_metrics_batch(pred_logits, targets, roi):
    """
    计算单个批次的 IOU, MIOU, DICE
    pred_logits: 模型输出 (未经过sigmoid)
    targets: 真实标签 (0或1)
    roi: 感兴趣区域 (0或1)
    """
    probs = torch.sigmoid(pred_logits)
    pred = (probs > 0.5).float()
    
    # 仅在 ROI 区域内计算
    pred = pred * roi
    targets = targets * roi
    
    # Flatten
    pred = pred.reshape(-1)
    targets = targets.reshape(-1)
    
    # Intersection & Union
    tp = (pred * targets).sum().item()
    fp = (pred * (1 - targets)).sum().item()
    fn = ((1 - pred) * targets).sum().item()
    tn = ((1 - pred) * (1 - targets)).sum().item() # 注意：这也受 ROI 限制，ROI 外全是0不应计入
    
    # --- 1. Foreground IOU (Class 1) ---
    iou_fg = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    
    # --- 2. Background IOU (Class 0) ---
    # 背景的 "True Positive" 是 TN
    iou_bg = (tn + 1e-6) / (tn + fp + fn + 1e-6)
    
    # --- 3. MIOU (Mean of Class 0 and Class 1) ---
    miou = (iou_fg + iou_bg) / 2.0
    
    # --- 4. DICE ---
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    
    return {'iou': iou_fg, 'miou': miou, 'dice': dice}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- 3. 可视化函数 (恢复!) ---
def save_visual_result(run_dir, img_name, img_t, pred_0, pred_rot, diff_map, angle):
    """
    保存论文风格的对比图: Input | Pred 0° | Pred Rot° | Diff
    """
    viz_dir = os.path.join(run_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 反归一化图片
    img_np = img_t.cpu().squeeze().permute(1,2,0).numpy() * 0.5 + 0.5
    img_np = np.clip(img_np, 0, 1)
    
    p0_np = pred_0.cpu().squeeze().numpy()
    prot_np = pred_rot.cpu().squeeze().numpy()
    diff_np = diff_map.cpu().squeeze().numpy()
    
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img_np)
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(p0_np, cmap='gray')
    plt.title("Pred 0°")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(prot_np, cmap='gray')
    plt.title(f"Pred {angle}°")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    # 使用 jet colormap 显示差异，论文中差异越小越蓝/黑，差异大为红
    plt.imshow(diff_np, cmap='jet', vmin=0, vmax=1) 
    plt.title(f"RD Diff Map ({angle}°)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{img_name}_{angle}.png"), dpi=100)
    plt.close()

# --- 4. 验证与测试流程 ---
def validate(model, loader):
    model.eval()
    metrics_sum = {'iou': 0, 'miou': 0, 'dice': 0}
    steps = 0
    
    with torch.no_grad():
        for img, mask, roi, _ in loader:
            img, mask, roi = img.to(DEVICE), mask.to(DEVICE), roi.to(DEVICE)
            out = model(img)
            m = calculate_metrics_batch(out, mask, roi)
            for k in metrics_sum:
                metrics_sum[k] += m[k]
            steps += 1
            
    return {k: v / steps for k, v in metrics_sum.items()}

def final_test(model, loader, run_dir):
    model.eval()
    scenarios = ['0', '90', '180', '270', 'random']
    # 存储每个角度下的所有图片指标
    results = {s: {'IOU': [], 'MIOU': [], 'DICE': [], 'RD': []} for s in scenarios}
    
    with torch.no_grad():
        for i, (img, mask, roi, name) in enumerate(loader):
            img, mask, roi = img.to(DEVICE), mask.to(DEVICE), roi.to(DEVICE)
            fname = name[0]
            
            # 0度基准预测 (Probabilities)
            out_base_logits = model(img)
            out_base_prob = torch.sigmoid(out_base_logits)
            
            for angle in scenarios:
                # 1. 准备输入
                if angle == '0':
                    img_in, mask_tgt, roi_tgt = img, mask, roi
                    inv_func = lambda x: x
                elif angle == 'random':
                    deg = random.uniform(0, 360)
                    img_in = TF.rotate(img, deg, interpolation=TF.InterpolationMode.BILINEAR)
                    mask_tgt = TF.rotate(mask, deg, interpolation=TF.InterpolationMode.NEAREST)
                    roi_tgt = TF.rotate(roi, deg, interpolation=TF.InterpolationMode.NEAREST)
                    inv_func = lambda x: TF.rotate(x, -deg, interpolation=TF.InterpolationMode.NEAREST)
                else: 
                    k = int(angle) // 90
                    img_in = torch.rot90(img, k=k, dims=(2, 3))
                    mask_tgt = torch.rot90(mask, k=k, dims=(2, 3))
                    roi_tgt = torch.rot90(roi, k=k, dims=(2, 3))
                    inv_func = lambda x: torch.rot90(x, k=-k, dims=(2, 3))
                
                # 2. 预测
                out = model(img_in)
                
                # 3. 计算分割指标 (IOU, MIOU, DICE)
                m = calculate_metrics_batch(out, mask_tgt, roi_tgt)
                results[angle]['IOU'].append(m['iou'])
                results[angle]['MIOU'].append(m['miou'])
                results[angle]['DICE'].append(m['dice'])
                
                # 4. 计算 RD 并 可视化
                pred_prob = torch.sigmoid(out)
                
                if angle == '0':
                    results[angle]['RD'].append(0.0)
                else:
                    # 逆旋转回 0 度
                    pred_back = inv_func(pred_prob)
                    
                    # 仅在 ROI 内计算差异
                    diff = torch.abs(pred_back - out_base_prob) * roi
                    rd = diff.mean().item()
                    results[angle]['RD'].append(rd)
                    
                    # --- 可视化：每种角度保存前 2 张图片 ---
                    if i < 2:
                        # 二值化用于显示
                        p0_bin = (out_base_prob > 0.5).float() * roi
                        prot_bin = (pred_back > 0.5).float() * roi
                        save_visual_result(run_dir, fname, img, p0_bin, prot_bin, diff, angle)
    
    # 汇总平均值
    final = {}
    for s in scenarios:
        final[s] = {k: np.mean(v) * 100 if k != 'RD' else np.mean(v) for k, v in results[s].items()} 
        # 注意：IOU/MIOU/DICE 乘100变百分比，RD 论文里好像是小数或者百分比，这里保持小数方便看差异
        # 如果论文 RD 是百分比 (e.g. 1.36%)，可以在这里 * 100
        # 看了下论文表格，RD 大概是 1.xx 到 3.xx，所以这里 * 100 比较符合阅读习惯
        final[s]['RD'] = np.mean(results[s]['RD']) * 100

    return final

# --- 5. 实验主程序 ---
def run_experiment(model_type, run_id, dataset_root):
    run_dir = os.path.join(OUTPUT_DIR, model_type, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n>>> Start {model_type} | Run {run_id}")
    
    # === 数据准备 (同之前逻辑) ===
    # 训练集：全部20张
    train_ds = DriveDataset(dataset_root, 'training', is_train=True)
    
    # 验证池：20张 -> 随机分 14 Test + 6 Val
    test_img_dir = os.path.join(dataset_root, 'test', 'images')
    all_test_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.tif')])
    random.shuffle(all_test_files)
    
    num_test = int(len(all_test_files) * 0.7)
    test_files = all_test_files[:num_test]
    val_files = all_test_files[num_test:]
    
    val_ds = DriveDataset(dataset_root, 'test', is_train=False, file_list=val_files)
    test_ds = DriveDataset(dataset_root, 'test', is_train=False, file_list=test_files)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    # 模型
    if model_type == 'PreCM':
        model = PreCM_unet_fixed(in_channels=3, classes=1).to(DEVICE)
    else:
        model = Standard_unet(in_channels=3, classes=1).to(DEVICE)
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    criterion = SafeDiceBCELoss()
    
    # 日志记录
    history = {'epoch': [], 'loss': [], 'val_iou': [], 'val_miou': [], 'val_dice': []}
    best_iou = 0.0
    best_epoch = 0
    
    model.train()
    
    # === 训练循环 (每轮都记录) ===
    for epoch in range(EPOCHS):
        ep_loss = 0
        for img, mask, _, _ in train_dl:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            out = model(img)
            
            if torch.isnan(out).any():
                print("NaN Detected! Stopping.")
                return None
                
            loss = criterion(out, mask)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            
        scheduler.step()
        avg_loss = ep_loss / len(train_dl)
        
        # 每轮都验证
        val_metrics = validate(model, val_dl)
        
        # 记录日志
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        history['val_iou'].append(val_metrics['iou'])
        history['val_miou'].append(val_metrics['miou'])
        history['val_dice'].append(val_metrics['dice'])
        
        # 保存最佳模型
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            
        # 打印进度 (每10轮)
        if (epoch + 1) % 10 == 0:
            print(f"Ep {epoch+1:03d} | Loss: {avg_loss:.4f} | Val IoU: {val_metrics['iou']:.4f} | Best: {best_iou:.4f} (Ep {best_epoch})")

    # === 保存训练曲线 ===
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(run_dir, "training_log.csv"), index=False)
    
    plt.figure(figsize=(10, 5))
    plt.plot(history['epoch'], history['loss'], label='Loss')
    plt.plot(history['epoch'], history['val_iou'], label='Val IoU')
    plt.title(f"Training History ({model_type})")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "history_plot.png"))
    plt.close()

    # === 最终测试 ===
    print(f"Testing Best Model (Epoch {best_epoch})...")
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth")))
    final_metrics = final_test(model, test_dl, run_dir)
    return final_metrics

def main():
    root = 'E:/PyCharm/Projects/data/DRIVE' 
    if not os.path.exists(root):
        print(f"Error: {root} not found!")
        return
        
    targets = ['Standard'] 
    
    for t_model in targets:
        all_metrics = []
        for i in range(1, NUM_REPEATS + 1):
            res = run_experiment(t_model, i, root)
            if res: all_metrics.append(res)
            
        print(f"\n====== Final Report: {t_model} (Avg of {len(all_metrics)} runs) ======")
        
        # 打印详细表格
        headers = ["Angle", "IOU(%)", "MIOU(%)", "DICE(%)", "RD(%)"]
        print(f"{headers[0]:<8} | {headers[1]:<8} | {headers[2]:<8} | {headers[3]:<8} | {headers[4]:<8}")
        print("-" * 55)
        
        angles = ['0', '90', '180', '270', 'random']
        for ang in angles:
            if all_metrics:
                # 计算五次实验的平均值
                avg_iou = np.mean([x[ang]['IOU'] for x in all_metrics])
                avg_miou = np.mean([x[ang]['MIOU'] for x in all_metrics])
                avg_dice = np.mean([x[ang]['DICE'] for x in all_metrics])
                avg_rd = np.mean([x[ang]['RD'] for x in all_metrics])
                
                print(f"{ang:<8} | {avg_iou:<8.2f} | {avg_miou:<8.2f} | {avg_dice:<8.2f} | {avg_rd:<8.2f}")
            else:
                print("No Data")

if __name__ == '__main__':
    main()