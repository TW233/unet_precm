import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


# 引入我们刚才修改后的模型类
# 假设你把上面的代码保存为 networks/UNet/unet_PreCM_fixed.py
from networks.UNet.unet_PreCM_fixed import unet_gconv as PreCM_unet_fixed
from networks.UNet.unet import Unet as Standard_unet
from networks.UNet.unet_FConv import unet_gconv as FConv_unet

# --- 配置 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4 
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
EPOCHS = 250 # DRIVE数据集论文设定为250
IMG_SIZE = (448, 448) # 必须固定为448，因为PreCM内部硬编码了尺寸
NUM_REPEATS = 5 # 论文设定重复5次取平均
OUTPUT_DIR = './output_drive_reproduce_01_11'

# --- 1. 数据集 (保持你的代码不变) ---
class DriveDataset(Dataset):
    def __init__(self, root_dir, split='training', is_train=False, file_list=None):
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.masks_dir = os.path.join(root_dir, split, '1st_manual')
        self.roi_dir = os.path.join(root_dir, split, 'mask')
        self.image_files = file_list if file_list else sorted([f for f in os.listdir(self.images_dir) if f.endswith('.tif')])
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
        roi = Image.open(roi_path).convert('L') if os.path.exists(roi_path) else Image.new('L', img.size, 255)

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
        return img, mask, roi, name # 返回文件名以便可视化

# --- 2. Loss & Helper ---
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_visual_result(run_dir, img_name, img_t, pred_0, pred_rot, angle):
    # 保存论文风格的可视化结果: Original, Pred(0), Pred(Rot), Difference
    viz_dir = os.path.join(run_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 转回CPU numpy
    img_np = img_t.cpu().squeeze().permute(1,2,0).numpy() * 0.5 + 0.5 # denorm
    p0_np = pred_0.cpu().squeeze().numpy()
    prot_np = pred_rot.cpu().squeeze().numpy()
    
    diff = np.abs(p0_np - prot_np)
    
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1); plt.imshow(img_np); plt.title("Input"); plt.axis('off')
    plt.subplot(1, 4, 2); plt.imshow(p0_np, cmap='gray'); plt.title("Pred 0°"); plt.axis('off')
    plt.subplot(1, 4, 3); plt.imshow(prot_np, cmap='gray'); plt.title(f"Pred {angle}°"); plt.axis('off')
    plt.subplot(1, 4, 4); plt.imshow(diff, cmap='jet'); plt.title("Diff"); plt.axis('off') # jet colormap for heat
    
    plt.savefig(os.path.join(viz_dir, f"{img_name}_{angle}.png"))
    plt.close()

# --- 3. 训练与评估 ---
def evaluate(model, loader, run_dir, save_viz=False):
    model.eval()
    scenarios = ['0', '90', '180', '270', 'random']
    results = {s: {'IOU': [], 'MIOU': [], 'DICE': [], 'RD': []} for s in scenarios}
    
    with torch.no_grad():
        for i, (img, mask, roi, fname) in enumerate(loader):
            img, mask, roi = img.to(DEVICE), mask.to(DEVICE), roi.to(DEVICE)
            fname = fname[0]
            
            # Base prediction (0 degree)
            out_base = torch.sigmoid(model(img))
            pred_base = (out_base > 0.5).float() * roi
            
            for angle in scenarios:
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
                
                out = model(img_in)
                pred_prob = torch.sigmoid(out)
                pred = (pred_prob > 0.5).float()
                
                # Metrics
                p_roi, g_roi = pred * roi_tgt, mask_tgt * roi_tgt
                inter = (p_roi * g_roi).sum()
                union = p_roi.sum() + g_roi.sum() - inter
                iou = (inter + 1e-6) / (union + 1e-6)
                dice = (2 * inter + 1e-6) / (p_roi.sum() + g_roi.sum() + 1e-6)
                
                roi_sum = roi_tgt.sum()
                p_bg = roi_tgt - p_roi
                g_bg = roi_tgt - g_roi
                inter_bg = (p_bg * g_bg).sum()
                union_bg = p_bg.sum() + g_bg.sum() - inter_bg
                miou = (iou + (inter_bg + 1e-6)/(union_bg + 1e-6)) / 2.0
                
                results[angle]['IOU'].append(iou.item())
                results[angle]['MIOU'].append(miou.item())
                results[angle]['DICE'].append(dice.item())
                
                # RD Calculation
                if angle == '0':
                    results[angle]['RD'].append(0.0)
                else:
                    pred_back = inv_func(pred) * roi
                    diff = torch.abs(pred_back - pred_base)
                    rd = diff.mean().item()
                    results[angle]['RD'].append(rd)
                    
                    if save_viz and i < 3: # 只保存前3张图
                        save_visual_result(run_dir, fname, img, pred_base, pred_back, angle)
                    
    final = {}
    for s in scenarios:
        final[s] = {k: np.mean(v)*100 if v else 0 for k,v in results[s].items()}
    return final

def run_experiment(model_type, run_id, dataset_root, all_files):
    run_dir = os.path.join(OUTPUT_DIR, model_type, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n--- {model_type} | Run {run_id} ---")
    
    # 论文设定：验证集和测试集在一起，每次随机抽70%做测试
    num_test = int(len(all_files) * 0.7)
    test_files = random.sample(all_files, num_test)
    
    train_ds = DriveDataset(dataset_root, 'training', True)
    test_ds = DriveDataset(dataset_root, 'test', False, test_files)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=1)
    
    if model_type == 'PreCM':
        model = PreCM_unet_fixed(in_channels=3, classes=1).to(DEVICE)
    elif model_type == 'FConv':
        model = FConv_unet(in_channels=3, classes=1).to(DEVICE)
    else:
        # 如果你想复现论文的标准Unet，这里也许该把start_channel也改小
        # 但为了对比，我们先保持32或16
        model = Standard_unet(in_channels=3, classes=1).to(DEVICE)
        
    params_m = count_parameters(model)/1e6
    print(f"Params: {params_m:.2f} M (Target: ~3.15M for PreCM)")
    if model_type == 'PreCM' and params_m > 4.0:
        print("Warning: Model size is too big! Did you set start_ch=16?")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    criterion = SafeDiceBCELoss()
    
    loss_history = []
    model.train()
    
    # 为了快速演示，这里可以把EPOCHS调小，正式跑请用250
    for epoch in range(EPOCHS):
        ep_loss = 0
        for img, mask, _, _ in train_dl:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(img)
            
            # NaN检查
            if torch.isnan(out).any():
                print(f"Error: NaN detected at epoch {epoch}. Stopping.")
                return None
                
            loss = criterion(out, mask)
            loss.backward()

            # 【移除】梯度裁剪
            # 原论文没说用裁剪，PreCM 4倍求和虽然方差大，但如果有正确的 BN 和 Init，
            # 裁剪可能会掩盖梯度流动的真实情况。
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            ep_loss += loss.item()
            
        scheduler.step()
        avg_loss = ep_loss / len(train_dl)
        loss_history.append(avg_loss)
        
        if (epoch+1) % 50 == 0:
            print(f"Ep {epoch+1} | Loss: {avg_loss:.4f}")
            
    metrics = evaluate(model, test_dl, run_dir, save_viz=True)
    
    # Save artifacts
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pth"))
    plt.figure(); plt.plot(loss_history); plt.savefig(os.path.join(run_dir, "loss.png")); plt.close()
    
    del model
    return metrics

def main():
    # 设置你的数据路径
    root = './data/DRIVE' 
    # 确保此处指向的是测试集的图片目录
    test_dir = os.path.join(root, 'test', 'images')
    if not os.path.exists(test_dir):
        print(f"Path not found: {test_dir}")
        return
        
    files = sorted([f for f in os.listdir(test_dir) if f.endswith('.tif')])
    
    target_model = 'Standard'  # 可选 'PreCM', 'FConv', 'Standard'
    all_metrics = []
    
    for i in range(1, NUM_REPEATS + 1):
        m = run_experiment(target_model, i, root, files)
        if m: all_metrics.append(m)
        
    print(f"\n=== Final Results: {target_model} (Avg of {len(all_metrics)} runs) ===")
    print(f"{'Angle':<8} | {'IOU':<8} | {'MIOU':<8} | {'DICE':<8} | {'RD':<8}")
    for ang in ['0', '90', '180', '270', 'random']:
        if all_metrics:
            avgs = {k: np.mean([r[ang][k] for r in all_metrics]) for k in ['IOU', 'MIOU', 'DICE', 'RD']}
            print(f"{ang:<8} | {avgs['IOU']:.2f}     | {avgs['MIOU']:.2f}     | {avgs['DICE']:.2f}     | {avgs['RD']:.2f}")
        else:
            print("No successful runs.")

if __name__ == '__main__':

    main()
