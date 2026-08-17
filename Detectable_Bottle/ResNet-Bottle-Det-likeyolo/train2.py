import os
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import YOLODataset
from model import Net
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torch.nn as nn
from tool import compute_loss  # ← 只导入函数, 不触发 MLP

if __name__ == "__main__":
    model = Net()   # 从头训, 不加载旧权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = YOLODataset(
        image_dir="./dataset/images/train",
        label_dir="./dataset/labels/train",
        img_width=256, img_height=342,
        grid_height=11, grid_width=8,
        mosaic_prob=0.5,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,
                            num_workers=4, pin_memory=True,
                            persistent_workers=True)

    # 验证集（无 mosaic）, 每 5 个 epoch 看一眼真实泛化
    valid_set = YOLODataset(
        image_dir="./dataset/images/valid",
        label_dir="./dataset/labels/valid",
        img_width=256, img_height=342,
        grid_height=11, grid_width=8,
        mosaic_prob=0,
    )
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False,
                              num_workers=4, pin_memory=True)
    model.train()

    epoch = 200

    # ---------- 优化器: 迁移学习分 lr ----------
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if 'backbone.' in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    opt = optim.SGD([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': head_params,    'lr': 1e-3},
    ], momentum=0.9, nesterov=True)

    # ---------- 余弦退火 ----------
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=epoch, eta_min=1e-5)

    losses = []
    best_loss = float("inf")

    for e in range(epoch):
        loss1 = 0
        n_batch = 0
        lr_now = opt.param_groups[0]['lr']
        for img, target in dataloader:
            img, target = img.to(device), target.to(device)
            opt.zero_grad()

            output = model(img)
            t = target   # 确认 target 布局 (B,8,11,5) 与 output 一 致

            # ===== 用 compute_loss =====
            loss = compute_loss(output, t, grid_w=8, grid_h=11)

            if not torch.isfinite(loss):
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            loss1 += loss.item()
            n_batch += 1

        scheduler.step()

        avg_loss = loss1 / max(n_batch, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_long.pth")
            print(f"✅ best: {best_loss:.4f}")
        losses.append(avg_loss)
        print(f"epoch {e+1}/{epoch} | lr={lr_now:.1e} | loss={avg_loss:.5f}")

        # 每 5 个 epoch 在验证集上测一次 loss（无 mosaic）
        if (e + 1) % 5 == 0:
            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for img, target in valid_loader:
                    img, target = img.to(device), target.to(device)
                    v_loss += compute_loss(model(img), target, grid_w=8, grid_h=11).item()
            model.train()
            print(f"    📊 valid loss: {v_loss / len(valid_loader):.5f}")

    torch.save(model.state_dict(), "model_long.pth")
    plt.plot(losses)
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.show()
