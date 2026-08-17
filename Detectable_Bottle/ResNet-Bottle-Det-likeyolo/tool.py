import torch
import torch.nn as nn


def compute_loss(output, target, grid_w=8, grid_h=11):
    """
    仿 YOLO 损失（单类, 每格 1 框）。
    output/target 布局: (B, grid_w, grid_h, 5) = [sx, sy, w, h, conf]
      - sx, sy: 格内偏移 (0~1), 模型已过 sigmoid
      - w, h  : 归一化宽高 (0~1), 模型已过 sigmoid
      - conf  : 置信度 (0~1), 模型已过 sigmoid
    损失 = 5 * CIoU(有物格子) + BCE(conf vs IoU, 有物) + 0.5 * BCE(conf vs 0, 无物)

    注意: 张量布局是 (grid_w, grid_h), 展平后第 k 个格子的
    列 = k // grid_h, 行 = k % grid_h（行优先展平）。
    """
    output = output.float()
    target = target.float()
    B = output.shape[0]

    out = output.view(B, -1, 5)   # (B, N, 5)
    tgt = target.view(B, -1, 5)

    off_x, off_y = out[..., 0], out[..., 1]
    w, h = out[..., 2], out[..., 3]
    conf = out[..., 4]

    t_off_x, t_off_y = tgt[..., 0], tgt[..., 1]
    t_w, t_h = tgt[..., 2], tgt[..., 3]
    t_conf = tgt[..., 4]

    # 展平顺序: k = col * grid_h + row → col = k // grid_h, row = k % grid_h
    idx = torch.arange(out.shape[1], device=output.device)
    col_idx = (idx // grid_h).float()   # 列（宽方向）
    row_idx = (idx % grid_h).float()    # 行（高方向）

    # 中心（全图归一化 0~1）
    cx_pred = (col_idx + off_x) / grid_w
    cy_pred = (row_idx + off_y) / grid_h
    gcx = (col_idx + t_off_x) / grid_w
    gcy = (row_idx + t_off_y) / grid_h

    # CIoU
    b1 = torch.stack([cx_pred - w / 2, cy_pred - h / 2, cx_pred + w / 2, cy_pred + h / 2], dim=-1)
    b2 = torch.stack([gcx - t_w / 2, gcy - t_h / 2, gcx + t_w / 2, gcy + t_h / 2], dim=-1)

    inter_x1 = torch.max(b1[..., 0], b2[..., 0]); inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2]); inter_y2 = torch.min(b1[..., 3], b2[..., 3])
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    area1 = torch.clamp(b1[..., 2] - b1[..., 0], min=1e-6) * torch.clamp(b1[..., 3] - b1[..., 1], min=1e-6)
    area2 = torch.clamp(b2[..., 2] - b2[..., 0], min=1e-6) * torch.clamp(b2[..., 3] - b2[..., 1], min=1e-6)
    union = area1 + area2 - inter_area + 1e-7
    iou = (inter_area / union).clamp(0, 1)

    center_d2 = (cx_pred - gcx) ** 2 + (cy_pred - gcy) ** 2
    enclose_x1 = torch.min(b1[..., 0], b2[..., 0]); enclose_y1 = torch.min(b1[..., 1], b2[..., 1])
    enclose_x2 = torch.max(b1[..., 2], b2[..., 2]); enclose_y2 = torch.max(b1[..., 3], b2[..., 3])
    enc_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7
    ciou = iou - center_d2 / enc_diag

    obj = (t_conf > 0.5).float()
    noobj = (t_conf <= 0.5).float()

    coord_loss = 5.0 * (obj * (1 - ciou)).sum()
    bce = nn.BCELoss(reduction='none')                # conf 已过 sigmoid, 用普通 BCE
    iou_obj = (iou * obj).detach()                    # 有物格子的 conf 目标 = 当前 IoU（类似 YOLOv3）
    conf_obj = (obj * bce(conf, torch.clamp(iou_obj, 0, 1))).sum()
    conf_noobj = (0.5 * noobj * bce(conf, torch.zeros_like(conf))).sum()

    return (coord_loss + conf_obj + conf_noobj) / B
