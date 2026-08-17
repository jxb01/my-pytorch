import torch
import torch.nn as nn


def box_cxcywh_to_xyxy(boxes):
    """
    将 [cx, cy, w, h] 转换为 [x1, y1, x2, y2]
    boxes: [B, 4] 或 [B, N, 4]
    """
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(boxes1, boxes2):
    """
    计算两组框的 IoU
    boxes1, boxes2: [B, 4] 或 [B, N, 4]
    返回: [B] 或 [B, N]
    """
    # 转换为 [x1, y1, x2, y2]
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)

    # 计算交集
    inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    # 交集面积（确保非负）
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # 各自面积
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 并集面积
    union_area = area1 + area2 - inter_area + 1e-6  # 加小值防止除零

    # IoU
    iou = inter_area / union_area
    return iou


def giou_loss(pred_boxes, target_boxes):
    """
    GIoU Loss
    pred_boxes, target_boxes: [B, 4] 格式为 [cx, cy, w, h]
    """
    # 转换为 [x1, y1, x2, y2]
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)

    # 计算交集
    inter_x1 = torch.max(pred_xyxy[..., 0], target_xyxy[..., 0])
    inter_y1 = torch.max(pred_xyxy[..., 1], target_xyxy[..., 1])
    inter_x2 = torch.min(pred_xyxy[..., 2], target_xyxy[..., 2])
    inter_y2 = torch.min(pred_xyxy[..., 3], target_xyxy[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # 各自面积
    pred_area = (pred_xyxy[..., 2] - pred_xyxy[..., 0]) * (pred_xyxy[..., 3] - pred_xyxy[..., 1])
    target_area = (target_xyxy[..., 2] - target_xyxy[..., 0]) * (target_xyxy[..., 3] - target_xyxy[..., 1])

    # 并集面积
    union_area = pred_area + target_area - inter_area + 1e-6

    # IoU（直接算，不用 box_iou 函数）
    iou = inter_area / union_area

    # 最小外接矩形
    outer_x1 = torch.min(pred_xyxy[..., 0], target_xyxy[..., 0])
    outer_y1 = torch.min(pred_xyxy[..., 1], target_xyxy[..., 1])
    outer_x2 = torch.max(pred_xyxy[..., 2], target_xyxy[..., 2])
    outer_y2 = torch.max(pred_xyxy[..., 3], target_xyxy[..., 3])

    outer_w = (outer_x2 - outer_x1).clamp(min=0)
    outer_h = (outer_y2 - outer_y1).clamp(min=0)
    outer_area = outer_w * outer_h + 1e-6

    # GIoU
    giou = iou - (outer_area - union_area) / outer_area

    return 1 - giou.mean()  # GIoU Loss


def ciou_loss(pred_boxes, target_boxes):
    """
    CIoU Loss（考虑中心点距离和宽高比）
    最精确，但计算稍复杂
    """
    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes)

    # 分离坐标
    pred_cx, pred_cy, pred_w, pred_h = pred_boxes[..., 0], pred_boxes[..., 1], pred_boxes[..., 2], pred_boxes[..., 3]
    target_cx, target_cy, target_w, target_h = target_boxes[..., 0], target_boxes[..., 1], target_boxes[..., 2], \
    target_boxes[..., 3]

    # IoU
    iou = box_iou(pred_boxes, target_boxes)

    # 中心点距离（欧氏距离）
    center_dist = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

    # 外接矩形对角线距离
    outer_x1 = torch.min(pred_xyxy[..., 0], target_xyxy[..., 0])
    outer_y1 = torch.min(pred_xyxy[..., 1], target_xyxy[..., 1])
    outer_x2 = torch.max(pred_xyxy[..., 2], target_xyxy[..., 2])
    outer_y2 = torch.max(pred_xyxy[..., 3], target_xyxy[..., 3])
    outer_diag = (outer_x2 - outer_x1) ** 2 + (outer_y2 - outer_y1) ** 2 + 1e-6

    # 距离惩罚
    v = (4 / (torch.pi ** 2)) * (torch.atan(target_w / (target_h + 1e-6)) - torch.atan(pred_w / (pred_h + 1e-6))) ** 2
    alpha = v / (1 - iou + v + 1e-6)

    ciou = iou - center_dist / outer_diag - alpha * v
    return 1 - ciou.mean()


class IoULoss(nn.Module):
    """
    IoU Loss 包装类，方便使用
    """

    def __init__(self, loss_type='giou'):
        """
        loss_type: 'iou', 'giou', 'ciou'
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred, target):
        if self.loss_type == 'iou':
            return 1 - box_iou(pred, target).mean()
        elif self.loss_type == 'giou':
            return giou_loss(pred, target)
        elif self.loss_type == 'ciou':
            return ciou_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


def l1_loss(pred, target):
    """L1 Loss（作为辅助损失）"""
    return torch.abs(pred - target).mean()


def combined_loss(pred, target, alpha=0.5):
    """
    组合损失：GIoU + L1
    alpha: GIoU 损失的权重
    """
    giou = giou_loss(pred, target)
    l1 = l1_loss(pred, target)
    return alpha * giou + (1 - alpha) * l1