import cv2
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from model import Net
import os
import cv2 as cv
def compute_iou(box1, box2):
    """
    计算两个边界框的 IoU (交并比)
    box: [x1, y1, x2, y2]
    """
    # 计算交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算各自面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area
class Bottledataset(torch.utils.data.Dataset):
    def __init__(self, img_root, labels_root,transform):
        super(Bottledataset, self).__init__()
        self.img_root = img_root
        self.labels_root = labels_root
        self.transform = transform
        self.img_filenames = os.listdir(self.img_root)
        self.labels_filenames = os.listdir(self.labels_root)
    def __len__(self):
        return len(self.labels_filenames)
    def __getitem__(self, index):
        with open(
                self.labels_root+"\\"+ self.labels_filenames[index],
                "r") as f:
            lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        lines = lines[0].split()
        a,_=os.path.splitext(self.labels_filenames[index])
        cx = float(lines[1])
        cy = float(lines[2])
        width = float(lines[3])
        height = float(lines[4])
        labels = torch.tensor([
            cx - 0.5 * width,  # x1
            cy - 0.5 * height,  # y1
            cx + 0.5 * width,  # x2
            cy + 0.5 * height  # y2
        ], dtype=torch.float32)
        img = cv.imread(self.img_root+"\\" +a+".jpg")
        img=self.transform(img)
        return img ,labels
    def getrealimg(self,index):
        a,_=os.path.splitext(self.labels_filenames[index])
        img = cv.imread(self.img_root+"\\" +a+".jpg")
        return img

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform1 = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor()])
    datasets=Bottledataset(transform=transform1,img_root="data\\test\\images",labels_root="data\\test\\labels")
    print("device is",device)
    dataloader = torch.utils.data.DataLoader(dataset=datasets,batch_size=64,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
    print(len(dataloader))
    model = Net().to(device)
    if os.path.isfile("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
    model.eval()
    # with torch.no_grad():
    #     total_iou = 0
    #     num_samples = 0
    #     for img, labels in dataloader:
    #         img = img.to(device)
    #         labels = labels.to(device)  # [batch, 4]
    #         outputs = model(img)  # [batch, 4]
    #
    #         # 逐个样本计算 IoU
    #         for i in range(len(outputs)):
    #             pred_box = outputs[i].cpu().numpy()  # [x1, y1, x2, y2]
    #             true_box = labels[i].cpu().numpy()  # [x1, y1, x2, y2]
    #             total_iou += compute_iou(pred_box, true_box)
    #             num_samples += 1
    #     avg_iou = total_iou / num_samples if num_samples > 0 else 0
    #     print(f"IoU is {avg_iou:.4f}")
    cv2.namedWindow("img", 0)
    cap=cv2.VideoCapture(0)
    # model.eval()
    # img=cv2.imread("data\\test\\images\\2020-02-04-05_31_56-179924_jpg.rf.1a8106769119f59bbbfea344ca994a9f.jpg")
    # img=cv2.resize(img,(640,640))
    # img1=transform1(img)
    # img1=img1.to(device)
    # img1=img1.unsqueeze(0)
    with torch.no_grad():
        # realimg =
        # h, w, e = realimg.shape
        # img = transform1(realimg)
        # result = model(torch.tensor(img).to(device).unsqueeze(0))
        # print(result)
        # cv2.rectangle(realimg,(int(result[0][0]*w),int(result[0][1]*h)),(int(result[0][2]*w),int(result[0][3]*h)),(0,0,255),2)
        # cv2.resizeWindow("img",600,600)
        # cv2.imshow("img", realimg)
        # cv2.waitKey(0)
        while True:
            ret,frame=cap.read()
            realimg=frame
            realimg=cv2.resize(realimg,(224,224))
            h, w, e = realimg.shape
            img = transform1(realimg)
            result = model(torch.tensor(img).to(device).unsqueeze(0))
            print(result)
            cv2.rectangle(realimg,(int(result[0][0]*w),int(result[0][1]*h)),(int(result[0][2]*w),int(result[0][3]*h)),(0,0,255),2)
            cv2.resizeWindow("img",600,600)
            cv2.imshow("img", realimg)
            cv2.waitKey(32)

