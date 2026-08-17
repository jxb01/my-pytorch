import torch.nn as nn
import torch
import pickle
import numpy as np

def calculate_landing(v0, angle_deg, wind_speed=0.0,g=9.8, k=0.1, dt=0.005):
    """
    使用数值积分计算含空气阻力的抛体落点
    返回: np.array([distance, t_flight, max_height], dtype=np.float32)
    """
    theta = np.radians(angle_deg)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    x, y = 0.0, 0.0
    t = 0.0
    max_height = 0.0
    while y >= 0:
        # 记录最高点
        if y > max_height:
            max_height = y
        # 计算相对风速
        rel_vx = vx - wind_speed
        rel_vy = vy
        rel_v = np.sqrt(rel_vx**2 + rel_vy**2)
        # 空气阻力加速度 (与相对速度平方成正比，方向相反)
        if rel_v > 0:
            drag_ax = -k * rel_v * rel_vx
            drag_ay = -k * rel_v * rel_vy
        else:
            drag_ax, drag_ay = 0.0, 0.0
        # 更新速度
        vx += drag_ax * dt
        vy += (drag_ay - g) * dt
        # 更新位置
        x += vx * dt
        y += vy * dt
        t += dt
    return np.array([x, t, max_height], dtype=np.float32)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(3,64)
        self.fc2=nn.Linear(64,32)
        self.fc3=nn.Linear(32,12)
        self.fc4=nn.Linear(12,3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.15)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Net().to(device)
    model.load_state_dict(torch.load('model.pth', map_location=torch.device(device)))
    with open('scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    x_scaler = scalers['x_scaler']
    y_scaler = scalers['y_scaler']
    model.eval()
    cri=nn.MSELoss()
    with torch.no_grad():
        while True:
            v=float(input())
            a=float(input())
            w=float(input())
            numpy1=x_scaler.transform([[v,a,w]])
            result=model.forward(torch.tensor(numpy1,dtype=torch.float32).to(device))
            print("model is ",y_scaler.inverse_transform(result.cpu().numpy()))
            print("real is",calculate_landing(v,a,w))
            loss=cri(torch.tensor(y_scaler.inverse_transform(result.cpu().numpy())),torch.tensor(calculate_landing(v,a,w)).unsqueeze(0))
            print("loss is",loss.numpy())