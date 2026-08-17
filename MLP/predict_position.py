import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import os
def calculate_landing(v0, angle_deg, wind_speed=0, g=9.8, k=0.1, dt=0.005):
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
def geta():
    file_exist=os.path.exists("scalers.pkl")
    v0 = np.arange(5, 105, 5)
    angle_deg = np.arange(5, 105, 5)
    wind_speed = np.arange(0, 200, 10)

    # 总数据量：20 × 20 × 21 = 8400 个样本
    x_array=[]
    for v1 in range(len(v0)):
        for a1 in range(len(angle_deg)):
            for w1 in range(len(wind_speed)):
                x_array.append([v0[v1],angle_deg[a1],wind_speed[w1]])
    x=torch.tensor(x_array, dtype=torch.float32)
    #print(torch.tensor([v0,angle_deg,wind_speed], dtype=torch.float32).T)
    y=[]
    for v,a,w in x_array:
        y.append(calculate_landing(v,a,w))
    if file_exist:
        with open('scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
        x_scaler = scalers['x_scaler']
        y_scaler = scalers['y_scaler']
        x_train = x_scaler.transform(x.numpy())
        y_train = y_scaler.transform(np.stack(y))
    else:
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x.numpy())
        y_train = y_scaler.fit_transform(np.stack(y))

    data1=TensorDataset(torch.tensor(x_train),torch.tensor(y_train))
    return data1,y_scaler,x_scaler





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
    print("device is ",device)
    data,y_scaler,x_scaler=geta()
    data_tran=DataLoader(data,batch_size=128,shuffle=True,drop_last=True,num_workers=4,pin_memory=True,persistent_workers=True)
    epoch=50
    model=Net().to(device)
    model.load_state_dict(torch.load("model.pth"))
    criterion = nn.MSELoss()
    optimizer=optim.Adam(model.parameters(),lr=0.0001,weight_decay=0.0001)
    losses=[]
    weights = torch.tensor([5.0, 1.0, 1.0]).to(device)
    for epoch in range(epoch):
        d=0
        for  x,y in data_tran:
            x1=x.to(device)
            y1=y.to(device)
            output=model.forward(x1)
            loss = criterion(output,y1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            d=loss
        print(f"the {epoch+1} epoch loss is {d/len(data_tran)}")
    model.eval()
    torch.save(model.state_dict(), "model.pth")
    with open("scalers.pkl", "wb") as f:
        pickle.dump({"x_scaler":x_scaler,
                     "y_scaler":y_scaler},f)
    # with torch.no_grad():
    #     while True:
    #         v=float(input())
    #         a=float(input())
    #         w=float(input())
    #         numpy1=x_scaler.transform([[v,a,w]])
    #         result=model.forward(torch.tensor(numpy1,dtype=torch.float32).to(device))
    #         print("model is ",y_scaler.inverse_transform(result.cpu().numpy()))
    #         print("real is",calculate_landing(v,a,w))



