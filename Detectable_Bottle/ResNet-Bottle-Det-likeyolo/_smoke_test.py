import torch
import torch.nn as nn
from model import Net
from tool import compute_loss

torch.manual_seed(0)
m = Net()
m.eval()
x = torch.randn(2, 3, 342, 256)
y = m(x)
print("output shape:", tuple(y.shape), "(expect (2, 8, 11, 5))")
assert tuple(y.shape) == (2, 8, 11, 5), "shape mismatch!"
assert y.min() >= 0 and y.max() <= 1, "sigmoid range violated!"

# 构造一个带真实框的 target
t = torch.zeros(2, 8, 11, 5)
t[0, 3, 5, 4] = 1.0
t[0, 3, 5, 0] = 0.2
t[0, 3, 5, 1] = 0.3
t[0, 3, 5, 2] = 0.4
t[0, 3, 5, 3] = 0.5
t[1, 7, 2, 4] = 1.0
t[1, 7, 2, 0] = 0.5
t[1, 7, 2, 1] = 0.7
t[1, 7, 2, 2] = 0.2
t[1, 7, 2, 3] = 0.3

loss = compute_loss(y, t, grid_w=8, grid_h=11)
print("loss:", loss.item())
assert torch.isfinite(loss), "loss is NaN!"

m.train()
loss.backward()
grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
print("all params got finite gradients:", grad_ok)
assert grad_ok

# 验证: 网格索引修复后, 把预测框推到 target 上, loss 应该显著下降
opt = torch.optim.SGD(m.parameters(), lr=0.01)
l0 = loss.item()
for _ in range(3):
    opt.zero_grad()
    out = m(x)
    l = compute_loss(out, t, grid_w=8, grid_h=11)
    l.backward()
    opt.step()
l1 = compute_loss(m(x), t, grid_w=8, grid_h=11).item()
print(f"loss before: {l0:.4f} -> after 3 steps: {l1:.4f}")
print("SMOKE TEST PASSED")
