import torch
from model import swin_tiny_patch4_window7_224
from torchviz import make_dot

# 创建模型实例
model = swin_tiny_patch4_window7_224(num_classes=3)

# 生成一个随机输入
x = torch.randn(1, 3, 224, 224)

# 前向传播
out = model(x)

# 创建图形
dot = make_dot(out, params=dict(model.named_parameters()))

# 保存图形
dot.render('swin_tiny_model', format='png', cleanup=True, view=True)