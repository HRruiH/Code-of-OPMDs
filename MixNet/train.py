import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mixnet import MixNet  # 假设 mixnet.py 文件中定义了 MixNet 模型
from tqdm import tqdm  # 导入 tqdm 库



# 训练过程
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    with tqdm(train_loader, desc=f'Train Epoch {epoch}', unit='batch') as t:
        for data, target in t:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=loss.item())

# 验证过程
def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation', unit='batch'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc = 100. * correct / len(val_loader.dataset)
    print(
        f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({val_acc:.0f}%)')
    return val_acc

for run in range(1,6):
  torch.manual_seed(run)
  # 设置设备
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # 数据预处理
  transform_train = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  
  transform_val = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  
  # 加载数据集
  train_dataset = datasets.ImageFolder(root='../data_3/train', transform=transform_train)
  train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
  
  val_dataset = datasets.ImageFolder(root='../data_3/val', transform=transform_val)
  val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
  
  # 定义模型
  model = MixNet(arch='s', num_classes=3)  # 修改为三分类
  model = model.to(device)
  
  # 加载预训练权重
  pretrained_weights_path = 'pretrained_weights/mixnet_s_top1v_75.2.pkl'  # 预训练权重路径
  if os.path.exists(pretrained_weights_path):
      print(f"Loading pretrained weights from {pretrained_weights_path}")
      model.load_state_dict(torch.load(pretrained_weights_path, map_location=device), strict=False)
  else:
      print("No pretrained weights found. Training from scratch.")
  
  # 定义损失函数和优化器
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
  
  # 初始化最佳验证准确率和早停计数器
  best_val_acc = 0.0
  early_stopping_patience = 10  # 设置早停的耐心值
  early_stopping_counter = 0
  
  # 确保保存模型的目录存在
  os.makedirs('weights', exist_ok=True)
  # 主训练循环
  for epoch in range(1, 100):
      train(model, device, train_loader, optimizer, criterion, epoch)
      val_acc = validate(model, device, val_loader, criterion)
  
      # 保存最佳模型
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          torch.save(model.state_dict(), f'weights/best_mixnet_s_trained{run}.pth')
          print(f"Saved better model with validation accuracy: {best_val_acc:.4f}%")
          early_stopping_counter = 0  # 重置早停计数器
      else:
          early_stopping_counter += 1
          print(f"No improvement in validation accuracy. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
  
      # 检查是否触发早停机制
      if early_stopping_counter >= early_stopping_patience:
          print(f"Early stopping triggered after {epoch} epochs.")
          break
  
  print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}%")