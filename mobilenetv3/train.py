import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mobilenetv3 import mobilenetv3
import json
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义数据集路径，需要根据实际情况修改
train_data_dir = '../data_3/train'
test_data_dir = '../data_3/val'

# 加载自定义数据集
train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
# 打印数据集的大小
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(test_dataset)}")
# 获取数据集的类别数量
num_classes = 3

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_mapping_path = './class_mapping.json'
# 保存类别映射
class_to_idx = train_dataset.class_to_idx
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
class_mapping = {
    "class_to_idx": class_to_idx,
    "idx_to_class": idx_to_class
}

with open(class_mapping_path, 'w') as f:
    json.dump(class_mapping, f, indent=4)

print(f"Class mapping saved to {class_mapping_path}")

# 初始化模型，根据数据集的类别数量调整 n_class 参数
model = mobilenetv3(n_class=num_classes, mode='large', width_mult=1.0)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

num_epochs = 100
best_accuracy = 0.0
# 早停阈值，即连续多少个 epoch 准确率没有提升就停止训练
early_stopping_patience = 10# 记录连续没有提升的 epoch 数量
early_stopping_counter = 0
save_path='./weights/best_model_weights1.pth'
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 打印每个 epoch 的损失
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

    # 验证模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy}%')

    # 保存最佳模型权重
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), save_path)
        print(f'Saved best model weights at epoch {epoch + 1} with accuracy {accuracy}%')
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1} due to no improvement in accuracy.')
            break

print('Training finished.')
print(f'Best accuracy: {best_accuracy}%')