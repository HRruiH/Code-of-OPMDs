import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from swin_transformer_model import swin_tiny_patch4_window7_224
from densent_model import densenet169

# 定义类别名称
CLASS_NAMES = ["Normal", "OCA", "OLK", "OLP", "OSF"]


class SwinWithCAM(nn.Module):
    def __init__(self, model):
        super(SwinWithCAM, self).__init__()
        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.layers = model.layers
        self.norm = model.norm
        self.avgpool = model.avgpool
        self.head = model.head  # 分类头（用于CAM权重）

    def forward(self, x):
        # 提取特征图
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x, H, W = layer(x, H, W)
        feature_map = self.norm(x)  # [B, L, C]
        B, L, C = feature_map.shape
        feature_map = feature_map.permute(0, 2, 1).view(B, C, H, W)  # 转换为[B, C, H, W]

        # 分类输出
        x = self.avgpool(feature_map.flatten(2))
        x = torch.flatten(x, 1)
        outputs = self.head(x)

        return outputs, feature_map


class DenseNetWithCAM(nn.Module):
    def __init__(self, model):
        super(DenseNetWithCAM, self).__init__()
        self.features = model.features  # 特征提取部分
        self.classifier = model.classifier  # 分类头

    def forward(self, x):
        feature_map = self.features(x)  # [B, C, H, W]
        x = F.relu(feature_map, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        outputs = self.classifier(x)
        return outputs, feature_map


def load_models(swin_weights_path, densenet_weights_path, device):
    # 加载Swin模型
    swin_model = swin_tiny_patch4_window7_224(num_classes=3)
    swin_model.load_state_dict(torch.load(swin_weights_path, map_location=device, weights_only=True))
    swin_model = SwinWithCAM(swin_model)
    swin_model.to(device)
    swin_model.eval()

    # 加载DenseNet模型
    densenet_model = densenet169(num_classes=3)
    densenet_model.load_state_dict(torch.load(densenet_weights_path, map_location=device, weights_only=True))
    densenet_model = DenseNetWithCAM(densenet_model)
    densenet_model.to(device)
    densenet_model.eval()

    return swin_model, densenet_model


def calculate_cam(feature_map, weight, class_idx):
    """计算类激活图"""
    C = feature_map.size(1)
    w = weight[class_idx].view(1, C, 1, 1)  # 类别权重
    cam = torch.sum(w * feature_map, dim=1, keepdim=True)  # 加权求和
    cam = F.relu(cam)  # 只保留正贡献
    cam = cam.squeeze().cpu().numpy()  # 转为numpy
    # 归一化
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def visualize_cam(image, cam):
    """将CAM叠加到原始图像"""
    # 图像反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.numpy().transpose(1, 2, 0)  # [H, W, 3]
    image = (image * std + mean) * 255.0
    image = image.astype(np.uint8)

    # 调整CAM大小并上色
    H, W = image.shape[:2]
    cam = cv2.resize(cam, (W, H))
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # 热图颜色映射
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)  # 转为RGB

    # 叠加图像
    superimposed = cv2.addWeighted(image, 0.6, cam, 0.4, 0)  # 原图权重0.6，CAM权重0.4
    return image, superimposed


def generate_cam_visualizations(swin_model, densenet_model, data_loader, device, save_dir="cam_results"):
    """生成并保存所有样本的CAM可视化结果"""
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            B = images.size(0)

            # 第一层：Swin分类
            swin_outputs, swin_features = swin_model(images)
            swin_probs = torch.softmax(swin_outputs, dim=1)
            swin_preds = torch.argmax(swin_probs, dim=1)

            # 第二层：DenseNet分类（针对OPMDs样本）
            opmd_mask = (swin_preds == 2)
            opmd_indices = torch.nonzero(opmd_mask).squeeze(dim=1).cpu().numpy()
            densenet_features = None

            if opmd_mask.any():
                opmd_images = images[opmd_mask]
                _, densenet_features = densenet_model(opmd_images)

            # 为每个样本生成CAM
            for i in range(B):
                img_idx = batch_idx * data_loader.batch_size + i
                true_label = labels[i].item()
                pred_label = swin_preds[i].item()
                image = images[i].cpu()

                # 计算CAM
                if pred_label < 2:
                    # 使用Swin特征
                    class_idx = pred_label
                    weight = swin_model.head.weight
                    cam = calculate_cam(swin_features[i:i + 1], weight, class_idx)
                else:
                    # 使用DenseNet特征
                    class_idx = pred_label - 2
                    mask_idx = np.where(opmd_indices == i)[0][0]
                    feat = densenet_features[mask_idx:mask_idx + 1]
                    weight = densenet_model.classifier.weight
                    cam = calculate_cam(feat, weight, class_idx)

                # 可视化并保存
                orig_img, superimposed_img = visualize_cam(image, cam)
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.imshow(orig_img)
                plt.title(f"True: {CLASS_NAMES[true_label]}")
                plt.axis('off')
                plt.subplot(122)
                plt.imshow(superimposed_img)
                plt.title(f"Pred: {CLASS_NAMES[pred_label]}")
                plt.axis('off')
                plt.savefig(f"{save_dir}/cam_{img_idx}.png", bbox_inches='tight')
                plt.close()  # 关闭图像释放内存

            print(f"已处理完第{batch_idx + 1}批，共{len(data_loader)}批")


def main():
    # 确保F可用
    global F
    import torch.nn.functional as F

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据
    val_dataset = ImageFolder(
        root=os.path.join("../data", "test"),
        transform=transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 加载模型
    swin_model, densenet_model = load_models(
        "swin_best_model2.pth",
        "densenet_best_model1.pth",
        device
    )

    # 生成CAM可视化
    generate_cam_visualizations(swin_model, densenet_model, val_loader, device)
    print("CAM可视化已完成，结果保存在cam_results文件夹中")


if __name__ == '__main__':
    main()