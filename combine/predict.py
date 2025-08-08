import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from swin_transformer_model import swin_tiny_patch4_window7_224
from densent_model import densenet169

# 定义类别名称，按索引顺序对应
CLASS_NAMES = ["Normal", "OCA", "OLK", "OLP", "OSF"]


def load_models(swin_weights_path, densenet_weights_path, device):
    # 加载Swin Transformer模型
    swin_model = swin_tiny_patch4_window7_224(num_classes=3)
    swin_model.load_state_dict(torch.load(swin_weights_path, map_location=device))
    swin_model.to(device)
    swin_model.eval()

    densenet_model = densenet169(num_classes=3)
    densenet_model.load_state_dict(torch.load(densenet_weights_path, map_location=device))
    densenet_model.eval()
    densenet_model.to(device)

    return swin_model, densenet_model


def predict_single_image(swin_model, densenet_model, image_path, device):
    """
    预测单张图片的类别概率
    :param swin_model: 第一层Swin模型
    :param densenet_model: 第二层DenseNet模型
    :param image_path: 图片路径
    :param device: 运行设备
    :return: 五个类别的概率列表
    """
    # 定义与训练时相同的预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载并预处理图片
    image = Image.open(image_path).convert('RGB')
    image = data_transform(image).unsqueeze(0)  # 添加批次维度
    image = image.to(device)

    with torch.no_grad():
        # 初始化五分类概率张量
        full_probs = torch.zeros((1, 5), device=device)

        # 第一层分类 (Swin Transformer)
        swin_outputs = swin_model(image)
        swin_probs = torch.softmax(swin_outputs, dim=1)

        # 将第一层的概率填充到前三个类别
        full_probs[:, :3] = swin_probs
        swin_preds = torch.argmax(swin_probs, dim=1)

        # 对预测为OPMDs的样本进行第二层分类
        if swin_preds == 2:  # OPMDs对应索引2
            # 第二层分类 (DenseNet)
            densenet_outputs = densenet_model(image)
            densenet_probs = torch.softmax(densenet_outputs, dim=1)

            # 将第二层的概率填充到对应位置（OLK, OSF, OLP对应索引2,3,4）
            full_probs[:, 2:] = densenet_probs

    # 转换为numpy数组并返回
    return full_probs.squeeze().cpu().numpy()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    swin_model, densenet_model = load_models(
        "swin_best_model2.pth",
        "densenet_best_model1.pth",
        device
    )

    # 单张图片预测
    image_path = input("请输入图片路径: ")

    if os.path.exists(image_path):
        probabilities = predict_single_image(swin_model, densenet_model, image_path, device)

        print("\n图片分类概率:")
        for class_name, prob in zip(CLASS_NAMES, probabilities):
            print(f"{class_name}: {prob:.6f} ({prob * 100:.2f}%)")

        # 输出预测类别和置信度
        predicted_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = probabilities[predicted_idx] * 100

        print(f"\n预测类别: {predicted_class}")
        print(f"置信度: {confidence:.2f}%")

        # 保存预测结果
        with open('prediction_result.txt', 'w') as f:
            f.write(f"图片路径: {image_path}\n\n")
            f.write("分类概率:\n")
            for class_name, prob in zip(CLASS_NAMES, probabilities):
                f.write(f"{class_name}: {prob:.6f} ({prob * 100:.2f}%)\n")
            f.write(f"\n预测类别: {predicted_class}\n")
            f.write(f"置信度: {confidence:.2f}%")

        print(f"\n预测结果已保存至 prediction_result.txt")
    else:
        print(f"错误: 图片路径不存在 - {image_path}")


if __name__ == '__main__':
    main()