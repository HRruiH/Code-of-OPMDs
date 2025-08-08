import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve, auc, confusion_matrix
from tabulate import tabulate
import pandas as pd
from utils import read_split_data
from model import efficientnet_b0 as create_model
from my_dataset import MyDataSet

def main():
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据预处理
    data_transform = {
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    _, _,_, _, val_images_path, val_images_label = read_split_data('/mnt/workspace/data_3')

    validate_dataset = MyDataSet(images_path=val_images_path,
                                 images_class=val_images_label,
                                 transform=data_transform["val"])

    val_num = len(validate_dataset)
    print(f"测试集样本数: {val_num}")

    # 加载类别名称
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"类别索引文件 {json_path} 不存在"

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    batch_size = 16
    validate_loader = DataLoader(validate_dataset,
                                 batch_size=batch_size, shuffle=False,
                                 num_workers=2)

    # 创建模型
    model = create_model(num_classes=3).to(device)

    # 加载模型权重
    model_weight_path = "./weights/best_model.pth"
    assert os.path.exists(model_weight_path), f"模型权重文件 {model_weight_path} 不存在"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # 验证模式
    model.eval()

    # 验证过程
    all_preds = []
    all_labels = []
    all_probs = []  # 存储所有样本的预测概率

    with torch.no_grad():
        for step, data in enumerate(validate_loader, start=0):
            images, labels = data
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)  # 获取概率分布
            predict_y = torch.max(probs, dim=1)[1]

            # 收集所有预测、标签和概率
            all_preds.extend(predict_y.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            rate = (step + 1) / len(validate_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print(f"\r{step + 1}/{len(validate_loader)} [{a}->{b}] {rate:.3%}", end="")

    print()

    # 将真实标签和预测概率转换为 numpy 数组
    real_labels = np.array(all_labels)
    pred_probs = np.array(all_probs)
    pred_classes = np.argmax(pred_probs, axis=1)  # 获取预测的类别索引

    # 获取类别名称列表
    class_names = [class_indict[str(i)] for i in range(len(class_indict))]

    # 计算每个类别的 AUC
    auc_scores = []
    nc = len(class_names)  # 获取类别数量
    for i in range(nc):
        fpr, tpr, _ = roc_curve((real_labels == i).astype(int), pred_probs[:, i])
        auc_score = roc_auc_score((real_labels == i).astype(int), pred_probs[:, i])
        auc_scores.append(auc_score)
        print(f"Class {i} ({class_names[i]}) AUC: {auc_score:.4f}")  # 打印每个类别的 AUC

    # 计算平均 AUC
    average_auc = np.mean(auc_scores)
    print(f"Average AUC: {average_auc:.4f}")  # 打印平均 AUC

    # 计算评估指标
    accuracy = accuracy_score(real_labels, pred_classes)
    precision = precision_score(real_labels, pred_classes, average='macro')
    recall = recall_score(real_labels, pred_classes, average='macro')
    f1 = f1_score(real_labels, pred_classes, average='macro')

    print("模型评估结果：")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1 值: {f1:.4f}")

    # 打印每个类别的详细评估报告
    print("\n每个类别详细评估报告：")
    report = classification_report(real_labels, pred_classes, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).T
    print(tabulate(report_df, headers='keys', tablefmt='grid', floatfmt=".4f"))

    # 计算混淆矩阵并绘制热力图
    conf_matrix = confusion_matrix(real_labels, pred_classes, normalize='true')  # 使用归一化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'fontsize': 18})  # 设置矩阵中数字的字体大小为 18
    plt.xlabel('Predicted', fontsize=18)  # 横坐标：预测类别，字体大小设置为 18
    plt.ylabel('True', fontsize=18)       # 纵坐标：真实类别，字体大小设置为 18
    plt.title('Confusion Matrix - Efficientb0', fontsize=20)  # 图表标题，字体大小设置为 20

    # 设置横坐标和纵坐标的字体大小
    plt.xticks(fontsize=14)  # 设置横坐标字体大小
    plt.yticks(fontsize=14)  # 设置纵坐标字体大小

    plt.savefig("confusion_matrix.png", bbox_inches='tight')  # 保存为文件
    plt.show()  # 在屏幕上显示

    # 计算每个类别的准确率
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    for i, acc in enumerate(class_accuracy):
        print(f"Class {i} ({class_names[i]}) Accuracy: {acc:.4f}")

    # 绘制 ROC 曲线
    plt.figure(figsize=(10, 8))
    for i in range(nc):
        fpr, tpr, _ = roc_curve((real_labels == i).astype(int), pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} ({class_names[i]}) (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('ROC Curve - Efficientb0', fontsize=20)
    plt.legend(loc='lower right', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("roc_curve.png", bbox_inches='tight')  # 保存为文件
    plt.show()

    # 保存验证结果
    with open('validation_results.txt', 'w') as f:
        f.write("模型评估结果：\n")
        f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
        f.write(f"精确率 (Precision): {precision:.4f}\n")
        f.write(f"召回率 (Recall): {recall:.4f}\n")
        f.write(f"F1 值: {f1:.4f}\n")
        f.write(f"Average AUC: {average_auc:.4f}\n\n")

        f.write("每个类别详细评估报告：\n")
        f.write(tabulate(report_df, headers='keys', tablefmt='grid', floatfmt=".4f"))
        f.write("\n\n")

        f.write("混淆矩阵 (比例):\n")
        f.write(str(conf_matrix))
        f.write("\n\n")

        f.write("每个类别的准确率:\n")
        for i, acc in enumerate(class_accuracy):
            f.write(f"Class {i} ({class_names[i]}) Accuracy: {acc:.4f}\n")
        f.write("\n")

        f.write("每个类别的 AUC:\n")
        for i in range(nc):
            f.write(f"Class {i} ({class_names[i]}) AUC: {auc_scores[i]:.4f}\n")

    print("验证完成，结果已保存到validation_results.txt、confusion_matrix.png和roc_curve.png")

if __name__ == '__main__':
    main()