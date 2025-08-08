from ultralytics import YOLO
import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate

# 加载训练好的模型
model = YOLO('runs/classify/train/weights/best.pt')  # 替换为你的权重文件路径

# 测试集路径（假设每个类别的图片放在单独的文件夹中）
test_base_path = '/mnt/workspace/data_5/val'  # 替换为你的测试集路径

# 定义具体的类别名称
class_names = ['OCA','OPMDs','Normals']

# 获取类别名称和索引映射
class_to_index = {name: idx for idx, name in enumerate(class_names)}  # 类别名称映射为索引
index_to_class = {idx: name for idx, name in enumerate(class_names)}  # 索引映射为类别名称

# 存储真实标签和预测概率
real_labels = []
pred_probs = []

# 遍历测试集的每个类别文件夹
for class_name in sorted(os.listdir(test_base_path)):
    class_path = os.path.join(test_base_path, class_name)
    if os.path.isdir(class_path):  # 确保是文件夹
        # 获取当前类别的索引
        label = class_to_index[class_name]
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                # 使用模型进行预测
                results = model.predict(img_path)
                # print(f"Processing image: {img_path}")  # 打印正在处理的图片路径

                # 获取预测的概率分布
                if isinstance(results, list) and len(results) > 0:
                    result = results[0]  # 如果返回的是列表，取第一个元素
                else:
                    result = results

                if hasattr(result, 'probs'):
                    pred_prob = result.probs.data  # 提取 Probs 对象中的概率数据
                    if isinstance(pred_prob, torch.Tensor):
                        pred_prob = pred_prob.cpu().numpy()  # 转换为 NumPy 数组
                else:
                    raise ValueError("无法获取预测概率，请检查 results 的结构。")

                # print(f"Predicted probabilities:\n{pred_prob}")  # 打印预测概率
                pred_class = np.argmax(pred_prob)  # 获取预测的类别索引
                real_labels.append(label)
                pred_probs.append(pred_prob)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# 将真实标签转换为 numpy 数组
real_labels = np.array(real_labels)
print(f"Real labels: {real_labels}")  # 打印真实标签

# 将预测概率转换为二维 NumPy 数组
pred_probs = np.array(pred_probs)
pred_classes = np.argmax(pred_probs, axis=1)  # 获取预测的类别索引

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
plt.title('Confusion Matrix  - YOLOv11l-cls', fontsize=20)  # 图表标题，字体大小设置为 24

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
plt.title('ROC Curve - YOLOv11l-cls', fontsize=20)
plt.legend(loc='lower right', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("roc_curve.png", bbox_inches='tight')  # 保存为文件
