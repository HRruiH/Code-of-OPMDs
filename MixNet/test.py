import os
import json
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    roc_curve, auc, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate
from mixnet import MixNet  # 导入 MixNet 模型定义
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理配置
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 类别配置
class_names = ["Normal","OCA", "OPMDs"]
class_indict = {str(i): class_name for i, class_name in enumerate(class_names)}
label_to_index = {v: k for k, v in class_indict.items()}  # 标签名称转索引

# 确保结果目录存在
os.makedirs("results", exist_ok=True)

# 模型初始化（在循环外创建一次即可）
model = MixNet(arch='s', num_classes=len(class_names)).to(device)

for run in range(1, 6):
    weights_path = f"weights/best_mixnet_s_trained{run}.pth"
    assert os.path.exists(weights_path), f"权重文件 {weights_path} 不存在！"
    
    # 加载模型权重
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  # 切换至评估模式

    # 存储结果的列表
    all_labels = []    # 真实标签名称
    all_preds = []     # 预测标签名称
    all_probs = []     # 预测概率分布

    # 遍历测试集文件夹
    test_root = "../data_3/test"
    assert os.path.exists(test_root), f"测试集目录 {test_root} 不存在！"
    
    for class_name in os.listdir(test_root):
        class_path = os.path.join(test_root, class_name)
        if not os.path.isdir(class_path):
            continue  # 跳过非目录文件
            
        # 获取图片列表（支持更多格式）
        img_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
        img_path_list = [
            os.path.join(class_path, f) for f in os.listdir(class_path)
            if any(f.lower().endswith(ext) for ext in img_extensions)
        ]

        with torch.no_grad():
            for start_idx in range(0, len(img_path_list), 8):
                end_idx = min(start_idx + 8, len(img_path_list))
                batch_img_paths = img_path_list[start_idx:end_idx]
                
                # 加载并预处理图片
                batch_imgs = []
                for img_path in batch_img_paths:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        img = data_transform(img)
                        batch_imgs.append(img)
                    except Exception as e:
                        print(f"加载图片失败: {img_path}, 错误: {e}")
                        continue  # 跳过加载失败的图片
                
                if not batch_imgs:
                    continue  # 跳过空批次

                # 模型推理
                batch_tensor = torch.stack(batch_imgs).to(device)
                outputs = model(batch_tensor).cpu()
                probs = torch.softmax(outputs, dim=1)  # 概率分布
                preds = torch.argmax(probs, dim=1)      # 预测类别索引

                # 收集结果
                for idx, pred_idx in enumerate(preds):
                    true_label = class_name
                    pred_label = class_indict[str(pred_idx.item())]
                    
                    all_labels.append(true_label)
                    all_preds.append(pred_label)
                    all_probs.append(probs[idx].numpy())

    # ---------------------
    # 评估指标计算
    # ---------------------
    # 转换为索引数组
    y_true = np.array([label_to_index[label] for label in all_labels], dtype=int)
    y_pred = np.array([label_to_index[pred] for pred in all_preds], dtype=int)
    y_probs = np.array(all_probs)  # 形状: (样本数, 类别数)

    # 基础分类指标
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro")
    macro_recall = recall_score(y_true, y_pred, average="macro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    
    # 分类报告（含各分类指标）
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        digits=4
    )
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # 按行归一化
    
    # ROC-AUC 计算
    num_classes = len(class_names)
    auc_scores = []
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)
    macro_auc = np.mean(auc_scores)  # 宏平均 AUC

    # ---------------------
    # 结果可视化
    # ---------------------
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title(f"Confusion Matrix - MixNet-S")
    plt.savefig(f"results/confusion_matrix{run}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 绘制归一化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"fontsize": 18}
    )
    plt.title(f"Confusion Matrix - MixNet-S", fontsize=20)
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('True', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"results/confusion_matrix_normalized{run}.png")
    plt.close()

    # 绘制 ROC 曲线
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc_scores[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(f"ROC Curve - MixNet-S", fontsize=20)
    plt.legend(loc="lower right", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"results/roc_curves{run}.png", bbox_inches='tight')
    plt.close()

    # ---------------------
    # 结果保存至文本文件
    # ---------------------
    with open(f"results/validation{run}.txt", "w", encoding="utf-8") as f:
        
        # 整体指标
        f.write("模型评估结果：\n")
        f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
        f.write(f"精确率 (Precision): {macro_precision:.4f}\n")
        f.write(f"召回率 (Recall): {macro_recall:.4f}\n")
        f.write(f"F1 值: {macro_recall:.4f}\n")
        f.write(f"Average AUC: {macro_auc:.4f}\n\n")
        
        # 类别级指标
        f.write("【类别级评估报告】\n")
        f.write(tabulate(
            pd.DataFrame(report).T,
            headers=["指标", "精确率", "召回率", "F1值", "样本数"],
            tablefmt="grid",
            floatfmt=".4f",
            stralign="left"
        ))
        f.write("\n\n")
        
        # 混淆矩阵
        f.write("【原始混淆矩阵】\n")
        f.write("    \t" + "\t".join(class_names) + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}\t" + "\t".join(map(str, cm[i])) + "\n")
        f.write("\n")
        
        # 归一化混淆矩阵
        f.write("【归一化混淆矩阵（行归一化）】\n")
        f.write("    \t" + "\t".join(class_names) + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}\t" + "\t".join([f"{x:.2f}" for x in cm_normalized[i]]) + "\n")
        f.write("\n")
        
        # AUC 值
        f.write("【各分类 AUC 值】\n")
        for i, name in enumerate(class_names):
            f.write(f"{name}: {auc_scores[i]:.4f}\n")
    
    print(f"Run {run} 评估完成，结果已保存至 results/validation_run{run}.txt")