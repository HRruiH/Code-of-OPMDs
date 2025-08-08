import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  # 添加 roc_auc_score 导入
from tabulate import tabulate
import pandas as pd
import seaborn as sns

import config
import model
from dataset import CUDAPrefetcher, ImageDataset
from torch import nn
from torch.utils.data import DataLoader
from utils import load_state_dict, accuracy, Summary, AverageMeter, ProgressMeter

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def build_model() -> nn.Module:
    model_instance = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes)
    model_instance = model_instance.to(device=config.device, memory_format=torch.channels_last)
    return model_instance


def load_dataset() -> CUDAPrefetcher:
    test_dataset = ImageDataset(config.test_image_dir,
                                config.image_size,
                                config.model_mean_parameters,
                                config.model_std_parameters,
                                "Test")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)
    return CUDAPrefetcher(test_dataloader, config.device)


def main() -> None:
    # Initialize the model
    model_instance = build_model()
    print(f"Build `{config.model_arch_name}` model successfully.")

    # Load model weights
    model_instance, _, _, _, _, _ = load_state_dict(model_instance, config.model_weights_path)
    print(f"Load `{config.model_arch_name}` "
          f"model weights `{os.path.abspath(config.model_weights_path)}` successfully.")

    # Start the verification mode of the model
    model_instance.eval()

    # Load test dataloader
    test_prefetcher = load_dataset()

    # Read class names from JSON file
    with open('./class_indices.json', 'r') as f:
        class_indices = json.load(f)
    class_names = [class_indices[str(i)] for i in range(len(class_indices))]

     # 运行评估
    all_labels, all_preds, all_probs = run_evaluation(model_instance, test_prefetcher, class_names)
    
    # 计算并打印指标
    accuracy, precision, recall, f1, average_auc = calculate_metrics(all_labels, all_preds, all_probs, class_names)
    
    # 可视化结果
    visualize_results(all_labels, all_preds, all_probs, class_names, accuracy, precision, recall, f1, average_auc)


def run_evaluation(model, test_prefetcher, class_names):
    # 初始化指标
    batches = len(test_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.4f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, acc1], prefix=f"Test: ")

    # 存储所有预测和标签
    all_labels = []
    all_preds = []
    all_probs = []

    # 初始化数据加载器
    test_prefetcher.reset()
    batch_data = test_prefetcher.next()
    end = time.time()
    
    # 添加批次计数器
    current_batch = 0

    with torch.no_grad():
        while batch_data is not None:
            # 转移数据到设备
            images = batch_data["image"].to(device=config.device, non_blocking=True)
            target = batch_data["target"].to(device=config.device, non_blocking=True)
            batch_size = images.size(0)

            # 推理
            output = model(images)
            probs = torch.softmax(output, dim=1)
            _, preds = torch.max(probs, 1)

            # 存储预测和标签
            all_labels.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # 测量准确率
            top1 = accuracy(output, target, topk=(1,))
            acc1.update(top1[0].item(), batch_size)

            # 测量时间
            batch_time.update(time.time() - end)
            end = time.time()

            # 使用批次计数器打印进度
            if current_batch % config.test_print_frequency == 0:
                progress.display(current_batch + 1)

            # 加载下一批次
            batch_data = test_prefetcher.next()
            current_batch += 1  # 更新批次计数器

    # 打印最终总结
    progress.display_summary()
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def calculate_metrics(all_labels, all_preds, all_probs, class_names):
    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # 计算AUC
    num_classes = len(class_names)
    if num_classes == 2:
        average_auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc_scores = []
        for i in range(num_classes):
            class_labels = (all_labels == i).astype(int)
            auc_scores.append(roc_auc_score(class_labels, all_probs[:, i]))
        average_auc = np.mean(auc_scores)
    
    # 打印指标
    print("\n=== 整体性能指标 ===")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    # Per-class metrics
    print("\n=== 各类别详细评估 ===")
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("\n=== 混淆矩阵 ===")
    print(pd.DataFrame(conf_matrix, index=class_names, columns=class_names))

    # Class-specific accuracy
    print("\n=== 各类别准确率 ===")
    for i, class_name in enumerate(class_names):
        class_acc = conf_matrix[i, i] / np.sum(conf_matrix[i, :])
        print(f"{class_name}: {class_acc:.4f}")
        
    return accuracy, precision, recall, f1, average_auc

def visualize_results(all_labels, all_preds, all_probs, class_names, accuracy, precision, recall, f1, average_auc):
    # Create results directory
    results_dir = os.path.join("results", config.exp_name)
    os.makedirs(results_dir, exist_ok=True)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    conf_matrix = confusion_matrix(all_labels, all_preds, normalize='true')
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,annot_kws={'fontsize': 18})
    plt.xlabel('Predicted', fontsize=18)  # 横坐标：预测类别，字体大小设置为 18
    plt.ylabel('True', fontsize=18)       # 纵坐标：真实类别，字体大小设置为 18
    plt.title('Confusion Matrix - Inception-v4', fontsize=20)  # 图表标题，字体大小设置为 20
    plt.tight_layout()
    plt.xticks(fontsize=14)  # 设置横坐标字体大小
    plt.yticks(fontsize=14) 
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
     
    # Plot ROC curves (for multi-class)
    plt.figure(figsize=(10, 8))
    num_classes = len(class_names)
    
    if num_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    else:
        # Multi-class classification
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('ROC Curve - Inception-v4', fontsize=20)
    plt.legend(loc='lower right', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
    plt.close()

    # 保存验证结果
    with open('./results/validation_results.txt', 'w') as f:
        f.write("模型评估结果：\n")
        f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
        f.write(f"精确率 (Precision): {precision:.4f}\n")
        f.write(f"召回率 (Recall): {recall:.4f}\n")
        f.write(f"F1 值: {f1:.4f}\n")
        f.write(f"Average AUC: {average_auc:.4f}\n\n")

    print("验证完成，结果已保存到validation_results.txt、confusion_matrix.png和roc_curve.png")



if __name__ == "__main__":
    main()