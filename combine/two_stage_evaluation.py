import os
import torch
import numpy as np
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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

    densent_model = densenet169(num_classes=3)
    densent_model.load_state_dict(torch.load(densenet_weights_path, map_location=device))
    densent_model.eval()
    densent_model.to(device)

    return swin_model, densent_model


def evaluate_two_stage(swin_model, densenet_model, data_loader, device):
    all_true_labels = []
    all_pred_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.size(0)

            # 初始化五分类概率张量
            full_probs = torch.zeros((batch_size, 5), device=device)

            # 第一层分类 (Swin Transformer)
            swin_outputs = swin_model(images)
            swin_probs = torch.softmax(swin_outputs, dim=1)

            # 将第一层的概率填充到前三个类别
            full_probs[:, :3] = swin_probs
            swin_preds = torch.argmax(swin_probs, dim=1)

            # 对预测为OPMDs的样本进行第二层分类
            opmd_mask = (swin_preds == 2)  # OPMDs对应索引2
            if opmd_mask.any():
                opmd_images = images[opmd_mask]
                opmd_labels = labels[opmd_mask]

                # 第二层分类 
                densenet_outputs = densenet_model(opmd_images)
                densenet_probs = torch.softmax(densenet_outputs, dim=1)

                # 将第二层的概率填充到对应位置（OLK, OSF, OLP对应索引2,3,4）
                full_probs[opmd_mask, 2:] = densenet_probs

                densenet_preds = torch.argmax(densenet_probs, dim=1)
                swin_preds[opmd_mask] = densenet_preds + 2  # 映射到整体类别索引

            # 收集预测结果
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(swin_preds.cpu().numpy())
            all_probs.extend(full_probs.cpu().numpy())

    # 计算评估指标（宏平均）
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    precision = precision_score(all_true_labels, all_pred_labels, average='macro')
    recall = recall_score(all_true_labels, all_pred_labels, average='macro')
    f1 = f1_score(all_true_labels, all_pred_labels, average='macro')

    # 计算每个类别的单独指标
    per_class_precision = precision_score(all_true_labels, all_pred_labels, average=None)
    per_class_recall = recall_score(all_true_labels, all_pred_labels, average=None)
    per_class_f1 = f1_score(all_true_labels, all_pred_labels, average=None)

    # 计算AUC
    num_classes = 5
    all_true_labels_onehot = np.zeros((len(all_true_labels), num_classes))
    all_true_labels_onehot[np.arange(len(all_true_labels)), all_true_labels] = 1
    auc = roc_auc_score(all_true_labels_onehot, all_probs, multi_class='ovr')
    all_true_labels_onehot = np.array(all_true_labels_onehot)
    all_probs = np.array(all_probs)
    per_class_auc = [roc_auc_score(all_true_labels_onehot[:, i], all_probs[:, i]) for i in range(num_classes)]

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'per_class_auc': per_class_auc
    }, all_true_labels, all_pred_labels, all_probs


def plot_macro_roc_curve(true_labels, probs, class_names):
    """绘制宏平均ROC曲线和各类别的ROC曲线"""
    plt.figure(figsize=(10, 8))

    # 将真实标签转换为one-hot编码
    num_classes = len(class_names)
    true_labels_onehot = np.zeros((len(true_labels), num_classes))
    true_labels_onehot[np.arange(len(true_labels)), true_labels] = 1

    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_onehot[:, i], probs[:, i])
        roc_auc[i] = roc_auc_score(true_labels_onehot[:, i], probs[:, i])
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

    # 计算宏平均ROC曲线
    # 首先汇总所有FPR
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # 然后插值所有TPR到相同的FPR点
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # 最后取平均并计算AUC
    mean_tpr /= num_classes
    roc_auc_macro = roc_auc_score(true_labels_onehot, probs, average='macro')

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('ROC Curve', fontsize=20)
    plt.legend(loc="lower right", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.grid(True, alpha=0.3)
    plt.savefig("macro_roc_curve.png", bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'fontsize': 18})
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('True', fontsize=18)
    plt.title('Confusion Matrix', fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("confusion_matrix.png", bbox_inches='tight')
    plt.show()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载验证数据集
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    val_dataset = ImageFolder(
        root=os.path.join("../data", "test"),
        transform=data_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    # 加载模型
    swin_model, densenet_model = load_models(
        "swin_best_model2.pth",
        "densenet_best_model1.pth",
        device
    )

    # 评估
    metrics, all_true_labels, all_pred_labels, all_probs = evaluate_two_stage(
        swin_model, densenet_model, val_loader, device)

    # 打印整体评估结果
    print("\n两层分类系统评估结果 (宏平均):")
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall): {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    print(f"AUC值: {metrics['auc']:.4f}")

    # 打印每个类别的详细指标
    print("\n各类别评估指标:")
    print(f"{'类别':<12}{'精确率':<12}{'召回率':<12}{'F1值':<12}{'AUC值':<12}")
    print("-" * 60)
    for i, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name:<12}{metrics['per_class_precision'][i]:.4f}{metrics['per_class_recall'][i]:.4f}{metrics['per_class_f1'][i]:.4f}{metrics['per_class_auc'][i]:.4f}")

    # 绘制宏平均ROC曲线（替代PR曲线）
    plot_macro_roc_curve(np.array(all_true_labels), np.array(all_probs), CLASS_NAMES)

    # 绘制混淆矩阵（使用类别名称）
    plot_confusion_matrix(np.array(all_true_labels), np.array(all_pred_labels), CLASS_NAMES)

    # 保存评估结果
    with open('validation_results.txt', 'w') as f:
        f.write("两层分类系统评估结果 (宏平均):\n")
        f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n")
        f.write(f"精确率 (Precision): {metrics['precision']:.4f}\n")
        f.write(f"召回率 (Recall): {metrics['recall']:.4f}\n")
        f.write(f"F1分数: {metrics['f1']:.4f}\n")
        f.write(f"AUC值: {metrics['auc']:.4f}\n\n")

        f.write("各类别评估指标:\n")
        f.write(f"{'类别':<12}{'精确率':<12}{'召回率':<12}{'F1值':<12}{'AUC值':<12}\n")
        f.write("-" * 60 + "\n")
        for i, class_name in enumerate(CLASS_NAMES):
            f.write(
                f"{class_name:<12}{metrics['per_class_precision'][i]:.4f}{metrics['per_class_recall'][i]:.4f}{metrics['per_class_f1'][i]:.4f}{metrics['per_class_auc'][i]:.4f}\n")


if __name__ == '__main__':
    main()