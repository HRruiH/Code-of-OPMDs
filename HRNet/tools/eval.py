# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import os
import sys
import shutil
import pprint

import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve, auc, confusion_matrix
from tabulate import tabulate
import pandas as pd

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import _init_paths
import models
from config import config
from config import update_config
from core.function import validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='../experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
                        type=str)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()
    for run in range(1,6):
      logger, final_output_dir, tb_log_dir = create_logger(
          config, args.cfg, 'valid')
  
      logger.info(pprint.pformat(args))
      logger.info(pprint.pformat(config))
  
      # cudnn related setting
      cudnn.benchmark = config.CUDNN.BENCHMARK
      torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
      torch.backends.cudnn.enabled = config.CUDNN.ENABLED
  
      # 设置设备
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      print(f"使用设备: {device}")
      
      model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
          config)
  
      dump_input = torch.rand(
          (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
      )
      logger.info(get_model_summary(model, dump_input))
  
          
      if config.TEST.MODEL_FILE:
          model_path = f'/home/huangr/HRNet/tools/output/imagenet/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100/best_model{run}.pth.tar'
      else:
          model_path = os.path.join(final_output_dir, f'final_state{run}.pth.tar')
      
      if os.path.exists(model_path):
          logger.info(f'=> loading model from {model_path}')
          checkpoint = torch.load(model_path, map_location=device)
          
          # 检查是否为完整的检查点
          if 'state_dict' in checkpoint:
              logger.info(f"=> loaded checkpoint with keys: {list(checkpoint.keys())}")
              pretrained_dict = checkpoint['state_dict']
          else:
              pretrained_dict = checkpoint
          
          # 处理DataParallel包装的模型
          if any(key.startswith('module.') for key in pretrained_dict):
              logger.info("Detected DataParallel prefix in state_dict keys")
              # 移除module.前缀
              pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
          
          # 获取模型当前的state_dict
          model_dict = model.state_dict()
          
          # 过滤掉不匹配的键
          pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                            if k in model_dict and model_dict[k].shape == v.shape}
          
          logger.info(f"=> loading {len(pretrained_dict)}/{len(model_dict)} layers")
          
          # 加载匹配的权重
          if len(pretrained_dict) == 0:
              logger.warning("No matching layers found for loading!")
          else:
              model_dict.update(pretrained_dict)
              model.load_state_dict(model_dict, strict=False)
          
          logger.info(f"=> loaded model weights")
      else:
          logger.error(f'=> model file {model_path} does not exist')
          raise FileNotFoundError(f'Model file {model_path} not found')
  
      gpus = list(config.GPUS)
      model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
  
      # Data loading code
      valdir = '/home/huangr/data_3/test'
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
  
      data_transform = {
          "val": transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])
      }
  
      validate_dataset = datasets.ImageFolder(
          root=valdir,
          transform=data_transform["val"]
      )
      val_num = len(validate_dataset)
  
      validate_loader = torch.utils.data.DataLoader(
          validate_dataset,
          batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
          shuffle=False,
          num_workers=config.WORKERS,
          pin_memory=True
      )
  
      print(f"测试集样本数: {val_num}")
  
      # 加载类别映射文件
      with open('./class_indices.json', 'r') as f:
          class_indices = json.load(f)
      class_names = [class_indices[str(i)] for i in range(len(class_indices))]
  
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
      plt.ylabel('True', fontsize=18)  # 纵坐标：真实类别，字体大小设置为 18
      plt.title('Confusion Matrix - HRNet-W18-C', fontsize=20)  # 图表标题，字体大小设置为 20
  
      # 设置横坐标和纵坐标的字体大小
      plt.xticks(fontsize=14)  # 设置横坐标字体大小
      plt.yticks(fontsize=14)  # 设置纵坐标字体大小
  
      plt.savefig(f"./results/confusion_matrix{run}.png", bbox_inches='tight')  # 保存为文件
  
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
      plt.title('ROC Curve - HRNet-W18-C', fontsize=20)
      plt.legend(loc='lower right', fontsize=14)
      plt.xticks(fontsize=14)
      plt.yticks(fontsize=14)
      plt.savefig(f"./results/roc_curve{run}.png", bbox_inches='tight')  # 保存为文件
  
      # 保存验证结果
      with open(f'./results/validation_results{run}.txt', 'w') as f:
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