import numpy as np
import os
import re

# 自定义目录地址（假设所有txt文件均在此目录下）
directory = './results'

# 检查目录是否存在
if not os.path.exists(directory):
    print(f"错误：目录 '{directory}' 不存在！")
    exit(1)

# 存储所有文件的评估结果
all_results = []

# 定义指标提取正则表达式（支持中文冒号、英文冒号及空格）
pattern = re.compile(
    r'(准确率 \(Accuracy\)|精确率 \(Precision\)|召回率 \(Recall\)|F1 值|Average AUC):\s*([\d.]+)'
)

# 遍历目录下的所有txt文件
for file in os.listdir(directory):
    if file.endswith('.txt'):
        file_path = os.path.join(directory, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                result = {}
                # 使用正则表达式提取指标值
                matches = pattern.findall(content)
                for metric, value in matches:
                    # 统一指标名称
                    metric_name = {
                        '准确率 (Accuracy)': '准确率（Accuracy）',
                        '精确率 (Precision)': '精确率（Precision）',
                        '召回率 (Recall)': '召回率（Recall）',
                        'F1 值': 'F1值',
                        'Average AUC': 'Average AUC'
                    }[metric]
                    result[metric_name] = float(value)
                all_results.append(result)
                print(f"成功读取文件：{file}，提取到 {len(result)} 个指标")
        except Exception as e:
            print(f"警告：读取文件 '{file}' 失败，错误：{e}")

# 检查是否有有效数据
if not all_results:
    print("错误：未找到任何有效txt文件或指标！")
    exit(1)

# 计算平均值±标准差（忽略缺失值）
def calculate_mean_std(metrics_list, metric_key):
    values = [item.get(metric_key, np.nan) for item in metrics_list]
    valid_values = [v for v in values if not np.isnan(v)]
    if len(valid_values) < 1:
        return "无有效数据"
    mean = np.mean(valid_values)
    std = np.std(valid_values, ddof=1) if len(valid_values) >= 2 else 0.0
    return f"{mean:.4f} ± {std:.4f}" if len(valid_values) >= 2 else f"{mean:.4f}（样本数={len(valid_values)}）"

# 定义需要统计的指标
metrics_to_calculate = [
    "准确率（Accuracy）",
    "精确率（Precision）",
    "召回率（Recall）",
    "F1值",
    "Average AUC"
]

# 生成统计结果
result_text = "--------------------- 多文件指标统计 ---------------------\n"
for metric in metrics_to_calculate:
    result_text += f"{metric}: {calculate_mean_std(all_results, metric)}\n"

# 保存结果到文件
output_file = os.path.join(directory, "汇总结果.txt")
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result_text)
    print(f"\n统计结果已保存至：{output_file}")
except Exception as e:
    print(f"\n保存结果失败：{e}")

# 打印结果到控制台
print(result_text)