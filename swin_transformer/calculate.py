import numpy as np
import os

# 自定义目录地址
directory = './results'

# 检查目录是否存在
if not os.path.exists(directory):
    print(f"错误：目录 '{directory}' 不存在！")
    exit(1)

# 存储所有文件的评估结果
all_results = []

# 遍历指定目录下的所有txt文件
for file in os.listdir(directory):
    if file.endswith('.txt'):
        file_path = os.path.join(directory, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                result = {}
                # 提取关键指标（注意冒号格式可能为中文或英文）
                for line in content.split('\n'):
                    line = line.strip()
                    # 支持中文冒号和英文冒号
                    if "准确率 (Accuracy)" in line:
                        value = float(line.split(': ')[1])
                        result["准确率（Accuracy）"] = value
                    elif "精确率 (Precision)" in line:
                        value = float(line.split(': ')[1])
                        result["精确率（Precision）"] = value
                    elif "召回率 (Recall)" in line:
                        value = float(line.split(': ')[1])
                        result["召回率（Recall）"] = value
                    elif "F1 值" in line or "F1值" in line:
                        value = float(line.split(': ')[1])
                        result["F1值"] = value
                    elif "Average AUC" in line:
                        value = float(line.split(': ')[1])
                        result["Average AUC"] = value
                all_results.append(result)
                print(f"成功读取文件：{file}，提取到 {len(result)} 个指标")
        except Exception as e:
            print(f"警告：读取文件 '{file}' 失败，错误：{e}")

# 检查是否有有效数据
if not all_results:
    print("错误：未找到任何有效txt文件或指标！")
    exit(1)
# 定义指标提取函数
def extract_metric(metric_key):
    return [result.get(metric_key, np.nan) for result in all_results]

# 计算平均值±标准差（忽略缺失值）
def mean_std(values):
    valid_values = [v for v in values if not np.isnan(v)]
    if len(valid_values) < 2:
        return f"{np.mean(valid_values):.4f} ± 无（样本数<2）"
    mean = np.mean(valid_values)
    std = np.std(valid_values, ddof=1)  # 样本标准差
    return f"{mean:.4f} ± {std:.4f}"

# 提取各指标值
accuracy_values = extract_metric("准确率（Accuracy）")
precision_values = extract_metric("精确率（Precision）")
recall_values = extract_metric("召回率（Recall）")
f1_values = extract_metric("F1值")
auc_values = extract_metric("Average AUC")


# 添加统计结果
result_text = "--------------------- 统计结果 ---------------------\n"
result_text += f"准确率（Accuracy）: {mean_std(accuracy_values)}\n"
result_text += f"精确率（Precision）: {mean_std(precision_values)}\n"
result_text += f"召回率（Recall）: {mean_std(recall_values)}\n"
result_text += f"F1值: {mean_std(f1_values)}\n"
result_text += f"Average AUC: {mean_std(auc_values)}\n"

# 保存结果到文件
output_file = os.path.join(directory, f"汇总结果.txt")
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result_text)
    print(f"\n结果已成功保存到: {output_file}")
except Exception as e:
    print(f"\n警告：结果保存失败，错误: {e}")
else:
    # 同时在控制台打印结果
    print(result_text)