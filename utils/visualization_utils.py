import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'AR PL UKai CN', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

def generate_visualizations(results: dict, results_dir: str) -> None:
    """生成模型评估结果的可视化图表。
    
    根据模型评估结果生成多种可视化图表，包括精度指标对比、推理时间对比、
    吞吐量对比、GPU内存占用对比、CPU内存占用对比以及混淆矩阵。
    
    Args:
        results (dict): 包含模型评估结果的字典，键为模型类型（如'fp32'、'int8'），
            值为该模型的评估指标字典。
        results_dir (str): 保存可视化图表的目录路径。
    
    Returns:
        None
    """
    print("生成可视化图表...")
    models = ['fp32', 'int8']
    model_names = ['FP32', 'INT8']
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数']
    colors = ['#1f77b4', '#ff7f0e']

    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    for i, model in enumerate(models):
        metric_values = [results[model][metric] for metric in metrics]
        bars = plt.bar(x + i*width - width/2, metric_values, width, label=model_names[i], color=colors[i])
        
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=10)

    plt.xlabel('评估指标')
    plt.ylabel('得分')
    plt.title('模型精度指标对比')
    plt.xticks(x, metric_names)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_metrics_comparison.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    inference_times = [results[model]['avg_inference_time_s'] for model in models]
    bars1 = plt.bar(model_names, inference_times, color=colors)
    plt.ylabel('平均推理时间 (秒)')
    plt.title('平均推理时间对比')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, v in zip(bars1, inference_times):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.001, f'{v:.4f}s', ha='center', va='bottom')

    plt.subplot(1, 2, 2)
    throughputs = [results[model]['avg_throughput_fps'] for model in models]
    bars2 = plt.bar(model_names, throughputs, color=colors)
    plt.ylabel('平均吞吐量 (FPS)')
    plt.title('平均吞吐量对比')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, v in zip(bars2, throughputs):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 1, f'{v:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance_comparison.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    gpu_mems = [results[model]['avg_gpu_memory_mb'] for model in models]
    bars3 = plt.bar(model_names, gpu_mems, color=colors)
    plt.ylabel('平均GPU内存占用 (MB)')
    plt.title('平均GPU内存占用对比')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, v in zip(bars3, gpu_mems):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f'{v:.1f}MB', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'gpu_memory_comparison.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    cpu_mems = [results[model]['avg_cpu_memory_mb'] for model in models]
    bars4 = plt.bar(model_names, cpu_mems, color=colors)
    plt.ylabel('平均CPU内存占用 (MB)')
    plt.title('平均CPU内存占用对比')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar, v in zip(bars4, cpu_mems):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f'{v:.1f}MB', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cpu_memory_comparison.png'), dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, (model, name) in enumerate(zip(models, model_names)):
        cm = np.array(results[model]['confusion_matrix'])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', 
                    cmap='Blues', ax=axes[i], cbar_kws={'label': '比例'})
        axes[i].set_title(f'{name} 混淆矩阵 (归一化)')
        axes[i].set_xlabel('预测标签')
        axes[i].set_ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()

    print("可视化图表已保存。")
