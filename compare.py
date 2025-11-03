import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_results(file_path):
    """加载验证结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_comparison(original_results, ablation_results, save_path="results/comparison_plot.png"):
    """绘制原模型和消融实验的对比图"""

    # 提取数据
    models = ['原模型 (8头)', '消融模型 (1头)']

    # 验证集指标
    val_loss = [
        original_results['validation']['loss'],
        ablation_results['validation']['loss']
    ]
    val_accuracy = [
        original_results['validation']['accuracy'],
        ablation_results['validation']['accuracy']
    ]
    val_perplexity = [
        original_results['validation']['perplexity'],
        ablation_results['validation']['perplexity']
    ]

    # 测试集指标
    test_loss = [
        original_results['test']['loss'],
        ablation_results['test']['loss']
    ]
    test_accuracy = [
        original_results['test']['accuracy'],
        ablation_results['test']['accuracy']
    ]
    test_perplexity = [
        original_results['test']['perplexity'],
        ablation_results['test']['perplexity']
    ]

    # 精确匹配率
    exact_match = [
        original_results['exact_match_rate'],
        ablation_results['exact_match_rate']
    ]

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Transformer模型消融实验对比 (多头注意力)', fontsize=16, fontweight='bold')

    # 设置颜色
    colors = ['#2E86AB', '#A23B72']

    # 1. 损失对比
    x = np.arange(len(models))
    width = 0.35

    # 验证集和测试集损失
    axes[0, 0].bar(x - width / 2, val_loss, width, label='验证集', color=colors[0], alpha=0.8)
    axes[0, 0].bar(x + width / 2, test_loss, width, label='测试集', color=colors[1], alpha=0.8)
    axes[0, 0].set_title('损失对比 (Loss)')
    axes[0, 0].set_ylabel('损失值')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 在柱状图上添加数值
    for i, (v, t) in enumerate(zip(val_loss, test_loss)):
        axes[0, 0].text(i - width / 2, v + 0.05, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        axes[0, 0].text(i + width / 2, t + 0.05, f'{t:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. 准确率对比
    axes[0, 1].bar(x - width / 2, [acc * 100 for acc in val_accuracy], width, label='验证集', color=colors[0],
                   alpha=0.8)
    axes[0, 1].bar(x + width / 2, [acc * 100 for acc in test_accuracy], width, label='测试集', color=colors[1],
                   alpha=0.8)
    axes[0, 1].set_title('准确率对比 (Accuracy)')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 在柱状图上添加数值
    for i, (v, t) in enumerate(zip(val_accuracy, test_accuracy)):
        axes[0, 1].text(i - width / 2, v * 100 + 1, f'{v * 100:.1f}%', ha='center', va='bottom', fontsize=9)
        axes[0, 1].text(i + width / 2, t * 100 + 1, f'{t * 100:.1f}%', ha='center', va='bottom', fontsize=9)

    # 3. 困惑度对比
    axes[0, 2].bar(x - width / 2, val_perplexity, width, label='验证集', color=colors[0], alpha=0.8)
    axes[0, 2].bar(x + width / 2, test_perplexity, width, label='测试集', color=colors[1], alpha=0.8)
    axes[0, 2].set_title('困惑度对比 (Perplexity)')
    axes[0, 2].set_ylabel('困惑度')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(models)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 在柱状图上添加数值
    for i, (v, t) in enumerate(zip(val_perplexity, test_perplexity)):
        axes[0, 2].text(i - width / 2, v + 0.2, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
        axes[0, 2].text(i + width / 2, t + 0.2, f'{t:.1f}', ha='center', va='bottom', fontsize=9)

    # 4. 精确匹配率对比
    axes[1, 0].bar(x, exact_match, color=colors, alpha=0.8)
    axes[1, 0].set_title('精确匹配率对比')
    axes[1, 0].set_ylabel('精确匹配率 (%)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models)
    axes[1, 0].grid(True, alpha=0.3)

    # 在柱状图上添加数值
    for i, v in enumerate(exact_match):
        axes[1, 0].text(i, v + 0.1, f'{v:.2f}%', ha='center', va='bottom', fontsize=10)

    # 5. 性能下降百分比
    performance_drop = {
        '验证集准确率': (val_accuracy[1] - val_accuracy[0]) / val_accuracy[0] * 100,
        '测试集准确率': (test_accuracy[1] - test_accuracy[0]) / test_accuracy[0] * 100,
        '精确匹配率': (exact_match[1] - exact_match[0]) / exact_match[0] * 100 if exact_match[0] > 0 else 0
    }

    metrics = list(performance_drop.keys())
    drops = list(performance_drop.values())

    bars = axes[1, 1].bar(metrics, drops, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    axes[1, 1].set_title('性能下降百分比 (消融 vs 原模型)')
    axes[1, 1].set_ylabel('性能下降 (%)')
    axes[1, 1].tick_params(axis='x', rotation=15)
    axes[1, 1].grid(True, alpha=0.3)

    # 在柱状图上添加数值
    for bar, drop in zip(bars, drops):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                        f'{drop:+.1f}%', ha='center', va='bottom', fontsize=10,
                        color='red' if drop < 0 else 'green')

    # 6. 综合评分对比
    # 综合评分 = 准确率 * 100 + 精确匹配率 - 困惑度
    def calculate_score(accuracy, exact_match, perplexity):
        return accuracy * 100 + exact_match - perplexity / 10

    original_score = calculate_score(test_accuracy[0], exact_match[0], test_perplexity[0])
    ablation_score = calculate_score(test_accuracy[1], exact_match[1], test_perplexity[1])

    scores = [original_score, ablation_score]
    score_labels = ['综合评分']

    axes[1, 2].bar(x, scores, color=colors, alpha=0.8)
    axes[1, 2].set_title('综合评分对比')
    axes[1, 2].set_ylabel('综合评分')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(models)
    axes[1, 2].grid(True, alpha=0.3)

    # 在柱状图上添加数值
    for i, score in enumerate(scores):
        axes[1, 2].text(i, score + 0.5, f'{score:.1f}', ha='center', va='bottom', fontsize=10)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    Path("results").mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # 打印详细对比信息
    print("=" * 60)
    print("消融实验详细对比分析")
    print("=" * 60)
    print(f"对比项目: 多头注意力 (8头 vs 1头)")
    print(f"原模型配置: d_model={original_results['config']['model']['d_model']}, "
          f"nhead={original_results['config']['model']['nhead']}")
    print(f"消融模型配置: d_model={ablation_results['config']['model']['d_model']}, "
          f"nhead={ablation_results['config']['model']['nhead']}")
    print()

    print("测试集性能对比:")
    print(f"  损失: {original_results['test']['loss']:.4f} → {ablation_results['test']['loss']:.4f} "
          f"({(ablation_results['test']['loss'] - original_results['test']['loss']):+.4f})")
    print(
        f"  准确率: {original_results['test']['accuracy'] * 100:.2f}% → {ablation_results['test']['accuracy'] * 100:.2f}% "
        f"({(ablation_results['test']['accuracy'] - original_results['test']['accuracy']) * 100:+.2f}%)")
    print(f"  困惑度: {original_results['test']['perplexity']:.2f} → {ablation_results['test']['perplexity']:.2f} "
          f"({ablation_results['test']['perplexity'] - original_results['test']['perplexity']:+.2f})")
    print(f"  精确匹配率: {original_results['exact_match_rate']:.2f}% → {ablation_results['exact_match_rate']:.2f}% "
          f"({ablation_results['exact_match_rate'] - original_results['exact_match_rate']:+.2f}%)")

    # 计算性能变化百分比
    accuracy_change = (ablation_results['test']['accuracy'] - original_results['test']['accuracy']) / \
                      original_results['test']['accuracy'] * 100
    perplexity_change = (ablation_results['test']['perplexity'] - original_results['test']['perplexity']) / \
                        original_results['test']['perplexity'] * 100

    print(f"\n性能变化百分比:")
    print(f"  准确率: {accuracy_change:+.2f}%")
    print(f"  困惑度: {perplexity_change:+.2f}%")
    print(f"  精确匹配率: {performance_drop['精确匹配率']:+.2f}%")


def main():
    # 加载结果文件
    original_file = "results/validation_results_8.json"
    ablation_file = "results/validation_results_1.json"

    try:
        original_results = load_results(original_file)
        ablation_results = load_results(ablation_file)

        print("成功加载结果文件")
        print(f"原模型: {original_file}")
        print(f"消融模型: {ablation_file}")

        # 绘制对比图
        plot_comparison(original_results, ablation_results)

    except FileNotFoundError as e:
        print(f"错误: 找不到结果文件 - {e}")
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件格式错误 - {e}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
