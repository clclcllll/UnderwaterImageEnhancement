# visualization.py：用于数据可视化的脚本

"""
plot_red_vs_blue： 绘制红色和蓝色通道的散点图，观察不同退化类型在颜色通道上的分布差异。
plot_brightness_distribution： 绘制亮度的直方图和核密度估计（KDE），比较不同退化类型的亮度分布。
plot_laplacian_variance： 绘制拉普拉斯方差的箱线图，分析不同退化类型的清晰度差异。
主函数： 调用各个绘图函数，将生成的图表保存到 results/metrics/plots/ 文件夹中。
"""

# 导入必要的库
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 设置Seaborn样式
sns.set(style="white grid")

# 读取分析结果数据
df = pd.read_csv('../../results/metrics/analysis_results.csv')

# 定义保存图表的路径
plot_folder = '../../results/metrics/plots/'


# 绘制红色和蓝色通道的散点图
def plot_red_vs_blue(df):
    """
    绘制红色和蓝色通道的散点图，按照退化类型着色

    参数：
    - df: 包含特征值和退化类型的DataFrame
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='red', y='blue', hue='degradation_type')
    plt.title('Red vs Blue Channel Distribution')
    plt.xlabel('Red Channel Value')
    plt.ylabel('Blue Channel Value')
    plt.legend(title='Degradation Type')
    plt.savefig(plot_folder + 'red_vs_blue_distribution.png')
    plt.close()


# 绘制亮度的直方图
def plot_brightness_distribution(df):
    """
    绘制亮度的直方图，按照退化类型区分

    参数：
    - df: 包含特征值和退化类型的DataFrame
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='brightness', hue='degradation_type', kde=True)
    plt.title('Brightness Distribution')
    plt.xlabel('Brightness')
    plt.ylabel('Count')
    plt.savefig(plot_folder + 'brightness_distribution.png')
    plt.close()


# 绘制拉普拉斯方差的箱线图
def plot_laplacian_variance(df):
    """
    绘制拉普拉斯方差的箱线图，按照退化类型区分

    参数：
    - df: 包含特征值和退化类型的DataFrame
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='degradation_type', y='laplacian_var')
    plt.title('Laplacian Variance by Degradation Type')
    plt.xlabel('Degradation Type')
    plt.ylabel('Laplacian Variance')
    plt.savefig(plot_folder + 'laplacian_variance.png')
    plt.close()


# 主函数
if __name__ == '__main__':
    # 调用绘图函数
    plot_red_vs_blue(df)
    plot_brightness_distribution(df)
    plot_laplacian_variance(df)
    print("所有图表已生成并保存到：" + plot_folder)
