```
UnderwaterImageEnhancement/
├── README.md                 # 项目说明文件
├── .gitignore                # Git 忽略规则
├── data/                     # 数据文件夹
│   ├── raw/                  # 原始数据（附件1和附件2）
│   │   ├── Attachment1/      # 附件1图像
│   │   ├── Attachment2/      # 附件2图像
│   ├── processed/            # 处理后的数据（如分类后的图像）
├── src/                      # 源代码文件夹
│   ├── analysis/             # 图像分析相关代码
│   │   ├── color_analysis.py # 偏色分析脚本
│   │   ├── blur_analysis.py  # 模糊度分析脚本
│   │   ├── light_analysis.py # 光照强度分析脚本
│   ├── enhancement/          # 图像增强算法代码
│   │   ├── deblurring.py     # 去模糊算法
│   │   ├── color_correction.py # 色彩校正算法
│   │   ├── low_light.py      # 弱光增强算法
│   │   ├── complex_model.py  # 复杂场景增强模型
│   ├── utils/                # 工具脚本
│       ├── metrics.py        # 评估指标计算脚本
│       ├── data_loader.py    # 数据加载工具
│       ├── visualization.py  # 结果可视化工具
├── models/                   # 存放训练好的模型文件
│   ├── complex_model.pth     # 复杂场景模型
├── results/                  # 实验结果
│   ├── enhanced_images/      # 增强后的图像
│   ├── metrics/              # 评估指标结果
│   │   ├── Answer1.xls       # 问题1的结果
│   │   ├── Answer2.xls       # 问题2的结果
│   │   ├── Answer3.xls       # 问题3的结果
├── docs/                     # 文档资料
│   ├── paper/                # 最终论文
│   │   ├── report.pdf        # 项目报告
│   │   ├── references.bib    # 参考文献
│   ├── slides/               # 答辩PPT
│       ├── presentation.pptx # 最终PPT
├── experiments/              # 实验相关记录
│   ├── configs/              # 实验配置文件
│   │   ├── config1.yaml      # 问题1的实验配置
│   │   ├── config2.yaml      # 问题2的实验配置
│   ├── logs/                 # 实验日志
│       ├── experiment1.log   # 问题1实验日志
│       ├── experiment2.log   # 问题2实验日志
├── tests/                    # 测试文件夹
│   ├── test_metrics.py       # 测试评估指标代码
│   ├── test_models.py        # 测试模型代码

```

