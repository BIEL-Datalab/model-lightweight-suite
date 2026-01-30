# 通用工具函数目录

## 目录说明

本目录包含跨工具链复用的通用代码（数据加载、通用量化/评估工具、可视化等）。

约束：utils 必须保持“可安全复用”，避免引入隐式的全局依赖或副作用（尤其是避免因导入 utils 而被动导入 modelopt 相关依赖）。

## 文件结构

```
utils/
├── __init__.py                       # 仅暴露通用 API（避免聚合导入工具链特定模块）
├── data_loader.py                    # 数据加载工具（训练/验证/校准）
├── calibration_data.py               # 校准数据生成（numpy）
├── prepare_calib_images.py           # 校准图像准备（命令行脚本）
├── quantization_utils.py             # 通用量化工具（报告/尺寸/统计等）
├── torch_quantization_utils.py       # Torch.FX 量化专用工具
├── tensorrt_quantization_utils.py    # TensorRT 量化专用工具
├── modelopt_quantization_utils.py    # ModelOpt 量化专用工具（仅在 modelopt_env 使用）
├── visualization_utils.py            # 评估结果可视化图表
├── env_checks.py                     # Conda 环境名检查（避免误用环境）
├── eval_artifacts.py                 # 评估产物写入（JSON/Markdown/目录）
├── eval_dataset.py                   # 评估数据集加载（ImageFolder 子集）
├── eval_metrics.py                   # 分类指标计算（sklearn）
├── gpu_memory.py                     # GPU 内存读取与 CUDA cache 清理（可选依赖 pynvml）
└── README.md
```

## 使用说明

### 校准数据生成

```python
from utils.calibration_data import prepare_calibration_data

# 生成校准数据
calibration_data = prepare_calibration_data(
    data_root='data_set/imagenette',
    total_samples=500,
    save_path='data_set/calibration_data_500.npy'
)
```

### 数据加载

```python
from utils.data_loader import get_data_loaders

# 获取数据加载器
train_loader, val_loader, calibration_loader, train_dataset, val_dataset = get_data_loaders(
    data_root='data_set/imagenette',
    batch_size=50,
    num_workers=4,
    device='cpu'
)
```

### 校准图像准备

命令行方式（推荐）：

```bash
python utils/prepare_calib_images.py --input-dir data_set/imagenette/train --output-dir data_set/calib_images --num-images 2000
```

### 可视化结果

```python
from utils.visualization_utils import generate_visualizations

generate_visualizations(results={"fp32": fp32_result, "int8": int8_result}, results_dir="results/Torch/demo")
```

