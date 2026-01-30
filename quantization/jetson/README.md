# Jetson 设备量化说明

## 环境准备

在Jetson设备上量化之前，请确保您已安装所有必要的依赖项：

### 1.Torch
```bash
pip install -r quantization/jetson/Jetson-requirements.txt
```

### 2.TensorRT工具链环境
```bash
pip install -r quantization/jetson/Jetson-TensorRT-requirements.txt
```
在Jetson设备上下载以下包：
- [torch-2.3.0](https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl)
- [torchaudio-2.3.0](https://nvidia.box.com/shared/static/9si945yrzesspmg9up4ys380lqxjylc3.whl)
- [torchvision-0.18.0](https://nvidia.box.com/shared/static/9si945yrzesspmg9up4ys380lqxjylc3.whl)

然后使用pip下载对应包，例如：
```bash
pip install torch-2.3.0-cp310-cp310-linux_aarch64.whl
pip install torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
pip install torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
```


## 使用说明

参考不同工具链的 [quantization/README.md](../README.md) 与根目录 README 的“环境隔离/快速开始”。Jetson 端依赖与 x86/Windows 环境不同，请以本目录 requirements 文件为准。
