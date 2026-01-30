#!/bin/bash

# 转换ONNX模型到TensorRT引擎的脚本
# 增强了鲁棒性，支持命令行参数输入

# 默认值
DEFAULT_ONNX_MODEL="models/quantized/int8/resnet50_imagenette_modelopt_int8_x86.onnx"
DEFAULT_ENGINE_OUTPUT="models/quantized/int8/resnet50_imagenette_modelopt_int8_x86.engine"
DEFAULT_WORKSPACE_SIZE="2048"
DEFAULT_FP16="true"
DEFAULT_SKIP_INFERENCE="true"
DEFAULT_MIN_SHAPE="input:1x3x224x224"
DEFAULT_OPT_SHAPE="input:32x3x224x224"
DEFAULT_MAX_SHAPE="input:64x3x224x224"
DEFAULT_VERBOSE="true"
DEFAULT_BEST="true"

# 显示帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help                  显示帮助信息"
    echo "  -i, --input <path>          输入ONNX模型路径 (默认: $DEFAULT_ONNX_MODEL)"
    echo "  -o, --output <path>         输出TensorRT引擎路径 (默认: $DEFAULT_ENGINE_OUTPUT)"
    echo "  -w, --workspace <size>      工作区大小(MB) (默认: $DEFAULT_WORKSPACE_SIZE)"
    echo "  --fp16 <true/false>        是否启用FP16精度 (默认: $DEFAULT_FP16)"
    echo "  --skip-inference <true/false> 是否跳过推理 (默认: $DEFAULT_SKIP_INFERENCE)"
    echo "  --min-shape <shape>         最小输入形状 (默认: $DEFAULT_MIN_SHAPE)"
    echo "  --opt-shape <shape>         最优输入形状 (默认: $DEFAULT_OPT_SHAPE)"
    echo "  --max-shape <shape>         最大输入形状 (默认: $DEFAULT_MAX_SHAPE)"
    echo "  --verbose <true/false>      是否启用详细输出 (默认: $DEFAULT_VERBOSE)"
    echo "  --best <true/false>         是否启用最佳优化 (默认: $DEFAULT_BEST)"
    echo ""
    echo "示例:"
    echo "  $0 -i models/my_model.onnx -o models/my_model.engine -w 4096"
    echo "  $0 --input models/my_model.onnx --fp16 true --skip-inference false"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--input)
            ONNX_MODEL="$2"
            shift 2
            ;;
        -o|--output)
            ENGINE_OUTPUT="$2"
            shift 2
            ;;
        -w|--workspace)
            WORKSPACE_SIZE="$2"
            shift 2
            ;;
        --fp16)
            FP16="$2"
            shift 2
            ;;
        --skip-inference)
            SKIP_INFERENCE="$2"
            shift 2
            ;;
        --min-shape)
            MIN_SHAPE="$2"
            shift 2
            ;;
        --opt-shape)
            OPT_SHAPE="$2"
            shift 2
            ;;
        --max-shape)
            MAX_SHAPE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="$2"
            shift 2
            ;;
        --best)
            BEST="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 设置默认值
ONNX_MODEL=${ONNX_MODEL:-$DEFAULT_ONNX_MODEL}
ENGINE_OUTPUT=${ENGINE_OUTPUT:-$DEFAULT_ENGINE_OUTPUT}
WORKSPACE_SIZE=${WORKSPACE_SIZE:-$DEFAULT_WORKSPACE_SIZE}
FP16=${FP16:-$DEFAULT_FP16}
SKIP_INFERENCE=${SKIP_INFERENCE:-$DEFAULT_SKIP_INFERENCE}
MIN_SHAPE=${MIN_SHAPE:-$DEFAULT_MIN_SHAPE}
OPT_SHAPE=${OPT_SHAPE:-$DEFAULT_OPT_SHAPE}
MAX_SHAPE=${MAX_SHAPE:-$DEFAULT_MAX_SHAPE}
VERBOSE=${VERBOSE:-$DEFAULT_VERBOSE}
BEST=${BEST:-$DEFAULT_BEST}

# 检查trtexec是否可用
if ! command -v trtexec &> /dev/null; then
    echo "错误: trtexec 命令未找到，请确保TensorRT已正确安装"
    exit 1
fi

# 检查输入文件是否存在
if [ ! -f "$ONNX_MODEL" ]; then
    echo "错误: 输入文件 '$ONNX_MODEL' 不存在"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR=$(dirname "$ENGINE_OUTPUT")
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "创建输出目录: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# 构建trtexec命令
TRTEXEC_CMD="trtexec"

# 添加ONNX模型路径
TRTEXEC_CMD="$TRTEXEC_CMD --onnx=$ONNX_MODEL"

# 添加引擎输出路径
TRTEXEC_CMD="$TRTEXEC_CMD --saveEngine=$ENGINE_OUTPUT"

# 添加最佳优化选项
if [ "$BEST" = "true" ]; then
    TRTEXEC_CMD="$TRTEXEC_CMD --best"
fi

# 添加详细输出选项
if [ "$VERBOSE" = "true" ]; then
    TRTEXEC_CMD="$TRTEXEC_CMD --verbose"
fi

# 添加工作区大小
TRTEXEC_CMD="$TRTEXEC_CMD --memPoolSize=workspace:$WORKSPACE_SIZE"

# 添加FP16选项
if [ "$FP16" = "true" ]; then
    TRTEXEC_CMD="$TRTEXEC_CMD --fp16"
fi

# 添加输入形状范围
TRTEXEC_CMD="$TRTEXEC_CMD --minShapes=$MIN_SHAPE"
TRTEXEC_CMD="$TRTEXEC_CMD --optShapes=$OPT_SHAPE"
TRTEXEC_CMD="$TRTEXEC_CMD --maxShapes=$MAX_SHAPE"

# 添加跳过推理选项
if [ "$SKIP_INFERENCE" = "true" ]; then
    TRTEXEC_CMD="$TRTEXEC_CMD --skipInference"
fi

# 添加分析选项
TRTEXEC_CMD="$TRTEXEC_CMD --profilingVerbosity=detailed"

# 添加导出选项
base_name=$(basename "$ENGINE_OUTPUT" .engine)
out_dir=$(dirname "$ENGINE_OUTPUT")
TRTEXEC_CMD="$TRTEXEC_CMD --exportTimes=${out_dir}/${base_name}_build_time.json"
TRTEXEC_CMD="$TRTEXEC_CMD --exportProfile=${out_dir}/${base_name}_profile.json"
TRTEXEC_CMD="$TRTEXEC_CMD --exportLayerInfo=${out_dir}/${base_name}_layers_info.json"

# 显示执行信息
echo "="*60
echo "ONNX 到 TensorRT 转换脚本"
echo "="*60
echo "输入ONNX模型: $ONNX_MODEL"
echo "输出TensorRT引擎: $ENGINE_OUTPUT"
echo "工作区大小: ${WORKSPACE_SIZE}MB"
echo "FP16精度: $FP16"
echo "跳过推理: $SKIP_INFERENCE"
echo "最小输入形状: $MIN_SHAPE"
echo "最优输入形状: $OPT_SHAPE"
echo "最大输入形状: $MAX_SHAPE"
echo "详细输出: $VERBOSE"
echo "最佳优化: $BEST"
echo "="*60
echo "执行命令:"
echo "$TRTEXEC_CMD"
echo "="*60

# 执行命令
echo "开始转换..."
$TRTEXEC_CMD

# 检查命令执行结果
if [ $? -eq 0 ]; then
    echo "="*60
echo "转换成功!"
echo "输出文件:"
echo "  - TensorRT引擎: $ENGINE_OUTPUT"
echo "  - 构建时间: ${out_dir}/${base_name}_build_time.json"
echo "  - 配置文件: ${out_dir}/${base_name}_profile.json"
echo "  - 层信息: ${out_dir}/${base_name}_layers_info.json"
echo "="*60
else
    echo "="*60
echo "转换失败!"
echo "="*60
    exit 1
fi
