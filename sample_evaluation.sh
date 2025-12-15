#!/bin/bash

# 定义日志文件名（包含运行ID和时间戳）
RUN_ID=$1  # 获取传入的参数（5f3p9o34）
LOG_FILE="evaluation_logs\evaluation_${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"
# 重定向所有输出到日志文件
exec > "$LOG_FILE" 2>&1

if [ $# -ne 1 ]; then
    echo "try: $0 <run_id>, missing your run_id, please check!"
    exit 1
fi


RUN_ID=$1

python sample.py --run_id "$RUN_ID"

python evaluation.py --datasets Istanbul_small --task Stat --experiment_comments "$RUN_ID"


