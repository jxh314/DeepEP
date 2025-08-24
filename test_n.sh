#!/bin/bash

if [ $# -eq 0 ]; then
    echo "用法: $0 <测试次数>"
    echo "示例: $0 100"
    exit 1
fi

TEST_COUNT=$1
NQPS=$2

export LD_LIBRARY_PATH=/usr/local/nvshmem/lib:${LD_LIBRARY_PATH}
# export LD_PRELOAD=/home/users/jinxihua/bccl/build/lib/libnccl.so.2.26.5 

export NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0
export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_IB_GID_INDEX=3

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,TUNING
export NCCL_DEBUG_FILE=$(pwd)/nccl_log/nccl.%h.%p.log
export NCCL_IB_GID_INDEX=3

export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_ADAPTIVE_ROUTING=1

export MASTER_ADDR=10.55.119.231
export MASTER_PORT=8362
export WORLD_SIZE=4
export RANK=1
export DEEPEP_NUM_QPS_FOR_DATA=$NQPS

# 创建带日期的日志目录
DATE_STR=$(date +%Y%m%d_%H%M%S)
LOG_DIR="test_logs_${DATE_STR}"
mkdir -p nccl_log
mkdir -p "$LOG_DIR"

echo "开始执行 $TEST_COUNT 次测试..."
echo "日志目录: $LOG_DIR"

# 执行测试
for i in $(seq 1 $TEST_COUNT); do
    echo "执行第 $i 次测试..."
    python3 tests/test_internode.py > "$LOG_DIR/test_${i}.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ 第 $i 次测试成功"
    else
        echo "✗ 第 $i 次测试失败"
    fi
    
    # 短暂等待
    sleep 1
done

echo "$TEST_COUNT 次测试完成，开始统计结果..."

# 统计结果
echo "=== 测试结果统计 ===" > "$LOG_DIR/summary.txt"
echo "生成时间: $(date)" >> "$LOG_DIR/summary.txt"
echo "测试次数: $TEST_COUNT" >> "$LOG_DIR/summary.txt"
echo "QPS数量: $DEEPEP_NUM_QPS_FOR_DATA" >> "$LOG_DIR/summary.txt"
echo "节点配置: $MASTER_ADDR:$MASTER_PORT, WORLD_SIZE=$WORLD_SIZE, RANK=$RANK" >> "$LOG_DIR/summary.txt"
echo "" >> "$LOG_DIR/summary.txt"

# 统计函数
calculate_stats() {
    local data="$1"
    local metric="$2"
    
    if [ -n "$data" ]; then
        local count=$(echo "$data" | wc -l)
        local min=$(echo "$data" | head -1)
        local max=$(echo "$data" | tail -1)
        local avg=$(echo "$data" | awk '{sum+=$1} END {print sum/NR}')
        local median=$(echo "$data" | awk '{arr[NR]=$1} END {if(NR%2==1) print arr[(NR+1)/2]; else print (arr[NR/2]+arr[NR/2+1])/2}')
        
        echo "数据点数量: $count" >> "$LOG_DIR/summary.txt"
        echo "最小值: $min" >> "$LOG_DIR/summary.txt"
        echo "最大值: $max" >> "$LOG_DIR/summary.txt"
        echo "平均值: $avg" >> "$LOG_DIR/summary.txt"
        echo "中位数: $median" >> "$LOG_DIR/summary.txt"
    else
        echo "未找到数据" >> "$LOG_DIR/summary.txt"
    fi
}

# 1. Dispatch FP8 RDMA带宽
echo "Dispatch FP8 RDMA带宽统计 (GB/s):" >> "$LOG_DIR/summary.txt"
fp8_rdma=$(grep -h "Best dispatch (FP8)" "$LOG_DIR"/test_*.log | grep -o 'BW: [0-9.]* GB/s (RDMA)' | cut -d' ' -f2 | sort -n)
calculate_stats "$fp8_rdma" "FP8 RDMA"
echo "" >> "$LOG_DIR/summary.txt"

# 2. Dispatch FP8 NVL带宽
echo "Dispatch FP8 NVL带宽统计 (GB/s):" >> "$LOG_DIR/summary.txt"
fp8_nvl=$(grep -h "Best dispatch (FP8)" "$LOG_DIR"/test_*.log | grep -o '[0-9.]* GB/s (NVL)' | cut -d' ' -f1 | sort -n)
calculate_stats "$fp8_nvl" "FP8 NVL"
echo "" >> "$LOG_DIR/summary.txt"

# 3. Dispatch BF16 RDMA带宽
echo "Dispatch BF16 RDMA带宽统计 (GB/s):" >> "$LOG_DIR/summary.txt"
bf16_rdma=$(grep -h "Best dispatch (BF16)" "$LOG_DIR"/test_*.log | grep -o 'BW: [0-9.]* GB/s (RDMA)' | cut -d' ' -f2 | sort -n)
calculate_stats "$bf16_rdma" "BF16 RDMA"
echo "" >> "$LOG_DIR/summary.txt"

# 4. Dispatch BF16 NVL带宽
echo "Dispatch BF16 NVL带宽统计 (GB/s):" >> "$LOG_DIR/summary.txt"
bf16_nvl=$(grep -h "Best dispatch (BF16)" "$LOG_DIR"/test_*.log | grep -o '[0-9.]* GB/s (NVL)' | cut -d' ' -f1 | sort -n)
calculate_stats "$bf16_nvl" "BF16 NVL"
echo "" >> "$LOG_DIR/summary.txt"

# 5. Combine RDMA带宽
echo "Combine RDMA带宽统计 (GB/s):" >> "$LOG_DIR/summary.txt"
combine_rdma=$(grep -h "Best combine" "$LOG_DIR"/test_*.log | grep -o 'BW: [0-9.]* GB/s (RDMA)' | cut -d' ' -f2 | sort -n)
calculate_stats "$combine_rdma" "Combine RDMA"
echo "" >> "$LOG_DIR/summary.txt"

# 6. Combine NVL带宽
echo "Combine NVL带宽统计 (GB/s):" >> "$LOG_DIR/summary.txt"
combine_nvl=$(grep -h "Best combine" "$LOG_DIR"/test_*.log | grep -o '[0-9.]* GB/s (NVL)' | cut -d' ' -f1 | sort -n)
calculate_stats "$combine_nvl" "Combine NVL"
echo "" >> "$LOG_DIR/summary.txt"

# 显示统计结果
echo ""
echo "=== 测试结果统计 ==="
cat "$LOG_DIR/summary.txt"

echo ""
echo "详细日志: $LOG_DIR/"
echo "统计结果: $LOG_DIR/summary.txt"
