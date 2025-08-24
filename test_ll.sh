#!/bin/bash
NQPS=${1:-1}

# export LD_LIBRARY_PATH=/workspace:/usr/local/nvshmem/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/nvshmem/lib:${LD_LIBRARY_PATH}
# export LD_PRELOAD=/workspace/libnccl.so.2.25.1 

export NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0
export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_IB_ENABLE_IBGDA=true

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,COLL,TUNING
export NCCL_DEBUG_FILE=$(pwd)/nccl_log/nccl.%h.%p.log
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_ADAPTIVE_ROUTING=1
#export P2P_NVL_CHUNKSIZE=131072
#export NCCL_NVLS_ENABLE=0
#export NCCL_CUMEM_ENABLE=0

export MASTER_ADDR=10.55.119.231
export MASTER_PORT=8362
export WORLD_SIZE=2
export RANK=0
export DEEPEP_NUM_QPS_PER_EXPERT=$NQPS

python3 tests/test_low_latency.py