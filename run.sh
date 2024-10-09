#!/bin/bash

mkdir -p "./data"

# GEMM
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 64 --n 12288 --k 49152 --iter 15 --warmup 5 > ./data/gemm_64.out
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 512 --n 12288 --k 49152 --iter 15 --warmup 5 > ./data/gemm_512.out
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 4096 --n 12288 --k 49152 --iter 15 --warmup 5 > ./data/gemm_4096.out
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --m 36864 --n 12288 --k 49152 --iter 15 --warmup 5 > ./data/gemm_36864.out

# GEMV
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 96 --m 1 --n 128 --k 64 --iter 15 --warmup 5 > ./data/bmm_1_64.out
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 96 --m 1 --n 128 --k 256 --iter 15 --warmup 5 > ./data/bmm_1_256.out
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 96 --m 1 --n 128 --k 1024 --iter 15 --warmup 5 > ./data/bmm_1_1024.out
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 64 --iter 15 --warmup 5 > ./data/bmm_32_64.out
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 256 --iter 15 --warmup 5 > ./data/bmm_32_256.out
OMP_NUM_THREADS=40 numactl -m 0 -C 0-39 python cpu_mbm.py --amx --bmm --bsz 3072 --m 1 --n 128 --k 1024 --iter 15 --warmup 5 > ./data/bmm_32_1024.out
