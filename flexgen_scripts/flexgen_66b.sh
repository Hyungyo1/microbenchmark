##Online

OMP_NUM_THREADS=40 numactl -m 0,1 -C 0-39 python -m flexgen.flex_opt --model facebook/opt-66b --path _DUMMY_ --percent 55 45 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --prompt-len 32 --gen-len 32 --offload-dir ./ --cpu-cache-compute --pin-weight 0 &> "./opt66b_32_32_b=1.log"

OMP_NUM_THREADS=40 numactl -m 0,1 -C 0-39 python -m flexgen.flex_opt --model facebook/opt-66b --path _DUMMY_ --percent 53 47 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --prompt-len 256 --gen-len 32 --offload-dir ./ --cpu-cache-compute --pin-weight 0 &> "./opt66b_256_32_b=1.log"

OMP_NUM_THREADS=40 numactl -m 0,1 -C 0-39 python -m flexgen.flex_opt --model facebook/opt-66b --path _DUMMY_ --percent 50 50 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --prompt-len 2016 --gen-len 32 --offload-dir ./ --cpu-cache-compute --pin-weight 0 &> "./opt66b_2016_32_b=1.log" 

OMP_NUM_THREADS=40 numactl -m 0,1 -C 0-39 python -m flexgen.flex_opt --model facebook/opt-66b --path _DUMMY_ --percent 53 47 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --prompt-len 32 --gen-len 256 --offload-dir ./ --cpu-cache-compute --pin-weight 0 &> "./opt66b_32_256_b=1.log"

OMP_NUM_THREADS=40 numactl -m 0,1 -C 0-39 python -m flexgen.flex_opt --model facebook/opt-66b --path _DUMMY_ --percent 50 50 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --prompt-len 256 --gen-len 256 --offload-dir ./ --cpu-cache-compute --pin-weight 0 &> "./opt66b_256_256_b=1.log"

OMP_NUM_THREADS=40 numactl -m 0,1 -C 0-39 python -m flexgen.flex_opt --model facebook/opt-66b --path _DUMMY_ --percent 50 50 100 0 100 0 --gpu-batch-size 1 --num-gpu-batches 1 --prompt-len 1792 --gen-len 256 --offload-dir ./ --cpu-cache-compute --pin-weight 0 &> "./opt66b_1792_256_b=1.log"

##Offline

OMP_NUM_THREADS=40 numactl -m 0,1 -C 0-39 python -m flexgen.flex_opt --model facebook/opt-66b --path _DUMMY_ --percent 50 50 0 100 0 100 --gpu-batch-size 64 --num-gpu-batches 1 --prompt-len 32 --gen-len 32 --offload-dir ./ --cpu-cache-compute --pin-weight 0 &> "./opt66b_32_32_b=64.log"

OMP_NUM_THREADS=40 numactl -m 0,1 -C 0-39 python -m flexgen.flex_opt --model facebook/opt-66b --path _DUMMY_ --percent 40 60 0 100 0 100 --gpu-batch-size 64 --num-gpu-batches 1 --prompt-len 32 --gen-len 256 --offload-dir ./ --cpu-cache-compute --pin-weight 0 &> "./opt66b_32_256_b=64.log"

OMP_NUM_THREADS=40 numactl -m 0,1 -C 0-39 python -m flexgen.flex_opt --model facebook/opt-66b --path _DUMMY_ --percent 0 100 0 100 0 100 --gpu-batch-size 100 --num-gpu-batches 9 --prompt-len 32 --gen-len 32 --offload-dir ./ --cpu-cache-compute --pin-weight 0 &> "./opt66b_32_32_b=900.log"


