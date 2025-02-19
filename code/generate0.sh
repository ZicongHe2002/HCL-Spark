#!/bin/bash

# 循环从8到30
for exit_layer in {2..38..2}
do
    # 输出当前的exit_layer值
    echo "Running with exit_layer=$exit_layer"

    # 运行 Python 脚本，传入动态的exit_layer值
    srun --account=bcyy-delta-gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G \
         torchrun --master_port=29505 gen0.py \
         --model facebook/layerskip-llama2-13B \
         --sample True \
         --max_steps 50 \
         --generation_strategy self_speculative \
         --exit_layer $exit_layer \
         --num_speculations 6 \
         # --temperature 1.0

    # 输出当前任务的结束时间
    echo "Job for exit_layer=$exit_layer finished at: $(date)"
done

# 脚本执行完后的回调
echo "All jobs finished at: $(date)"

