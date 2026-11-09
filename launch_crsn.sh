#!/bin/bash
set -e

cd /home/duoneural/crsn_ryzen
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PATH="/home/duoneural/srsn_env/bin:$PATH"

# Ensure RAM-disk datasets are hot in /dev/shm
if [ ! -f /dev/shm/train_golden.bin ]; then
    echo "[Setup] Copying train_golden.bin to /dev/shm..."
    cp /home/duoneural/srsn_ryzen/train_golden.bin /dev/shm/train_golden.bin
fi

if [ ! -f /dev/shm/val_golden.bin ]; then
    echo "[Setup] Copying val_golden.bin to /dev/shm..."
    cp /home/duoneural/srsn_ryzen/val_golden.bin /dev/shm/val_golden.bin
fi

echo "[Launch] Starting CRSN Pilot Pretraining Daemon (AMD Ryzen 5 3550H / Zen+ AVX2)..."
nohup taskset -c 0,1,2,3 /home/duoneural/srsn_env/bin/python -u /home/duoneural/crsn_ryzen/train_crsn.py \
  --train-bin /dev/shm/train_golden.bin \
  --val-bin /dev/shm/val_golden.bin \
  --data-dir /home/duoneural/srsn_ryzen \
  --total-optimizer-steps 2500 \
  --warmup-steps 25 \
  --batch-size 8 \
  --seq-len 256 \
  --grad-accum-steps 4 \
  --save-interval 25 \
  --eval-interval 25 > /home/duoneural/crsn_ryzen/train_crsn.log 2>&1 &

PID=$!
echo $PID > /home/duoneural/crsn_ryzen/train_crsn.pid
echo "[Launch] CRSN Daemon started with PID $PID on CPU cores 0,1,2,3."
