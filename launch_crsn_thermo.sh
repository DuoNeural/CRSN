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

mkdir -p /home/duoneural/srsn_ryzen/checkpoints_crsn_t

echo "[Launch] Starting CRSN-T Pretraining (Clifford Sheaf + 4 Laws of Thermodynamics)..."
nohup taskset -c 0,1,2,3 /home/duoneural/srsn_env/bin/python -u /home/duoneural/crsn_ryzen/train_crsn_thermo.py \
  --train-bin /dev/shm/train_golden.bin \
  --val-bin /dev/shm/val_golden.bin \
  --data-dir /home/duoneural/srsn_ryzen \
  --ckpt-dir /home/duoneural/srsn_ryzen/checkpoints_crsn_t \
  --resume /home/duoneural/srsn_ryzen/checkpoints_crsn_extension/crsn_step12500.pt \
  --total-optimizer-steps 25000 \
  --muon-lr 0.008 \
  --adam-lr 4e-4 \
  --warmup-steps 200 \
  --batch-size 8 \
  --seq-len 256 \
  --grad-accum-steps 4 \
  --save-interval 100 \
  --eval-interval 100 \
  --kappa 0.01 \
  --alpha-base 0.128 \
  --alpha-floor 0.05 \
  --t0 1.0 > /home/duoneural/crsn_ryzen/train_crsn_thermo.log 2>&1 &

PID=$!
echo $PID > /home/duoneural/crsn_ryzen/train_crsn_thermo.pid
echo "[Launch] CRSN-T Daemon started with PID $PID on CPU cores 0,1,2,3."
