#!/bin/bash
# ============================================================================
#  Single-GPU optimisation and multi-GPU A^H A on an H100 node (Slurm).
#
#  Usage (from the repository root on the login node), e.g.:
#      PART=test  sbatch --gres=gpu:h100:6 benchmarks/hpc/run_mgpu_h100.sh
#      PART=bench sbatch --gres=gpu:h100:6 benchmarks/hpc/run_mgpu_h100.sh
#  (add -w esplhpc-cp087 / --exclude=... to pick the node)
#
#  PART=test   correctness: tests/check_reference.py and check_bd_kernels.py on
#              one GPU, tests/check_mgpu.py on every allocated GPU
#  PART=single bench_aha_backend.py on one GPU: generic kernels one coil at a
#              time (MR_AHA_CUDA_GENERIC_BD=1, the previous operator), then the
#              specialised kernels with coil groups 1, 2, 4 and 6, each with
#              the per-stage profile (MR_AHA_CUDA_PROFILE=1)
#  PART=bench  benchmarks/bench_mgpu.py on 1..N GPUs, split off/auto
#  PART=split  the same with $COILS coils (default 8 and 7, which do not divide
#              the GPU counts), split off/on/auto: remainder-split timing and
#              the cost model's choice
#  PART=matlab builds the MEX files inside the Singularity MATLAB image (its
#              MATLAB R2025b and CUDA 12.9, into build_h100_matlab) and runs
#              tests/check_mgpu_matlab.m on 2 and all allocated GPUs, plus
#              tests/check_reference_matlab.m on one GPU
#
#  The library is built into build_h100_mgpu_<PART> (sm_90), one per part so
#  that the parts can run at the same time. Node esplhpc-cp087 fails with
#  cudaErrorSystemNotReady (NVSwitch fabric manager down, 2026-11-06): exclude it.
# ============================================================================
#SBATCH --job-name=mr_aha_mgpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=benchmarks/hpc/logs/%x_%j.out

set -uo pipefail

ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$ROOT"
PART=${PART:-test}
ENV_PY=${ENV_PY:-$HOME/.conda/envs/mr_aha/bin/python}
BUILD=build_h100_mgpu_$PART

module load cuda12.9/toolkit/12.9.1
export PATH=$(dirname "$ENV_PY"):$PATH
export MR_AHA_LIBRARY=$ROOT/$BUILD/lib/libmr_aha_c.so

echo "== $PART on $(hostname) $(date)"

if [[ $PART == matlab ]]; then
    MATLAB_IMG=${MATLAB_IMG:-/common/lidxxlab/containers/matlab_biri_v1.6.sif}
    module load singularity-apptainer 2>/dev/null || true
    NGPU=$(nvidia-smi -L | wc -l)
    sing() {
        singularity exec --nv --env MLM_LICENSE_FILE=27000@cslxlms \
            --bind "$ROOT":"$ROOT" "$MATLAB_IMG" bash -c "$1"
    }
    sing "export PATH=/usr/local/cuda-12.9/bin:\$PATH; cd '$ROOT' && \
        $(dirname "$ENV_PY")/cmake -S . -B build_h100_matlab -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CUDA_ARCHITECTURES=90 -DMR_AHA_BUILD_MATLAB=ON \
            -DMR_AHA_BUILD_PYTHON=OFF -DMR_AHA_BUILD_EXAMPLES=OFF \
            -DMatlab_ROOT_DIR=/opt/matlab/R2025b > /dev/null && \
        $(dirname "$ENV_PY")/cmake --build build_h100_matlab -j ${SLURM_CPUS_PER_TASK:-8} \
            --target AhA_Toeplitz_mex calc_toep_mex calc_mask_mex mr_aha_op_mex" || exit 1
    ADDP="addpath('$ROOT/matlab', '$ROOT/build_h100_matlab/matlab', '$ROOT/tests')"
    sing "matlab -batch \"$ADDP; check_mgpu_matlab('device_sets', {[1 2], 1:$NGPU}, \
        'split_modes', {'off', 'on'})\"" || echo "check_mgpu_matlab FAILED"
    sing "CUDA_VISIBLE_DEVICES=0 matlab -batch \"$ADDP; cd('$ROOT'); \
        check_reference_matlab\"" || echo "check_reference_matlab FAILED"
    echo "== done $(date)"
    exit 0
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
nvidia-smi topo -m

cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=90 -DMR_AHA_BUILD_PYTHON=ON \
      -DMR_AHA_BUILD_EXAMPLES=OFF -DMR_AHA_BUILD_MATLAB=OFF > /dev/null || exit 1
cmake --build "$BUILD" -j "${SLURM_CPUS_PER_TASK:-8}" --target mr_aha_c || exit 1

NGPU=$(nvidia-smi -L | wc -l)
ALL=$(seq -s ' ' 0 $((NGPU - 1)))

if [[ $PART == test ]]; then
    CUDA_VISIBLE_DEVICES=0 "$ENV_PY" -u tests/check_reference.py || echo "check_reference FAILED"
    CUDA_VISIBLE_DEVICES=0 "$ENV_PY" -u tests/check_bd_kernels.py || echo "check_bd_kernels FAILED"
    "$ENV_PY" -u tests/check_mgpu.py --devices $ALL || echo "check_mgpu FAILED"
elif [[ $PART == single ]]; then
    export CUDA_VISIBLE_DEVICES=0
    run() {
        echo "== $*"
        env "$@" MR_AHA_CUDA_PROFILE=1 "$ENV_PY" -u benchmarks/bench_aha_backend.py \
            --reps 5 2>&1 | grep -E "AhA median|skipped|CUDA profile|block-diag|FFT|pad|crop|total GPU" \
            | awk '!seen[$0]++'
    }
    run MR_AHA_CUDA_GENERIC_BD=1
    cp benchmarks/results/$(hostname -s)/aha_cuda.json \
       benchmarks/results/$(hostname -s)/aha_cuda_generic.json 2>/dev/null
    for g in 1 2 6 4; do run MR_AHA_CUDA_COIL_GROUP=$g; done
elif [[ $PART == bench ]]; then
    "$ENV_PY" -u benchmarks/bench_mgpu.py --cases stack 3d 2d \
        --gpus $(seq -s ' ' 2 "$NGPU") --split off auto --reps 10 \
        || echo "bench_mgpu FAILED"
elif [[ $PART == split ]]; then
    # Remainder split: coil counts that do not divide the GPU count.
    for c in ${COILS:-8 7}; do
        "$ENV_PY" -u benchmarks/bench_mgpu.py --cases stack 3d --no-masked \
            --gpus $(seq -s ' ' 2 "$NGPU") --split off on auto --reps 10 \
            --coils $c --out mgpu_cuda_split_c$c.json || echo "bench_mgpu FAILED"
    done
fi
echo "== done $(date)"
