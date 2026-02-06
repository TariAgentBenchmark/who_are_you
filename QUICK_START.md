# QUICK_START

Download the preprocessed dataset
```bash
huggingface-cli download tari-tech/13832472466-who_are_you --local-dir ./hf_who_are_you --repo-type dataset
```

## Conda CUDA Toolkit (for numba/NVVM)
If you only have the NVIDIA driver (no `nvcc`), install a CUDA toolkit inside conda.

```bash
conda create -n who-are-you python=3.12 -y
conda activate who-are-you

# Option A (preferred, NVIDIA channel)
conda install -c nvidia cuda-toolkit=12.4 -y

# Option B (fallback, conda-forge)
# conda install -c conda-forge cudatoolkit=12.4 cudatoolkit-dev=12.4 -y
```

Set NVVM paths for numba (bash/zsh):
```bash
export NUMBA_CUDA_NVVM=$CONDA_PREFIX/nvvm/lib64/libnvvm.so
export NUMBA_CUDA_LIBDEVICE=$CONDA_PREFIX/nvvm/libdevice
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/nvvm/lib64:$LD_LIBRARY_PATH
```

Fish shell:
```fish
set -x NUMBA_CUDA_NVVM $CONDA_PREFIX/nvvm/lib64/libnvvm.so
set -x NUMBA_CUDA_LIBDEVICE $CONDA_PREFIX/nvvm/libdevice
set -x LD_LIBRARY_PATH $CONDA_PREFIX/lib:$CONDA_PREFIX/nvvm/lib64:$LD_LIBRARY_PATH
```

Verify:
```bash
python check_env.py
```
