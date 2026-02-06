# QUICK_START

Download the preprocessed dataset
```bash
huggingface-cli download tari-tech/13832472466-who_are_you --local-dir ./hf_who_are_you --repo-type dataset
```

## CUDA Setup (recommended: system CUDA 12.4)
If you already have a system CUDA toolkit, prefer it for numba stability.

Set NVVM paths for numba (bash/zsh):
```bash
export CUDA_HOME=/usr/local/cuda-12.4
export NUMBA_CUDA_NVVM=$CUDA_HOME/nvvm/lib64/libnvvm.so
export NUMBA_CUDA_LIBDEVICE=$CUDA_HOME/nvvm/libdevice
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

Fish shell:
```fish
set -x CUDA_HOME /usr/local/cuda-12.4
set -x NUMBA_CUDA_NVVM $CUDA_HOME/nvvm/lib64/libnvvm.so
set -x NUMBA_CUDA_LIBDEVICE $CUDA_HOME/nvvm/libdevice
set -x LD_LIBRARY_PATH $CUDA_HOME/lib64 $CUDA_HOME/nvvm/lib64 $LD_LIBRARY_PATH
set -x PATH $CUDA_HOME/bin $PATH
```

## Conda CUDA Toolkit (optional)
If you only have the NVIDIA driver (no `nvcc`), install a CUDA toolkit inside conda.

```bash
bash scripts/create_env.sh
```

Optional: export `environment.yml`
```bash
bash scripts/create_env.sh --export-yml
```

Set NVVM paths for numba (bash/zsh):
```bash
export CUDA_HOME=$CONDA_PREFIX
export NUMBA_CUDA_NVVM=$CONDA_PREFIX/nvvm/lib64/libnvvm.so
export NUMBA_CUDA_LIBDEVICE=$CONDA_PREFIX/nvvm/libdevice
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/nvvm/lib64:$LD_LIBRARY_PATH
```

Fish shell:
```fish
set -x CUDA_HOME $CONDA_PREFIX
set -x NUMBA_CUDA_NVVM $CONDA_PREFIX/nvvm/lib64/libnvvm.so
set -x NUMBA_CUDA_LIBDEVICE $CONDA_PREFIX/nvvm/libdevice
set -x LD_LIBRARY_PATH $CONDA_PREFIX/lib:$CONDA_PREFIX/nvvm/lib64:$LD_LIBRARY_PATH
```

To force numba to use conda NVVM (recommended), create a project-level config:
```bash
cat > .numba_config.yaml <<'YAML'
CUDA_HOME: $CONDA_PREFIX
CUDA_NVVM: $CONDA_PREFIX/nvvm/lib64/libnvvm.so
CUDA_LIBDEVICE: $CONDA_PREFIX/nvvm/libdevice
YAML
```

Point numba to it (bash/zsh):
```bash
export NUMBA_CONFIG=$PWD/.numba_config.yaml
```

Fish shell:
```fish
set -x NUMBA_CONFIG $PWD/.numba_config.yaml
```

Verify:
```bash
python check_env.py
```
