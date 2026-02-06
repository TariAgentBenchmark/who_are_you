# QUICK_START

## Conda CUDA Toolkit (recommended for env isolation)
If you only have the NVIDIA driver (no `nvcc`), install a CUDA toolkit inside conda.

```bash
bash scripts/create_env.sh
```

Optional: export `environment.yml`
```bash
bash scripts/create_env.sh --export-yml
```

### Bash/Zsh (conda)
```bash
export CUDA_HOME=$CONDA_PREFIX
export NUMBA_CUDA_NVVM=$CONDA_PREFIX/nvvm/lib64/libnvvm.so
export NUMBA_CUDA_LIBDEVICE=$CONDA_PREFIX/nvvm/libdevice
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/nvvm/lib64:$LD_LIBRARY_PATH

cat > .numba_config.yaml <<'YAML'
CUDA_HOME: $CONDA_PREFIX
CUDA_NVVM: $CONDA_PREFIX/nvvm/lib64/libnvvm.so
CUDA_LIBDEVICE: $CONDA_PREFIX/nvvm/libdevice
YAML

export NUMBA_CONFIG=$PWD/.numba_config.yaml
```

### Fish (conda)
```fish
set -x CUDA_HOME $CONDA_PREFIX
set -x NUMBA_CUDA_NVVM $CONDA_PREFIX/nvvm/lib64/libnvvm.so
set -x NUMBA_CUDA_LIBDEVICE $CONDA_PREFIX/nvvm/libdevice
set -x LD_LIBRARY_PATH $CONDA_PREFIX/lib:$CONDA_PREFIX/nvvm/lib64:$LD_LIBRARY_PATH

cat > .numba_config.yaml <<'YAML'
CUDA_HOME: $CONDA_PREFIX
CUDA_NVVM: $CONDA_PREFIX/nvvm/lib64/libnvvm.so
CUDA_LIBDEVICE: $CONDA_PREFIX/nvvm/libdevice
YAML

set -x NUMBA_CONFIG $PWD/.numba_config.yaml
```

Verify:
```bash
python check_env.py
```

## System CUDA Setup (alternative: system CUDA 12.4)
If you already have a system CUDA toolkit, you can use it instead of conda.

### Bash/Zsh (system)
```bash
export CUDA_HOME=/usr/local/cuda-12.4
export NUMBA_CUDA_NVVM=$CUDA_HOME/nvvm/lib64/libnvvm.so
export NUMBA_CUDA_LIBDEVICE=$CUDA_HOME/nvvm/libdevice
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/nvvm/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

### Fish (system)
```fish
set -x CUDA_HOME /usr/local/cuda-12.4
set -x NUMBA_CUDA_NVVM $CUDA_HOME/nvvm/lib64/libnvvm.so
set -x NUMBA_CUDA_LIBDEVICE $CUDA_HOME/nvvm/libdevice
set -x LD_LIBRARY_PATH $CUDA_HOME/lib64 $CUDA_HOME/nvvm/lib64 $LD_LIBRARY_PATH
set -x PATH $CUDA_HOME/bin $PATH
```


## Datasets
Download the preprocessed dataset
```bash
huggingface-cli download tari-tech/13832472466-who_are_you --local-dir ./hf_who_are_you --repo-type dataset
```
