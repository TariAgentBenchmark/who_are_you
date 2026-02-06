# QUICK_START
由于平台抽成高，如果您对这次服务满意，后续有任何需求，可以直接加我wx：15637899910，直接对接防止中间商赚差价
由于平台有规则，您加或者不加都请不要在微信讨论这个事情。感谢！！！！

Dataset placement:
```
datasets/
  TIMIT/
  generated_TIMIT/
```

## Conda CUDA Toolkit (recommended for env isolation)
If you only have the NVIDIA driver (no `nvcc`), install a CUDA toolkit inside conda.

```bash
# Create conda env
bash scripts/create_env.sh
# Activate conda env
conda activate who-are-you
# Install pinned python packages. Don't use the fucking pip.
uv sync
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

run
```bash
uv run --python 3.12 --refresh python main.py --precision_target 0.999 --recall_target 0.995
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
