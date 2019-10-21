# Uninstall jax that was intalled by conda.
conda uninstall jaxlib jax

PYTHON_VERSION=cp37  # alternatives: cp27, cp35, cp36, cp37
CUDA_VERSION=cuda101  # alternatives: cuda90, cuda92, cuda100, cuda101
PLATFORM=linux_x86_64  # alternatives: linux_x86_64
BASE_URL='https://storage.googleapis.com/jax-releases'
pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.30-$PYTHON_VERSION-none-$PLATFORM.whl
pip install --upgrade jax
