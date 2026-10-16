FROM clagemann/hydrogym-nvhpc-26.1_cuda-12.9_hopper_blackwell@sha256:5e8deb0b0efce2e2fb47934f0ad7dd2c2b4a430d2b9b72d47bfc2dd3aff39b03
USER root
RUN source /etc/profile && module purge && module load MAIA/1.0-NVHPC-26.1 && \
    /home/easybuild/venvs/hydrogym_gpu/bin/python -m pip install --no-cache-dir torchrl==0.13.3 tensordict==0.13.0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
