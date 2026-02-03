
```bash
sudo docker build -t tilelang-ascend:8.5.0 .

sudo docker run --rm -it --ipc=host --privileged \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v $HOME:/mount_home \
    -w /mount_home \
    tilelang-ascend:8.5.0 /bin/bash

cd /sources/tilelang-ascend/examples/
python gemm/example_gemm.py
python sparse_flash_attention/example_sparse_flash_attn_gqa_pto_developer.py
```

Guide: https://github.com/tile-ai/tilelang-ascend?tab=readme-ov-file#installation

Base images:
- https://github.com/Ascend/cann-container-image/blob/main/cann/8.5.0-910b-ubuntu22.04-py3.11/Dockerfile (`quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11`)
- https://github.com/Ascend/cann-container-image/blb/main/manylinux/8.5.0-910b-manylinux_2_28-py3.11/Dockerfile (no quay.io?)
