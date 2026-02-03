FROM quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11

# On 2026/01/31
# https://github.com/tile-ai/tilelang-ascend/commit/d2405b5975edaa36166b702d2ef8c1952b7b7337
# include the new PTO backend example: https://github.com/tile-ai/tilelang-ascend/pull/367
ARG COMMIT=d2405b5975edaa36166b702d2ef8c1952b7b7337

WORKDIR /sources/
RUN git clone --recursive https://github.com/tile-ai/tilelang-ascend.git
WORKDIR /sources/tilelang-ascend
RUN git checkout $COMMIT

# https://pypi.org/project/torch-npu/#history
RUN pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir torch-npu==2.9.0 \
    && pip install --no-cache-dir pyyaml
RUN bash install_ascend.sh

RUN echo "source /sources/tilelang-ascend/set_env.sh" >> ~/.bashrc

WORKDIR /sources/tilelang-ascend/examples
