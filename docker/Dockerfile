FROM quay.io/ascend/cann:8.5.0-910b-ubuntu22.04-py3.11

# On 2026/02/14
# https://github.com/tile-ai/tilelang-ascend/commit/96c01413021dedc7b3a6648a4efba9dc365bcca1
# include the new PTO backend example: https://github.com/tile-ai/tilelang-ascend/pull/367
ARG COMMIT=96c01413021dedc7b3a6648a4efba9dc365bcca1

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
