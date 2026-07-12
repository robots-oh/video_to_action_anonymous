FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    libdbus-1-3 \
    libfontconfig1 \
    libfreetype6 \
    libgl1-mesa-glx \
    libegl1-mesa \
    libglib2.0-0 \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-glx0 \
    libxkbcommon-x11-0 \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libglfw3-dev \
    libxxf86vm-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}
RUN conda init bash
RUN conda update -n base -c defaults conda -y
SHELL ["/bin/bash", "-c"]
RUN conda create -n vtob python=3.10 -y
SHELL ["conda", "run", "-n", "vtob", "/bin/bash", "-c"]
RUN conda install -y \
    conda-forge::eigen=3.4.0 \
    cmake \
    conda-forge::boost \
    && conda clean -afy

RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

ENV COPPELIASIM_ROOT=/root/CoppeliaSim
RUN wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
    mkdir -p $COPPELIASIM_ROOT && \
    tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1 && \
    rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

RUN mkdir -p /opt/conda/envs/vtob/etc/conda/activate.d && \
    echo '#!/bin/sh' >> /opt/conda/envs/vtob/etc/conda/activate.d/coppeliasim_vars.sh && \
    echo 'export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}' >> /opt/conda/envs/vtob/etc/conda/activate.d/coppeliasim_vars.sh && \
    echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}' >> /opt/conda/envs/vtob/etc/conda/activate.d/coppeliasim_vars.sh && \
    echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}' >> /opt/conda/envs/vtob/etc/conda/activate.d/coppeliasim_vars.sh
RUN mkdir -p /opt/conda/envs/vtob/etc/conda/deactivate.d && \
    echo '#!/bin/sh' >> /opt/conda/envs/vtob/etc/conda/deactivate.d/coppeliasim_vars.sh && \
    echo 'export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}' >> /opt/conda/envs/vtob/etc/conda/deactivate.d/coppeliasim_vars.sh && \
    echo 'unset OLD_LD_LIBRARY_PATH' >> /opt/conda/envs/vtob/etc/conda/deactivate.d/coppeliasim_vars.sh && \
    echo 'unset QT_QPA_PLATFORM_PLUGIN_PATH' >> /opt/conda/envs/vtob/etc/conda/deactivate.d/coppeliasim_vars.sh

ENV WORK_DIR=/workspace
COPY dolbi_install.sh /workspace/dolbi_install.sh
RUN chmod +x /workspace/dolbi_install.sh

WORKDIR ${WORK_DIR}
CMD ["conda", "run", "-n", "vtob", "env", "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so", "/bin/bash"]
