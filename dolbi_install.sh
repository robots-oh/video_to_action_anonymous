#!/bin/bash

# 스크립트 실행 중 오류가 발생하면 즉시 중단합니다.
set -e

CONDA_ENV_NAME="vtob"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

# 활성화 성공 여부 확인
if [ "$CONDA_DEFAULT_ENV" != "${CONDA_ENV_NAME}" ]; then
    echo "오류: Conda 환경 '${CONDA_ENV_NAME}'을(를) 활성화하지 못했습니다."
    exit 1
fi
echo ">>> 현재 Conda 환경: ${CONDA_DEFAULT_ENV}"

export WORK_DIR="/workspace/video_to_action/"
cd "${WORK_DIR}"

COPPELIASIM_ROOT=/root/CoppeliaSim
if [ -z "$COPPELIASIM_ROOT" ]; then
    echo "오류: COPPELIASIM_ROOT 환경 변수가 설정되지 않았습니다."
    echo "예시: export COPPELIASIM_ROOT=/path/to/your/coppeliasim"
    exit 1
fi

# LD_LIBRARY_PATH를 설정합니다.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${COPPELIASIM_ROOT}"

echo ">>> PyRep 설치 시작..."
if [ ! -d "PyRep" ]; then
    echo "PyRep 폴더가 존재하지 않아 저장소를 복제합니다..."
    git clone https://github.com/markusgrotz/PyRep.git
else
    echo "PyRep 폴더가 이미 존재하므로 복제를 건너뜁니다."
fi
cd PyRep && \
pip install --no-cache-dir -r requirements.txt && \
pip install -e . --no-build-isolation && \
cd "${WORK_DIR}" # 작업 디렉토리로 복귀
echo ">>> PyRep 설치 완료."

echo ">>> YARR 설치 시작..."
if [ ! -d "YARR" ]; then
    echo "YARR 폴더가 존재하지 않아 저장소를 복제합니다..."
    git clone https://github.com/markusgrotz/YARR.git
else
    echo "YARR 폴더가 이미 존재하므로 복제를 건너뜁니다."
fi
cd YARR && \
pip install --no-cache-dir -r requirements.txt && \
pip install . -v && \
cd "${WORK_DIR}" # 작업 디렉토리로 복귀
echo ">>> YARR 설치 완료."

echo ">>> RLBench 설치 시작..."
if [ ! -d "RLBench" ]; then
    echo "RLBench 폴더가 존재하지 않아 저장소를 복제합니다..."
    git clone https://github.com/markusgrotz/RLBench.git
else
    echo "RLBench 폴더가 이미 존재하므로 복제를 건너뜁니다."
fi
cd ${WORK_DIR}/RLBench && \
pip install --no-cache-dir -r requirements.txt && \
pip install . -v && \
pip install -e . && \
cd "${WORK_DIR}" # 작업 디렉토리로 복귀
echo ">>> RLBench 설치 완료."


echo ">>> Pytorch3d 설치 시작..."
if [ ! -d "pytorch3d" ]; then
    echo "Pytorch3d 폴더가 존재하지 않아 저장소를 복제합니다..."
    git clone https://github.com/facebookresearch/pytorch3d.git
else
    echo "Pytorch3d 폴더가 이미 존재하므로 복제를 건너뜁니다."
fi
cd pytorch3d
pip install -e . --no-build-isolation
echo ">>> Pytorch3d 설치 완료..."


echo ">>> object_centric_diffusion 설치 시작..."
cd "${WORK_DIR}/src"

# 1. requirements 설치
pip install --no-cache-dir -r fp_requirements.txt

# [수정 1] nvdiffrast 설치 시 --no-build-isolation 옵션 추가
# 이렇게 해야 현재 환경에 설치된 PyTorch를 사용하여 CUDA 확장을 컴파일할 수 있습니다.
echo ">>> nvdiffrast 설치 중..."
python -m pip install --no-build-isolation --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

pip install --no-cache-dir -r dp3_requirements.txt

# Foundation Pose 빌드
cd "${WORK_DIR}/src/foundation_pose"

# [수정 2] pybind11 경로를 Python을 통해 직접 추출하여 정확도 향상
echo ">>> pybind11 경로 확인 및 빌드 시작..."
pip install pybind11  # pybind11이 확실히 설치되어 있도록 함

# Python 내부 명령어로 cmake 경로를 직접 가져옵니다.
PYBIND11_CMAKE_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

echo "Found pybind11 cmake dir at: ${PYBIND11_CMAKE_DIR}"

# 찾은 경로를 CMAKE_PREFIX_PATH로 주입하여 빌드 스크립트 실행
CMAKE_PREFIX_PATH="${PYBIND11_CMAKE_DIR}" bash build_all_conda.sh

echo ">>> object_centric_diffusion 설치 완료."


pip install --upgrade --force-reinstall wandb protobuf

echo "모든 설치가 성공적으로 완료되었습니다."