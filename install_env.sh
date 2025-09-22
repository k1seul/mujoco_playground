#!/bin/bash

# 환경 이름 설정 (선택 사항, 필요에 따라 수정)
ENV_NAME="mujoco"

# .yml 파일로 환경 생성 또는 업데이트
echo "Conda 환경 '${ENV_NAME}'을(를) 생성 중..."
conda env create -f environment.yml --name ${ENV_NAME}

# 환경이 성공적으로 생성되었는지 확인
if [ $? -eq 0 ]; then
    echo "Conda 환경 '${ENV_NAME}'이(가) 성공적으로 생성되었습니다."
    echo "활성화하려면 'conda activate ${ENV_NAME}' 명령어를 사용하세요."
else
    echo "Conda 환경 생성에 실패했습니다."
    exit 1
fi
