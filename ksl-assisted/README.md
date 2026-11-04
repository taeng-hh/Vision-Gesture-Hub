# 키포인트 주판정 + CNN 보조 수어 인식 사본

기존 Vision-Gesture-Hub와 분리된 독립 실행 폴더입니다. Python 3.11·OpenCV·MediaPipe·PyTorch를 사용하고 단어 23개·문장 7개를 인식합니다.

## 동작 방식

- 기본 모드는 assisted입니다. 키포인트 판정이 통과하면 CNN 없이 바로 출력합니다.
- 키포인트가 애매할 때만 CNN을 실행합니다. 같은 후보를 가리킬 때만 키포인트 80%·CNN 20%로 재확인합니다.
- 후보가 다르거나 CNN이 실패하면 보류합니다. CNN만으로 결과를 확정하지 않습니다.
- 촬영 중 얻은 관절을 재사용해 입력 종료 후 지연을 줄입니다. 추적이 부족하면 원본 재분석으로 복구합니다.
- 손·상체 스켈레톤, 모델별 후보, 분석 시간과 RGB 입력 ROI를 표시합니다.

## Windows 설치와 실행

이 폴더에서 setup.cmd를 실행한 뒤 run.cmd를 실행합니다. Python 3.11과 인터넷 연결이 필요합니다.
이미 필요한 패키지가 있는 환경이라면 아래처럼 직접 실행할 수 있습니다.

    python scripts/setup_assets.py
    python main.py --mode assisted

수동 설치:

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements-fusion.txt
    .venv\Scripts\python.exe scripts/setup_assets.py
    .venv\Scripts\python.exe main.py --mode assisted

MediaPipe 손·몸 모델과 기본 CNN 가중치 약 24MB는 초기 설정에서 공식 URL로 받고 SHA256을 검증합니다.
수어용 키포인트 모델과 세 시드 RGB GRU 가중치는 이 폴더에 포함되어 있습니다.
원본 AI Hub 영상·학습 캐시·개인 경로·웹캠 촬영 파일은 포함하지 않습니다.

## 조작

웹캠 창을 클릭한 후 Space로 입력 시작 → 한 단어 또는 문장 수행 → Space로 완료합니다.
완료 후 분석하는 방식이며 연속 수어 자동 분할 기능은 아닙니다.

- 1: 키포인트 단독 판정
- 2: 영상 진단 모드
- 3: 키포인트 주판정 + CNN 보조
- C: 초기화 / Backspace: 마지막 단어 삭제 / Q: 종료

같은 최근 동작을 다른 모드에서 비교할 수 있습니다. 아직 계산하지 않은 모델은 선택할 때 추가 분석합니다.
카메라를 열지 못하면 run.cmd --camera 1 또는 python main.py --camera 1을 시도하세요.
원본 재분석 비교는 --no-live-reuse, 과거 가중 평균 비교는 --mode fusion으로 실행합니다.

## 확인한 범위와 한계

관련 로직 테스트 77개가 통과했습니다. 실제 사용자 웹캠에서는 RGB 모델이 약하다는 피드백이 있어 보조 용도로 제한했습니다.
AI Hub REAL09 검증에서 키포인트 단독은 22/30, RGB 단독은 30/30이었으나, 이 수치는 웹캠 정확도가 아닙니다.
새 보조 정책으로 웹캠 정확도가 개선됐는지는 아직 측정하지 않았습니다. 점수는 보정된 정답 확률이 아닙니다.
이미 취득한 같은 키포인트를 재사용한 계산 시험은 촬영·추적·화면 표시·카메라 I/O 시간을 제외한 결과입니다.

테스트:

    python -m unittest tests.test_fusion tests.test_fusion_preview tests.test_fusion_speed tests.test_live_sign tests.test_assisted_policy -v

Linux에서 화면 관련 테스트에는 한글 폰트가 필요합니다. KSL_FONT에 한글 TTF/TTC 경로를 지정할 수 있습니다.
직접 웹캠 사용에는 화면 표시가 가능한 OpenCV 환경이 필요합니다.

## 모델 출처

원본 코드와 키포인트 모델: https://github.com/taeng-hh/Vision-Gesture-Hub
키포인트 가중치 커밋: 609e3cfe82ca91b4b93d8e1fc2d7eb72accfb192.
원본 설명상 REAL01 30개 동작을 증강한 750개 입력으로 학습한 모델입니다.
RGB는 REAL01~08 학습·REAL09 검증으로 선택한 MobileNetV3 Small + GRU 세 시드 앙상블입니다.
REAL10은 이전 RGB 평가에 이미 사용됐으므로 새 모델의 미사용 최종 테스트라고 주장하지 않습니다.
RGB 체크포인트는 학습 가중치·정규화·필수 설정만 보존했으며 텐서 값은 원본과 동일합니다.
