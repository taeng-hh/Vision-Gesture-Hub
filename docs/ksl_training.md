# 23개 단어 + 7개 문장 공통 학습 파이프라인

## 학습 대상과 팀 작업 경계

`config/ksl_classes.json`은 팀의 최종 엑셀에 적힌 WORD 23개, SEN 7개를 보존합니다.
WORD와 SEN의 ID가 모델의 고정 클래스 키이며 한국어 출력은 별도 메타데이터입니다.
게임 동작은 엑셀의 매핑을 metadata로만 제공합니다. 게임 구현이나 OS 입력 실행은 하지 않습니다.
단어 인식 결과와 문장 전체 인식 결과는 동일 GRU 분류기에서 경쟁합니다.
WORD1290 감사와 SEN0355 감사합니다는 동작이 겹칠 가능성이 있으므로 반드시
혼동 행렬을 확인하세요. 길이만으로 word/sentence를 구분하지 않습니다.

현재 데이터는 REAL01 단어 키포인트뿐입니다. 문장 데이터가 없으면 30클래스 전체 학습은
실패하도록 설계했습니다. 누락 클래스를 가짜 샘플로 채우거나 다른 단어에 대응시키지 않습니다.
현재 8단어 웹캠 기능은 별도 기준 모델입니다. 새 GRU 파일을 기존 NPZ 경로에 넣지 마세요.

## 설치 및 데이터 준비

Python 3.11 CMD, 저장소 루트에서:

```cmd
.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.venv\Scripts\python.exe -m ksl.prepare --source "C:\Users\01051\OneDrive\Desktop\기컴비\download" --output data\ksl30
```

다운로드 폴더에서 WORD/SEN의 morpheme 및 keypoint ZIP 또는 `.zip.part0`를 찾습니다.
분할 파일을 숫자 오프셋 순서로 가상 연결하므로 전체를 합치거나 압축 해제할 필요가 없습니다.
기본은 정면 F만 사용합니다. `--views F,L,R`로 바꿔도 같은 수행의 다른 시점은 같은
recording_id로 묶습니다. 입력 ZIP에 프레임 이미지는 없어도 됩니다.
프레임 번호/30fps와 morpheme 구간으로 자르며 문장은 첫 Gloss 시작부터 마지막 Gloss 끝까지
연결 동작을 포함합니다. 다른 FPS 데이터면 `--fps`로 실제 FPS를 지정하세요.
라벨이 registry의 expected_gloss와 다르면 건너뛰고 manifest의 label_conflicts에 기록합니다.
좌표/어깨가 유효하지 않은 구간도 invalid_clips로 기록합니다. 원본은 수정하지 않습니다.
동일 output에 덮어쓰지 않습니다. 새 데이터가 추가되면 새 output 폴더로 전체 준비를 재실행하세요.

## 학습

여러 촬영자/그룹이 있고 모든 클래스가 각 분할에 있을 때:

```cmd
.venv\Scripts\python.exe -m ksl.train --data data\ksl30 --validation-group REAL02 --output models\ksl30 --epochs 80
```

REAL02 전체를 검증에 남기며 같은 촬영 구간의 다른 각도를 학습/검증에 나누지 않습니다.
REAL은 데이터의 제공 코드입니다. 그룹이 실제로 다른 사람인지 데이터 가이드/촬영 기록으로 확인하세요.
다른 클래스가 누락된 그룹 분할은 허용하지 않습니다. 최종 성능 발표용 테스트 그룹은 별도 보관하세요.

아직 단일 그룹/단어 자료만 있을 때 파이프라인 동작만 확인하려면:

```cmd
.venv\Scripts\python.exe -m ksl.train --data data\ksl30 --smoke-only --allow-partial --output models\ksl30_smoke --epochs 5
```

이 명령은 명시적인 부분 학습입니다. validation_accuracy는 null이며 training_replay_accuracy는
학습 자료 재입력 성능입니다. 새 사용자 정확도로 보고하면 안 됩니다.
모델은 32프레임 × 90차원 입력, 2층 GRU(64 hidden), 분류층을 사용합니다.
표준화 통계는 학습 자료만으로 계산합니다. 시드·클래스 순서·분할 출처를 함께 저장합니다.
GPU가 필요하지 않으며 CPU 기본 설정입니다.

출력 폴더에는 model.pt, classes.json, metrics.json, provenance.json이 생성됩니다.
모델/원본 데이터는 Git에 올리지 말고 이용 조건에 맞는 별도 전달 경로로 공유하세요.

## 예측과 게임 연결

```cmd
.venv\Scripts\python.exe -m ksl.predict --model models\ksl30 --clip data\ksl30\clips\NIA_SL_WORD1577_REAL01_F.npz
```

공통 API:

```python
from ksl.predict import Recognizer
recognizer = Recognizer("models/ksl30")
event = recognizer.predict(frames)
# accepted, class_id, kind, text, game_action, candidate_id, score, margin
```

게임 팀은 accepted가 true이고 활성 게임 모드일 때만 game_action을 소비하세요.
중복 프레임에서 액션이 반복되지 않도록 별도 확정 이벤트/중립 복귀/쿨다운 처리가 필요합니다.
sentence이면 해당 문장을 출력하고 word이면 기존 문장 조합기로 넘길 수 있습니다.
등록 문장/단어를 한 구간씩 Space로 완성하는 입력 방식에 연결하는 API이며,
자유 연속 수어의 자동 문장 경계를 해결한 것은 아닙니다.
softmax 점수는 실제 정확도가 아닙니다. 기본 거절 임계값(score .75 / margin .15)은
보정되지 않은 초기값이므로 미등록 동작 검증 자료로 따로 평가해야 합니다.

## 영상 담당자와 맞출 공통 데이터 형식

영상에서 관절을 추출하는 코드는 아래 계약을 지키면 이 학습기를 그대로 사용할 수 있습니다.
RGB 영상 자체를 CNN으로 학습하려면 별도 입력 모델이 필요합니다.

- feature_spec: `ksl-xyc67-unmirrored-v1`
- 한 클립의 frames: `(T,67,3)` float32, T>=4, 세 번째 값은 **신뢰도 0~1**. Z좌표 아님.
- 0~20: 해부학적 왼손 21점, 21~41: 오른손 21점. 손 랜드마크 순서는 MediaPipe/OpenPose 손 순서.
- 42~66: OpenPose BODY_25. 필수 어깨는 44번(몸 2, 오른쪽), 47번(몸 5, 왼쪽).
- XY는 비반전 원본의 같은 픽셀 좌표계. 정규화 XY는 가로/세로 크기를 각각 곱해 픽셀로 바꿉니다.
- 없는 점은 confidence=0. 손이 없는 프레임과 어깨가 없는 프레임은 다릅니다. 어깨 누락은 준비 단계에서 거절합니다.
- 단어는 해당 단어 구간, 문장은 문장 전체 구간을 한 클립으로 저장합니다.
- performer_group은 동일 사람의 모든 영상을 같은 코드로 유지합니다.
- recording_id는 같은 원본 수행의 모든 카메라 각도에 같은 값입니다. 프레임/각도별로 그룹 ID를 새로 만들지 마세요.
- extractor에는 AIHub/OpenPose/MediaPipe 등 출처를 남깁니다. 서로 다른 추출기는 형식이 같아도 분포가 달라 실제 평가가 필요합니다.

영상 담당자는 `ksl.data.write_clip(...)`으로 저장하고 반환된 행들을
`{"feature_spec":"ksl-xyc67-unmirrored-v1","clips":[...]}` manifest.json에 기록하면 됩니다.
실제 영상 추출기는 이 변경에 포함되지 않습니다.

출처: 팀 제공 `KSL_텀프로젝트_최종_단어23_문장7_총30클래스.xlsx`의 최종 학습 목록.
AI Hub 데이터셋: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103
