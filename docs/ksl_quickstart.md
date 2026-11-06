# 한국수어 단어 인식 및 문장 조합 (첫 버전)

이 버전은 양손 21개 관절 시계열을 24단계로 보간하고 손 모양, 손목 위치,
손 유무를 비교하는 템플릿 기반 분류기입니다. 외부 딥러닝 학습 프레임워크 없이
작은 어휘의 기준 성능을 측정합니다. 저장 모델은 학습 영상에서 계산한 특징을 담습니다.
수어 학습 데이터나 학습 완료된 한국수어 모델은 포함되어 있지 않습니다.

## 1. 설치

저장소 루트에서 Python 3.11 가상환경을 사용합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_model.py
```

## 2. 실제 표현과 라벨 확정

`config/ksl_rules.json`의 WATER, REQUEST 등은 의미 라벨 예시입니다.
규칙은 검증된 한국수어 어순이나 사전 자료가 아닙니다. 한국수어 사전/사용자 검토로
실제 단어와 순서를 확인한 뒤 수정하세요. 서로 구분하기 쉬운 단어 2~3개부터 시작하세요.
학습과 실행에서 같은 조명, 카메라 방향, 상반신 구도를 사용하세요.

## 3. 한 단어를 하나의 동작 구간으로 수집

```powershell
.\.venv\Scripts\python.exe -m training.collect_sequences --label WATER --signer s01
.\.venv\Scripts\python.exe -m training.collect_sequences --label REQUEST --signer s01
```

Space: 녹화 시작 -> 단어 한 번 수행 -> Space: 저장. Q: 종료 (미저장 구간은 버림).
동작 앞뒤에 긴 대기 시간을 넣지 마세요. 사진/연속 프레임 수가 아닌 **독립적인 동작 반복 횟수**를 세세요.
학습자별 단어당 15~30회부터 실험하고 다른 촬영자의 데이터도 수집하세요.
학습용 단어별 최소 3개 클립이 필요합니다. 별도의 촬영자 `s03`으로 모든 단어를
촬영해 검증용으로 남겨 두세요. 동일 인물에게 다른 ID를 붙이면 평가가 무효가 됩니다.
데이터는 `data/ksl_clips/*.json`에 저장됩니다 (영상 자체는 저장하지 않음).

### 이미 있는 단어 영상 / AI Hub 원천 영상 가져오기

영상 사용 조건을 확인하고, 단어 ID와 한국어 대응표를 먼저 확보하세요.
CSV를 만듭니다 (아래 파일 경로와 구간은 형식 예시이며 실제 자료가 아닙니다).

```csv
video,label,signer,start_s,end_s
videos/water_s01.mp4,WATER,s01,0,2.5
videos/request_s01.mp4,REQUEST,s01,0,3
```

```powershell
.\.venv\Scripts\python.exe -m training.import_videos data/manifest.csv
```

현재 가져오기는 범용 CSV 매니페스트 방식입니다. AI Hub 고유 JSON의 자동 해석이나
단어표 자동 다운로드는 포함하지 않습니다. 문장 영상은 정답 단어 구간을 지정해야 합니다.
동일 촬영의 다른 각도는 같은 signer로 지정하고 검증 분리에 섞지 마세요.
기본적으로 영상도 웹캠과 같은 거울 방향으로 뒤집습니다. 이미 거울 영상이면
`--already-mirrored`를 지정하세요. AI Hub 제공 키포인트는 직접 섞지 않고 같은
MediaPipe 검출기로 다시 추출합니다.

## 4. 학습 및 촬영자 분리 평가

```powershell
.\.venv\Scripts\python.exe -m training.train_words --validation-signer s03
```

`s03` 전체를 검증용으로 제외합니다. 모델은 `models/ksl_words.npz`, 평가 결과는
`models/ksl_words.metrics.json`에 저장합니다. 검증자에게 모든 학습 단어가 있어야 합니다.
학습 데이터만으로 거리 임계값을 추정하며 검증 데이터로 임계값을 조정하지 않습니다.
정확도는 인식 거절도 오답으로 포함하고 혼동 행렬을 함께 기록합니다.
이 점수는 실제 수어 정확도를 보장하지 않습니다. 미등록 동작/대기/전환 동작을
별도 시험해 거절 성능도 확인하세요. 유사한 단어는 특징 추가나 다른 모델이 필요할 수 있습니다.

## 5. 실행

```powershell
.\.venv\Scripts\python.exe main.py --sign-model models/ksl_words.npz
```

- `1`: 수어 모드. 단어를 수행하고 **양손을 화면 밖으로 0.35초 이상** 빼면 한 단어를 확정합니다.
- `Enter`: 현재 단어 조합을 규칙과 대조해 문장 확정.
- `Backspace`: 마지막 단어 삭제. `C`: 단어/문장 초기화.
- 같은 단어도 별도의 구간으로 수행하면 반복 입력할 수 있습니다.
- 모델이 없으면 학습 안내만 표시합니다. 거리/클래스 간 차이 기준을 통과하지 못하면 인식을 거절합니다.
- 설정에 없는 단어 조합은 문장을 추측하지 않고 미지원으로 표시합니다.
- 모드 전환 시 진행 중인 동작은 버리며 이미 확정된 단어는 유지합니다.

한글 자막에는 Windows 맑은 고딕을 사용합니다. 다른 환경에서는 `KSL_FONT` 환경변수로
한글 TTF/TTC 경로를 지정하세요.

## 현재 범위

손 모양과 영상 기준 손목 위치/움직임을 사용합니다. 몸/얼굴 기준 위치, YOLO, HSV,
optical flow, 자연스러운 연속 수어의 자동 단어 경계, 자유 문장 번역은 아직 포함하지 않습니다.
손을 빼는 경계 동작은 시연을 위한 명시적 입력 규칙이며 자연 수어의 단어 경계가 아닙니다.
프레임별 보간 기반이라 긴 손 가림이나 검출 누락, 촬영 구도 변화에 약할 수 있습니다.
`training/collect_samples.py`의 기존 프레임 CSV는 이 모델의 학습 입력이 아닙니다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

자동 테스트는 합성 관절로 저장/복원, 거절, 구간 분리, 문장 조합, 모드 연결을 검증합니다.
실제 수어 정확도, 웹캠 FPS, 카메라 동작은 실제 데이터와 장치로 별도 측정해야 합니다.
