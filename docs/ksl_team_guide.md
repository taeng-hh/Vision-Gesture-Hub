# 팀원용: 한국수어 30클래스 설치부터 실행까지

단어 23개와 문장 7개를 인식하는 실험용 기능입니다. 입력 모드를 따로 고르지 않고 학습 클래스의 유형으로 단어/문장을 구분합니다. 전체 자유 문장 번역은 아닙니다.
기존 PC 제어·발표 제어 메뉴는 `main.py`로 실행하고, 아래 수어 기능은 `main.py --live-ksl`로 따로 실행합니다.

## 1. 준비물

- Windows 10/11, 웹캠, Python 3.11(팀에서 3.11.9로 확인)
- 이 Git 저장소의 최신 코드
- 별도 전달받은 `KSL30_학습용_모델포함.zip` (약 1.5MB)
- 원본 JSON을 확인할 사람만 `KSL30_선별원본_JSON.zip` (약 6.8MB)

ZIP, 키포인트, morpheme, 학습 모델, 생성된 골격 HTML은 Git에 올리지 않습니다. 원본 영상은 두 ZIP에 없습니다.
ZIP에도 최소 학습 코드가 들어 있지만, **최신 코드는 이 저장소를 사용하세요.** ZIP 안의 코드로 저장소를 덮어쓰지 않습니다.

## 2. 저장소와 자료 배치

OneDrive 밖의 짧은 경로(예: `C:\CV_Project\Vision-Gesture-Hub`)를 권장합니다.
이미 clone한 팀원은 작업 중인 변경을 보존한 상태에서 최신 코드를 받으세요.
학습용 ZIP은 먼저 임시 폴더에 풀고, 아래 세 항목만 저장소로 복사합니다.

| ZIP에서 가져올 항목 | 저장소 안에 놓을 위치 |
|---|---|
| `data/ksl30_shared` 폴더 | `data/ksl30_shared` |
| `models/ksl30_robust_v2` 폴더 | `models/ksl30_robust_v2` |
| `학습동작_골격보기.html` | 저장소 최상위(선택) |

정상 배치 예시:

```text
Vision-Gesture-Hub/
  main.py
  live_ksl.py
  수어인식실행.cmd
  data/ksl30_shared/
    manifest.json
    clips/                    ← NPZ 30개
  models/ksl30_robust_v2/
    model.pt
    classes.json
    metrics.json
    provenance.json
  학습동작_골격보기.html      ← 선택
```

`models/models/...`처럼 폴더가 두 번 겹치지 않았는지 확인하세요.
JSON ZIP은 원본 확인/재전처리가 필요한 경우에만 `data/ksl30_raw` 같은 별도 폴더에 풀면 됩니다. 웹캠 실행에는 필요하지 않습니다.

## 3. CMD 설치

저장소 폴더에서 CMD를 열고 순서대로 실행합니다. 가상환경 활성화는 필요하지 않습니다.

```cmd
cd /d C:\CV_Project\Vision-Gesture-Hub
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.venv\Scripts\python.exe scripts/download_model.py
.venv\Scripts\python.exe scripts/download_pose_model.py
```

이미 올바른 `.venv`가 있다면 다시 만들지 않아도 됩니다. 설치와 MediaPipe 모델 다운로드에는 인터넷이 필요합니다. 추론은 로컬에서 실행됩니다.

## 4. 웹캠 실행

```cmd
.venv\Scripts\python.exe main.py --live-ksl
```

또는 `수어인식실행.cmd`를 더블클릭합니다. 기본 모델은 `models/ksl30_robust_v2`입니다.
카메라가 열리지 않으면 다른 카메라 앱을 닫고 `--camera 1`을 시도하세요.

| 키 | 동작 |
|---|---|
| Space | 입력 시작 / 동작 완료 |
| Enter | 누적 단어 조합 |
| Backspace | 마지막 단어 삭제(문장만 있으면 출력 삭제) |
| C | 초기화 |
| Q / Esc | 카메라 종료 |

카메라 창을 클릭해 키 입력 초점을 맞춥니다. 양쪽 어깨와 손을 화면 안에 두고 시작 자세를 잡은 뒤, **Space → 단어 또는 문장 한 개 수행 → Space**로 입력하세요.
입력은 최대 12초입니다. 손이 잠깐 가려졌다고 자동 완료하지 않으며, 0.6초 이상 추적에 실패하면 취소합니다.

- 단어: 인식한 단어를 차례대로 누적합니다.
- 문장: 누적 단어를 비우고 해당 문장을 바로 출력합니다.
- 예: `커피`와 `부탁`을 각각 입력하면 `커피를 주세요.`를 표시합니다.
- 지원 규칙 외 조합은 인식한 단어를 이어 표시합니다. 자연스러운 한국어를 자유 생성하지 않습니다.
- 게임용 `game_action` 정보는 제공하지만 게임이나 OS 동작을 자동 실행하지 않습니다.

## 5. 실제 학습 동작 보기

전달받은 `학습동작_골격보기.html`을 브라우저에서 엽니다. 동작 선택, 반복 재생, 0.25/0.5/1배 속도, 거울 보기와 양손 확대를 지원합니다.
직접 다시 만들려면:

```cmd
.venv\Scripts\python.exe scripts/export_skeleton_viewer.py --data data/ksl30_shared
```

HTML에는 실제 좌표가 포함되므로 생성 파일도 Git에 올리지 않습니다. 골격은 얼굴 표정, 손바닥 방향, 깊이를 완전히 표현하지 못합니다.

## 6. 재학습(선택)

전달된 모델로 실행만 할 사람은 재학습하지 않아도 됩니다.
기본 GRU 재학습:

```cmd
.venv\Scripts\python.exe -m ksl.train --data data/ksl30_shared --output models/team_retrain --smoke-only --epochs 150
```

속도·입력 경계·작은 좌표 오차·짧은 손 추적 누락을 변형한 증강 학습:

```cmd
.venv\Scripts\python.exe scripts/train_robust_ksl30.py --data data/ksl30_shared --output models/team_augmented
.venv\Scripts\python.exe main.py --live-ksl --model models/team_augmented
```

증강 학습의 출력 폴더는 새 이름을 사용하세요. 기존 모델 폴더가 있으면 덮어쓰지 않고 중단합니다.
`--smoke-only`는 독립 검증 없이 학습 자료 재인식만 확인한다는 뜻입니다. 검증 정확도로 발표하지 마세요.
다른 촬영자가 모든 클래스를 갖추면 `--validation-group REAL02`처럼 촬영 그룹을 분리해 평가합니다. 같은 촬영의 여러 각도를 학습/검증에 나누지 않습니다.

## 7. 인식이 잘 안 될 때

- `Model not found`: `models/ksl30_robust_v2/model.pt`와 `classes.json` 배치를 확인합니다.
- `No module named torch/PIL`: 위 두 requirements 파일을 같은 `.venv`에 설치했는지 확인합니다.
- 추적 실패: 얼굴만 크게 잡지 말고 상반신·양쪽 어깨·손이 보이게 합니다.
- `인식 보류`: 후보 이름·점수·1/2위 차이를 확인합니다. 후보는 확정 결과가 아니며 자막에 추가하지 않습니다.
- 시작/끝에 긴 준비 동작을 넣지 말고, 동작 직전/직후 Space를 누릅니다.
- 창에서 키가 안 먹으면 카메라 창을 클릭합니다.
- 종료 후 수정 코드를 적용하려면 프로그램을 다시 실행합니다.

## 8. 데이터와 성능 해석

현재 원본은 **촬영자 REAL01, 정면 동작 30개(클래스당 1개)**입니다. 증강을 해도 독립 촬영자가 늘어나지는 않습니다.
학습 자료 재인식 30/30이나 합성 변형 자료 성능은 실제 웹캠·새 사용자 정확도가 아닙니다.
AIHub 키포인트와 웹캠 MediaPipe의 추출기 차이도 남아 있습니다. 점수는 보정된 정답 확률이 아닙니다.
실제 동작 정확도 개선에는 여러 사람의 독립 촬영·검증 자료가 필요합니다.
라벨 설정과 영상 담당자의 좌표 계약은 [학습 안내](ksl_training.md), 실행 세부 사항은 [웹캠 안내](live_ksl_30.md)를 참고하세요.

웹캠 실행은 영상을 저장하거나 전송하지 않습니다. 원본 데이터의 이용 조건은 사용자가 확인하고, 팀 자료는 별도로 전달합니다.
