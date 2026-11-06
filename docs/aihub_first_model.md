# AI Hub 키포인트 첫 기준 모델

실제 내려받은 단어 라벨과 키포인트에서 8개 단어, REAL01의 5개 방향씩 총 40개
클립을 추출해 학습한 시계열 템플릿 모델입니다. LSTM 등 신경망 학습 모델은 아닙니다.

| 라벨 | 단어 ID | 한국어 |
|---|---|---|
| GO | 0943 | 가다 |
| HURT | 1152 | 아프다 |
| I | 1157 | 나 |
| HEAD | 1249 | 머리 |
| HOSPITAL | 1496 | 병원 |
| FOOD | 1534 | 밥 |
| REQUEST | 1589 | 부탁 |
| HELP | 2388 | 돕다 |

제공된 morpheme 시작/종료 시각과 30fps 프레임 번호로 단어 구간을 추출했습니다.
양손 2D 손 모양과 어깨 중심/너비 기준 손목 위치, 손 유무를 32단계 시계열로 변환합니다.
2D의 세 번째 값은 깊이가 아닌 신뢰도로 처리합니다. 거울 반전은 적용하지 않습니다.

## 실행 (저장소 루트, NumPy 필요)

단어 예측:

```powershell
python -m core.aihub_words data/aihub_first/train/NIA_SL_WORD1152_REAL01_F.json
```

단어 인식 결과를 문장으로 조합하는 오프라인 데모:

```powershell
python -m training.predict_aihub_sentence data/aihub_first/train/NIA_SL_WORD1157_REAL01_F.json data/aihub_first/train/NIA_SL_WORD1249_REAL01_F.json data/aihub_first/train/NIA_SL_WORD1152_REAL01_F.json
```

예상 출력은 I, HEAD, HURT -> '저는 머리가 아파요.'입니다. 이 예제는 학습 클립을
다시 입력하는 동작 확인입니다. 미학습 문장/새 사용자 정확도 실험이 아닙니다.
문장 규칙은 한국어 의미를 조립하는 시연용이며 실제 한국수어 문법 검증을 대신하지 않습니다.

모델: `models/ksl_aihub_first.npz` (약 380KiB)
보고서: `models/ksl_aihub_first.metrics.json`

## 결과와 한계

- 학습 데이터 재입력: 40/40개를 원래 단어로 수락. 일반화 정확도로 해석하지 마세요.
- 별도로 제외한 고민/운전면허/노래 15개 클립 중 11개 거절, 4개 오수락.
- 거절 임계값은 고정 휴리스틱이며 보정되지 않았습니다. 진단 결과로 재조정하지 않았습니다.
- REAL01 한 코드뿐이며 단어당 5개는 같은 수행의 다른 방향입니다.
- 독립된 촬영자/반복 동작의 평가 자료가 없어 검증 정확도는 기록하지 않았습니다.
- 원본 키포인트 전체 손상 검사를 한 것은 아니지만 선택한 클립은 실제 JSON 읽기와 특징 변환을 통과했습니다.
- **이 모델을 main.py의 --sign-model로 넣지 마세요.** 그 경로는 MediaPipe XYZ 모델용입니다.
  이 모델은 어깨 좌표도 필요한 AI Hub 2D 형식입니다. 현재 웹캠 검출기에는 몸 자세가 없어서
  웹캠에서 바로 사용하는 것은 미지원입니다. 제공 좌표와 MediaPipe 간 변환 및 실제 촬영 평가가 필요합니다.

## 재현

```powershell
python -m training.prepare_aihub_first --source "C:/CV_Project/data/hand_gesture_word"
python -m unittest discover -s tests -v
```

분할 압축을 가상으로 연결해 필요한 파일만 읽으므로 25GB 전체를 풀 필요가 없습니다.
원본 다운로드 파일은 변경하지 않습니다. 모델과 추출 좌표는 개인정보/데이터 이용 조건에 따라
로컬 보관하며 자동으로 Git에 추가하지 않습니다.
