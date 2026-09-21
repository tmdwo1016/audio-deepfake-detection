# 프로젝트 요약

## 1. 연구 목적

실제 음악과 AI 생성 음악을 구분하고, 학습에서 제외한 생성기와 MP3 재인코딩 환경에서도 탐지 성능이 유지되는지 확인했다. 추가로 AI 생성 음악만을 대상으로 12개 생성기를 분류해 생성기별 음향 차이가 남는지도 살펴봤다.

같은 원곡에서 파생된 REAL·FAKE 자료가 서로 다른 split에 섞이지 않도록 모든 분할은 `original_audio` 단위로 수행했다.

## 2. 데이터와 전처리

Echoes manifest 4,468행 중 TTA만 사용하고 원곡 연결이 충돌한 3행을 제외해 FAKE 3,162곡을 남겼다. 제목·아티스트를 기준으로 FMA metadata를 매칭하고 파일 크기, 디코딩, 길이를 검사해 REAL 296곡을 확정했다.

| 단위 | 전체 | Train | Validation | Test |
|---|---:|---:|---:|---:|
| `original_audio` 그룹 | 296 | 207 | 44 | 45 |
| Track | 3,458 | 2,392 | 527 | 539 |
| 10초 segment | 10,077 | 6,967 | 1,538 | 1,572 |

Track은 REAL 296곡과 FAKE 3,162곡, segment는 REAL 888개와 FAKE 9,189개다. FAKE는 12개 생성기와 Electronic·Pop·Rock 세 장르로 구성된다.

오디오는 24 kHz mono로 읽고 부족한 구간만 뒤쪽을 0으로 채웠다. 입력 표현은 다음 세 가지다.

- Handcrafted: MFCC·delta·delta²·spectral·RMS·ZCR 통계 266차원
- Log-Mel: 128-bin spectrogram
- MERT: frozen MERT-v1-95M의 13개 layer, layer별 768차원 embedding

Scaler와 모델은 Train에서만 학습했다. Threshold, MERT layer와 early stopping 시점은 Validation에서 정한 뒤 Test에 고정 적용했다. 같은 곡의 segment 점수는 평균해 track 점수로 사용했다.

## 3. 사용 모델

- Logistic Regression, RBF-SVM + Handcrafted 특징
- 소형 2D CNN + Log-Mel
- Logistic Regression + frozen MERT embedding
- 12-way generator attribution: RBF-SVM 또는 MERT+LR

노트북별 역할과 실행 순서는 `CODE_INDEX.md`에 정리했다.

## 4. In-domain 탐지 결과

| 모델 | Track ROC-AUC | EER | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| RBF-SVM | 0.9644 | 0.1395 | 0.8696 | 0.7502 |
| Log-Mel CNN | 0.9772 | 0.1042 | 0.9111 | 0.8535 |
| MERT95M+LR | **0.9872** | **0.0455** | **0.9363** | 0.8107 |

세 모델 모두 segment보다 track 평균에서 ROC-AUC가 높고 EER이 낮았다. 이 조건에서는 MERT가 ROC-AUC와 EER 기준으로 가장 좋은 결과를 보였다.

## 5. 미노출 생성기 탐지

MusicGen 또는 Udio의 FAKE를 Train과 Validation에서 모두 제외한 뒤 평가했다.

| Holdout | 모델 | Track ROC-AUC | EER | Balanced Accuracy | FAKE Miss Rate |
|---|---|---:|---:|---:|---:|
| MusicGen | RBF-SVM | 0.6123 | 0.4000 | 0.5222 | 0.8000 |
| MusicGen | Log-Mel CNN | 0.8681 | 0.2667 | 0.7333 | 0.4000 |
| MusicGen | MERT+LR | **0.9407** | **0.0889** | **0.8556** | 0.2000 |
| Udio | RBF-SVM | 0.8542 | 0.2368 | 0.7361 | 0.4167 |
| Udio | Log-Mel CNN | 0.8227 | 0.2146 | 0.7653 | 0.2917 |
| Udio | MERT+LR | **0.9181** | **0.1292** | **0.8097** | 0.2917 |

MERT가 두 holdout에서 가장 높은 AUC를 보였지만, 평가 대상이 두 생성기에 한정되므로 모든 미노출 생성기로 확대 해석하지 않는다.

Holdout별 scaler·LR 가중치와 임계값은 대상 생성기를 제외한 자료로 다시 학습·선택했다. 다만 layer 4와 C=0.01은 전체 생성기 binary Validation에서 미리 선택된 설정이므로, 설정 선택까지 완전히 생성기 독립적인 실험은 아니다.

## 6. MP3 압축 강건성

같은 Test 539곡을 original, MP3 128 kbps, MP3 64 kbps 조건에서 비교했다.

| 모델 | Original AUC | MP3 128 AUC | MP3 64 AUC | Original 대비 64k 하락 |
|---|---:|---:|---:|---:|
| RBF-SVM | 0.9644 | 0.9536 | 0.9062 | 0.0583 |
| Log-Mel CNN | 0.9772 | 0.8954 | 0.8256 | 0.1516 |
| MERT+LR | 0.9845 | 0.9851 | 0.9605 | **0.0240** |

MERT의 AUC 감소가 가장 작았지만 64 kbps에서 REAL 오탐률은 0.0889에서 0.2889로 증가했다. AUC와 오류 방향을 함께 봐야 한다.

MP3 표의 MERT는 layer 4 valid-frame/full-decode cache를 사용했다. 일반 in-domain 표의 layer 9 MERT와 전처리·선택 설정이 달라 Original AUC가 0.9845와 0.9872로 서로 다르다.

## 7. 12-way 생성기 분류

| 표현 | 조건 | Accuracy | Macro-F1 | Top-3 Accuracy | Macro OVR AUC |
|---|---|---:|---:|---:|---:|
| Handcrafted+RBF-SVM | Full | 0.8846 | 0.8840 | 0.9798 | 0.9765 |
| MERT95M+LR | Full | **0.9393** | **0.9384** | 0.9879 | 0.9959 |
| Handcrafted+RBF-SVM | Strict balanced | 0.7544 | 0.7447 | 0.9386 | 0.9534 |
| MERT95M+LR | Strict balanced | **0.8991** | **0.8980** | 0.9912 | 0.9920 |

Strict-balanced 자료는 12개 생성기가 모두 존재하는 119개 원곡만 사용하고 `original_audio × generator`당 한 곡으로 맞췄다. 총 1,428곡이며 Test는 생성기별 19곡, 총 228곡이다. 이 조건에서도 생성기 분류 성능이 유지됐지만 특정 artifact의 인과관계를 증명하는 결과는 아니다.

## 8. 표현 공간과 장르 분석

원공간 generator silhouette는 Handcrafted -0.0183, MERT 0.0164로 전역 군집 분리는 약했다. 높은 supervised 분류 성능이 2차원에서 서로 분리된 단순 군집으로 나타나는 것은 아니었다.

Strict-balanced Test의 장르별 MERT Macro-F1은 Electronic 0.8767, Pop 0.9667, Rock 0.8700이었다. Train-only ANOVA에서는 `spectral_contrast_07_mean`이 가장 높은 단일 F-score를 보였고, 그룹 평균은 RMS와 Flatness가 높았다.

## 9. 해석 범위

- 데이터는 296개 원곡, 12개 생성기, 세 장르로 제한된다.
- 미노출 생성기 평가는 MusicGen과 Udio 두 종류만 포함한다.
- MP3 평가는 한 재인코딩 파이프라인의 128/64 kbps 조건이다.
- 원본 오디오를 재배포하지 않으므로 전체 재실행에는 별도 데이터 준비가 필요하다.
- Threshold 기반 지표는 모델별 score calibration의 영향을 받으므로 ROC-AUC·EER와 함께 해석한다.

표의 원자료와 더 높은 정밀도의 값은 `summary/*.csv`와 `results/`에서 확인할 수 있다.
