# 발표 이후 피드백: 단일 곡 AI 생성 여부 판별

## 목적

발표 이후 요청된 “곡 파일 하나를 넣었을 때 AI 생성 여부를 판별할 수 있는가?”를 확인하기 위해, 제공된 MP3 두 개를 프로젝트의 최종 튜닝 모델에 입력했다. 파일명과 메타데이터는 판정 특징으로 사용하지 않았고, 디코드된 오디오 신호만 사용했다.

실행 코드는 [23_post_presentation_feedback_single_track_inference.ipynb](../23_post_presentation_feedback_single_track_inference.ipynb), 재사용 가능한 추론 코드는 [`src/single_track_inference.py`](../src/single_track_inference.py), 원시 결과표는 [`results/post_presentation_feedback/`](../results/post_presentation_feedback/)에 있다.

## 입력 QC

| 파일 | 길이 | 원본 포맷 | 원본 샘플레이트 | 채널 | 평균 비트레이트 | SHA-256 앞 12자리 |
|---|---:|---|---:|---:|---:|---|
| `fake_00001_suno_1.mp3` | 225.96초 | MP3 Layer III | 16 kHz | mono | 40,479 bps | `117c81e8529a` |
| `fake_00001_suno_0.mp3` | 224.00초 | MP3 Layer III | 16 kHz | mono | 38,799 bps | `2c6d06d5d79e` |

프로젝트 규칙에 맞춰 각 곡을 24 kHz mono로 디코드한 뒤 시작·중간·끝 10초씩, 곡당 3개 구간을 사용했다. 각 모델의 곡 점수는 세 구간 점수의 평균이다.

## 모델별 판정

`AI score`는 AI 생성 방향의 모델 점수다. Logistic Regression과 CNN/MERT의 출력도 보정된 실제 확률로 단정하지 않는다. RBF-SVM은 특히 확률이 아닌 decision margin이다.

| 곡 | 모델 | 구간 점수 | 곡 평균 | Track 임계값 | 판정 |
|---|---|---|---:|---:|---|
| `suno_1` | Logistic Regression | 0.983363, 0.958351, 0.998879 | 0.980198 | 0.563456 | **AI 생성** |
| `suno_1` | RBF-SVM (margin) | 2.744112, 2.183724, 4.053620 | 2.993819 | 0.661505 | **AI 생성** |
| `suno_1` | Log-Mel CNN | 0.999949, 0.999332, 0.999443 | 0.999575 | 0.241069 | **AI 생성** |
| `suno_1` | Frozen MERT + LR | 0.993177, 0.999511, 0.999985 | 0.997558 | 0.588751 | **AI 생성** |
| `suno_0` | Logistic Regression | 0.991968, 0.955729, 0.998957 | 0.982218 | 0.563456 | **AI 생성** |
| `suno_0` | RBF-SVM (margin) | 3.101345, 2.301469, 3.129101 | 2.843972 | 0.661505 | **AI 생성** |
| `suno_0` | Log-Mel CNN | 0.999643, 0.999527, 0.999831 | 0.999667 | 0.241069 | **AI 생성** |
| `suno_0` | Frozen MERT + LR | 0.810181, 0.999670, 0.999697 | 0.936516 | 0.588751 | **AI 생성** |

## 결론

- `fake_00001_suno_1.mp3`: 네 모델 모두 **AI 생성**.
- `fake_00001_suno_0.mp3`: 네 모델 모두 **AI 생성**.
- 모델 합의: 두 곡 모두 **4/4 모델 AI 생성**.

이번 결과는 단일 입력에 대한 판정 시연이다. 두 파일만으로 모델의 일반화 성능을 새롭게 평가할 수 없고, AI 생성 여부를 법의학적으로 100% 확정하는 증거도 아니다. 두 입력은 16 kHz mono·약 39–40 kbps의 저비트레이트 MP3이므로 codec 분포 이동이 존재한다. 따라서 최종 표현은 “현재 고정 모델들이 네 모델 모두 AI 생성으로 판정했다”가 적절하다.

