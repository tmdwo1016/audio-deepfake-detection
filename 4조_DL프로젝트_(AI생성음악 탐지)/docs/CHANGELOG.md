# 변경사항

## 2026-09-22 — 발표 이후 피드백: 단일 곡 입력 판정

- `23_post_presentation_feedback_single_track_inference.ipynb`와 단일 파일 추론 코드 추가
- 제공된 AI 생성곡 한 곡과 별개 AI 샘플의 인간 원곡 한 곡을 Logistic Regression·RBF-SVM·Log-Mel CNN·Frozen MERT+LR로 확인
- 네 모델 모두 AI 생성곡은 `AI 생성`, 인간 원곡은 `인간 제작`으로 판정한 입력 QC·모델별 점수·주의사항을 `docs/23_post_presentation_feedback.md`에 기록
- 만장일치 인간 판정을 `모델 불일치`로 표시하던 합의 결과 조건을 수정

## 2026-09-22 — MERT 강건성 확장 및 제출본 정리

### 실험과 결과

- `22_mert_unseen_generator.ipynb` 추가: MusicGen·Udio holdout별 MERT+LR 재학습 및 평가
- 미노출 생성기 Track ROC-AUC: MusicGen 0.9407, Udio 0.9181
- 동일 Test 539곡의 MERT MP3 평가: Original 0.9845, 128 kbps 0.9851, 64 kbps 0.9605
- MERT MP3 64 kbps AUC 하락은 0.0240이었으며, REAL FPR은 0.0889에서 0.2889로 증가
- 관련 CSV·그림과 `docs/22_mert_unseen_generator_report.md` 추가

### 문서와 제출본

- 모든 노트북의 실행 결과 뒤에 제출용 결과 요약을 추가하고 표현을 정리
- `README.md`와 `docs/FINAL_REPORT.md`를 최신 MERT unseen-generator·MP3 결과로 갱신
- 전처리 데이터, 코드, 통합 요약표, 핵심 결과와 발표자료를 `4조_DL프로젝트_(AI생성음악 탐지)/`에 정리
- 제출본의 개인 절대경로를 제거하고 파일 목록·SHA-256 체크섬을 추가
- 대용량 원본 오디오와 재생성 가능한 Log-Mel/MERT 배열은 용량 및 재배포 조건 때문에 제외

## 2026-09-13 — 분석 최종본

### 문서

- 프로젝트 목적, 분석 흐름, 대표 결과, 실행 순서를 담은 `README.md` 추가
- 방법론, 전체 결과, 해석, 제한사항, 후속 연구를 담은 `docs/FINAL_REPORT.md` 추가
- 모든 완료 노트북의 마지막 마크다운을 실제 실행 결과 기준으로 갱신
- GitHub에서 바로 확인할 수 있도록 결과 그림과 상대 경로 연결

### 데이터 및 전처리

- Echoes TTA 충돌 및 누락 검증
- 296개 FMA REAL track 매칭·다운로드·decode QC
- 3,458-track master manifest 구축
- `original_audio` 기준 group split으로 source-family 누수 방지
- 10,077개 10초 segment와 266-D handcrafted feature 생성

### Detection

- Logistic Regression 및 RBF-SVM baseline 추가
- Log-Mel CNN 학습과 Segment/Track 평가 추가
- Frozen MERT-v1-95M 13-layer 탐색 및 LR baseline 추가
- MERT/Transformers 및 Apple Silicon 실행 안정성을 반영한 `17_mert_frozen_baseline_v2.ipynb` 추가

### Robustness

- MusicGen/Udio unseen-generator 실험 추가
- MP3 128/64 kbps full-decode robustness 실험 추가
- Validation threshold 고정 및 Train-only preprocessing 원칙 적용

### Generator fingerprint

- Handcrafted + RBF-SVM 12-way generator attribution 추가
- Source coverage를 통제한 controlled 및 strict-balanced 실험 추가
- Frozen MERT Layer 10 + LR attribution 추가
- Handcrafted/MERT PCA·UMAP 및 generator/genre silhouette 분석 추가
- Generator × genre recall과 Train-only acoustic feature ANOVA 추가

### 주요 결과

- In-domain Track ROC-AUC: RBF-SVM 0.9644, CNN 0.9772, MERT 0.9872
- MP3 64 kbps Track ROC-AUC: RBF-SVM 0.9062, CNN 0.8256
- Strict-balanced 12-way Macro-F1: Handcrafted 0.7447, MERT 0.8980
- 높은 supervised attribution 성능과 낮은 silhouette를 함께 확인

### 저장 정책

- 재현 가능한 compact CSV/JSON/PNG 결과를 `results/`에 보존
- 원본 오디오, 대용량 feature cache, model checkpoint는 `.gitignore`로 제외
