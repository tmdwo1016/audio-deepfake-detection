# AI 생성 음악 탐지 프로젝트 — 제출용 패키지

이 폴더는 프로젝트의 핵심 산출물을 제출·검토하기 쉽도록 모은 경량 패키지다. 전처리 후 데이터, 실행 코드, 실험 요약, 핵심 결과 CSV와 대표 그림을 포함한다.

## 먼저 볼 파일

1. `SUMMARY.md`: 데이터, 전처리, 실험과 핵심 결과 요약
2. `summary/`: 보고서용 통합 결과 CSV 5개와 제외 산출물 목록
3. `docs/FINAL_REPORT.md`: 최신 통합 분석 보고서
4. `data/README.md`: 포함한 전처리 데이터와 제외한 대형 파일 설명
5. `CODE_INDEX.md`: 노트북 실행 순서와 역할
6. `results/README.md`: 핵심 결과표 위치와 해석 주의사항

## 폴더 구성

```text
4조_DL프로젝트_(AI생성음악 탐지)/
├── notebooks/                    # 실행 결과를 포함한 분석 노트북 24개
├── src/                          # FMA 다운로드·추출 보조 코드
├── data/
│   ├── metadata/                 # 최종 manifest, split, 검증 기록
│   └── processed/                # Handcrafted CSV와 Log-Mel/MERT index
├── results/                      # 핵심 결과 CSV와 대표 그림
├── summary/                      # 제출용 통합 표
├── docs/                         # 기존 보고서와 최신 MERT 추가 보고서
└── 4조 발표자료.pdf               # 최종 발표자료
```

## 포함 범위

- 최종 track manifest와 고정 Train/Validation/Test split
- 모델이 사용한 10초 segment 10,077개의 manifest
- Handcrafted 266차원 특징 CSV와 MP3 128/64 kbps Test 특징 CSV
- Log-Mel·MERT cache의 행 대응 index와 메타데이터
- `notebooks/`의 분석 노트북 24개와 `src/` 스크립트
- 핵심 결과표 21개, 대표 그림, 통합 결과 CSV
- 최종 발표자료 PDF

## 포함하지 않은 항목

원본 오디오, Log-Mel/MERT 대형 배열, 모델 checkpoint, 재실행 중복 결과와 개별 예측값은 제외했다. 원본 오디오는 약 9.9 GB이며 곡별 재배포 권리가 서로 다르다. Log-Mel·MERT 배열은 재생성 가능한 cache로 합계가 수 GB에 달한다. 자세한 목록은 `summary/artifacts_not_included.csv`에 정리했다.

## 재실행 시 주의사항

- 노트북 복사본의 개인 로컬 경로는 제거했다. `notebooks/`에서 실행해도 상위의 `data/`와 `results/`를 자동으로 찾도록 구성했다.
- `09`번 Handcrafted baseline 이후 일부 분석은 포함된 CSV만으로 확인할 수 있다.
- `01`~`08`의 오디오 전처리와 CNN/MERT cache를 처음부터 재생성하려면 별도로 원본 Echoes·FMA 오디오를 준비해야 한다.
- `docs/FINAL_REPORT_pre_MERT_extensions.md`는 22번 후속 MERT 실험 이전 보고서다. 최신 결과는 `SUMMARY.md`, `docs/FINAL_REPORT.md`, `docs/22_mert_unseen_generator_report.md`를 기준으로 한다.

패키지 생성일: 2026-09-22
