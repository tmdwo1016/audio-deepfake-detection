# 코드 구성과 실행 순서

아래 노트북은 모두 `notebooks/` 폴더에 있다. 각 파일에는 저장된 실행 결과와 코드 셀별 결과 요약이 포함돼 있다.

## 데이터 준비

| 순서 | 노트북 | 역할 |
|---:|---|---|
| 01 | `01_data_quality_check.ipynb` | Echoes·FMA 구조, 누락, 중복, 충돌 점검 |
| 02 | `02_fma_matching_check.ipynb` | Echoes 원곡과 FMA REAL 매칭 |
| 03 | `03_fma_audio_validation.ipynb` | FMA 오디오 크기·디코딩·길이 검증 |
| 04 | `04_build_master_manifest.ipynb` | REAL·FAKE 통합 manifest 생성 |
| 05 | `05_group_split.ipynb` | `original_audio` 단위 Train/Val/Test 분할 |
| 06 | `06_eda.ipynb` | 데이터 분포 점검 |
| 07 | `07_build_segment_manifest.ipynb` | 10초 segment manifest 생성 |
| 08 | `08_extract_handcrafted_features.ipynb` | 266차원 Handcrafted 특징 추출 |

## 탐지와 강건성 실험

| 순서 | 노트북 | 역할 |
|---:|---|---|
| 09 | `09_baseline_models.ipynb` | Logistic Regression·RBF-SVM 기준 모델 |
| 10 | `10_baseline_subgroup_analysis.ipynb` | 생성기·장르별 subgroup 분석 |
| 11 | `11_unseen_generator_handcrafted.ipynb` | Handcrafted 미노출 생성기 평가 |
| 12 | `12_mp3_robustness_handcrafted.ipynb` | Handcrafted MP3 강건성 |
| 13 | `13_logmel_cnn.ipynb` | Log-Mel CNN 기준 모델 |
| 14 | `14_cnn_unseen_generator.ipynb` | CNN 미노출 생성기 평가 |
| 15 | `15_cnn_mp3_robustness.ipynb` | CNN MP3 강건성 |
| 16 | `16_final_integration_analysis.ipynb` | SVM·CNN 1차 통합 분석 |
| 17 | `17_mert_frozen_baseline_v2.ipynb` | Frozen MERT in-domain 기준 모델 |
| 22 | `22_mert_unseen_generator.ipynb` | 최신 MERT 미노출 생성기·MP3 평가 |

## 생성기 분류와 해석

| 순서 | 노트북 | 역할 |
|---:|---|---|
| 18 | `18_generator_attribution_handcrafted_v3.ipynb` | Handcrafted 12-way attribution |
| 18B | `18B_balanced_controlled_generator_attribution.ipynb` | Strict-balanced 자료 구성·평가 |
| 19 | `19_generator_attribution_mert.ipynb` | MERT 12-way attribution |
| 20 | `20_generator_fingerprint_pca_umap.ipynb` | PCA·UMAP과 silhouette 분석 |
| 20-v2 | `20_generator_fingerprint_pca_umap_v2.ipynb` | 20번 시각화 실행본 보존 사본 |
| 21 | `21_generator_genre_feature_analysis.ipynb` | 장르별 성능과 음향 특징 분석 |

`src/02_download_fma_real.py`와 `src/03_extract_fma_real_remote.py`는 선택한 FMA REAL 파일을 내려받거나 원격 ZIP에서 추출하는 보조 코드다.

패키지의 노트북은 개인 로컬 절대경로를 제거한 복사본이다. 원래 프로젝트의 코드와 실행 결과는 변경하지 않았다.
