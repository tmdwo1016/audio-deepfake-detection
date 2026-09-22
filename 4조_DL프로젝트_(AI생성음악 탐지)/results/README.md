# 결과표 안내

이 폴더에는 기존 실험의 개별 예측값과 재실행 중복 파일을 제외하고, 보고서에 필요한 핵심 CSV 21개와 대표 그림을 담았다. 발표 이후 단일 곡 시연의 입력 QC·구간별 예측·곡별 예측·합의 결과는 `post_presentation_feedback/`에 별도로 포함했다.

## 우선 확인할 표

| 주제 | 파일 |
|---|---|
| 데이터 split | `eda/group_split_summary.csv`, `eda/sample_split_summary.csv` |
| In-domain 세 모델 비교 | `mert/mert_vs_svm_cnn_track_test_v2.csv` |
| 미노출 생성기 세 모델 비교 | `mert_unseen_generator/mert_unseen_comparison.csv` |
| MP3 세 모델 비교 | `mert_mp3_robustness/mert_mp3_track_comparison.csv` |
| Full/Strict attribution 비교 | `generator_attribution/mert/handcrafted_vs_mert_generator_attribution.csv` |
| PCA·UMAP 정량 보조 지표 | `generator_fingerprint_visualization/generator_silhouette_scores.csv` |
| 장르별 attribution | `generator_genre_feature_analysis/genre_model_performance.csv` |
| 단일 곡 판정 시연 | `post_presentation_feedback/track_predictions.csv`, `post_presentation_feedback/consensus.csv` |

## 해석 주의사항

`mert/mert_vs_svm_cnn_track_test_v2.csv`의 MERT는 in-domain baseline의 layer 9 결과다. `mert_mp3_robustness/mert_mp3_track_comparison.csv`의 MERT는 압축 조건을 동일한 디코딩 경로로 비교하기 위한 layer 4 valid-frame cache다. 두 표의 Original AUC가 0.9872와 0.9845로 다른 이유다.

`../docs/FINAL_REPORT_pre_MERT_extensions.md`의 통합표는 최신 22번 MERT 강건성 실험 이전 결과다. 최신 수치는 `../SUMMARY.md`, `../summary/`, `mert_unseen_generator/`, `mert_mp3_robustness/`를 기준으로 한다.

`post_presentation_feedback/`는 AI 생성곡 한 곡과 별개 AI 샘플의 인간 원곡 한 곡에 대한 작동 확인이다. 표본이 두 개뿐이고 인간 원곡은 학습 split에 포함되어 있으므로 일반화 성능이나 정확도 추정으로 해석하지 않는다.
