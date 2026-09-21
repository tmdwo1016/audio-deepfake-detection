# 전처리 후 데이터

## 핵심 파일

| 파일 | 행 수 | 내용 |
|---|---:|---|
| `metadata/master_manifest_with_split.csv` | 3,458 | Track 라벨, 출처, 장르, 생성기, 고정 split |
| `metadata/original_audio_split.csv` | 296 | 원곡 그룹별 Train/Validation/Test 배정 |
| `metadata/segment_manifest_10s.csv` | 10,077 | 실제 모델 입력 구간의 시작·종료 시점과 padding 정보 |
| `processed/features/handcrafted_features_10s.csv` | 10,077 | Segment metadata와 266개 Handcrafted 특징 |
| `processed/features/mp3_robustness/handcrafted_test_mp3_128.csv` | 1,572 | MP3 128 kbps Test 특징 |
| `processed/features/mp3_robustness/handcrafted_test_mp3_64.csv` | 1,572 | MP3 64 kbps Test 특징 |

`metadata/`에는 매칭, 오디오 검증, duration cache도 함께 보존했다. 제외된 segment가 없어 내용이 비어 있던 `segment_exclusions.csv`는 제출본에서 생략했다. `metadata/attribution/strict_selected_tracks.csv`는 12개 생성기의 strict-balanced 실험에 사용한 1,428곡 목록이다.

## Index와 cache

Log-Mel·MERT의 대형 `.npy` 배열은 포함하지 않았다. 대신 다음 index와 메타데이터를 제공한다.

- `processed/logmel/logmel_10s_index.csv`
- `processed/mert/*_index.csv`, `*_meta.json`
- `processed/model_robustness/mp3/mert/*_index.csv`, `*_meta.json`

Index의 `segment_id` 또는 `cache_index`를 `segment_manifest_10s.csv`와 연결하면 원래 배열의 행 의미를 확인할 수 있다.

## 제외 이유

- 원본 Echoes·FMA 오디오는 약 9.9 GB이며 재배포 조건이 파일마다 다르다.
- Log-Mel·MERT 배열은 수 GB 크기의 재생성 가능한 cache다.
- Handcrafted checkpoint PKL은 최종 CSV와 내용이 중복된다.

CSV의 `audio_path`는 원래 프로젝트 안의 상대경로다. 오디오가 포함됐다는 의미는 아니다.
