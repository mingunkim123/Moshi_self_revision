# Moshi self-repair matched-bundle dataset v2

이 디렉터리는 pretrained Moshi의 목적지 자기수정 능력을 평가하기 위한 controlled evaluation
set을 재현 가능하게 만드는 계약·데이터·검증 보고서를 담는다. 일반 학습 corpus가 아니다.

현재 상태는 **Kokoro raw production 및 10-speaker independent MFA alignment 완료,
human review 대기**다. 로컬에서 600/600 raw 합성·canonical 변환·독립 정렬·자동 QC까지
통과했다. 다만 hash-bound 사람 정렬 검수와 독립 청취자 승인이 끝나기 전에는 accepted
audio 또는 release-ready로 표시하지 않는다. 저장공간은 local TTS/QC/alignment와 remote
Moshi evaluation으로 분리했다.

2026-08-26 현재 30개 blueprint, 300개 script, 60개 answer key와 600개 TTS rendition
target은 생성·검증되었다. 비공개 Edge calibration 180개도 완료했지만 22개가 clipping
QC에서 제외되었고 fast/slow voice의 자연 지연 구간이 겹치지 않았다. 따라서 4초 예시는
폐기하고 production provider에서 화자별 timing을 다시 동결해야 한다.
Kokoro v1.0 voice audition 10개는 10/10 기술 QC를 통과했고 29.70–53.35초 범위를 보였다.
현재 production source assignment는 Kokoro로 재생성됐지만 사람 이중 청취 전 provisional이다.
Blueprint의 두 review 기록은 자동화된 독립 agent 검토이며 human sign-off를 대체하지 않는다.
공개 release 전 의미·안전 문구의 사람 검토를 별도 gate로 유지한다.

## 핵심 수량

- 의미 설계도 30개
- 방향 2개: Boston→Seattle, Seattle→Boston
- 조건 5개
- 대본 300개
- text bundle당 서로 다른 화자 2명
- source track당 accepted audio 600개
- generation seed 5개를 사용할 때 Moshi output 3,000개

## 재현 순서

저장소 루트에서 Python 3.12 환경을 만든다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r experiments/self_repair/requirements-v2.txt
.venv/bin/python -m pip install -r experiments/self_repair/requirements-kokoro-tts.txt
```

텍스트와 assignment는 다음 순서로 재생성·검증한다.

```bash
.venv/bin/python experiments/self_repair/scripts/dataset_v2/validate_blueprints.py
.venv/bin/python experiments/self_repair/scripts/dataset_v2/validate_schemas.py blueprint
.venv/bin/python experiments/self_repair/scripts/dataset_v2/generate_scripts.py \
  --report experiments/self_repair/dataset_v2/reports/script_generation.json
.venv/bin/python experiments/self_repair/scripts/dataset_v2/validate_scripts.py \
  --report experiments/self_repair/dataset_v2/reports/script_validation.json
.venv/bin/python experiments/self_repair/scripts/dataset_v2/validate_schemas.py script
.venv/bin/python experiments/self_repair/scripts/dataset_v2/generate_answer_keys.py \
  --report experiments/self_repair/dataset_v2/reports/answer_key_generation.json
.venv/bin/python experiments/self_repair/scripts/dataset_v2/assign_speakers.py
```

전체 단위 테스트:

```bash
.venv/bin/python -m unittest discover -s experiments/self_repair/tests/dataset_v2 -v
```

Production audio가 승인·선정·정렬되어 `prepared_stimuli.jsonl`이 만들어진 뒤에는 resolved
model revision과 clean Git commit을 40자리 SHA로 동결하고 평가 manifest를 만든다. 아래의
`<...>` 값은 실행 환경에서 채우며 credential은 인자로 넘기거나 파일에 기록하지 않는다.

```bash
.venv/bin/python experiments/self_repair/scripts/dataset_v2/build_eval_adapter.py \
  --generation-config experiments/self_repair/dataset_v2/config/eval.json \
  --artifact-root experiments/self_repair/dataset_v2 \
  --model-repo kyutai/moshiko-pytorch-bf16 \
  --resolved-revision <40_HEX_MODEL_REVISION> \
  --code-commit <40_HEX_CLEAN_GIT_COMMIT>

.venv/bin/python experiments/self_repair/scripts/dataset_v2/run_eval_v2.py \
  --generation-config experiments/self_repair/dataset_v2/config/eval.json \
  --artifact-root experiments/self_repair/dataset_v2 \
  --response-root <PRIVATE_RESPONSE_ARTIFACT_ROOT> \
  --dry-run
```

Dry-run 다음에는 standard Moshi checkpoint에서 1개 trial을 먼저 실행해 24 kHz/1,920-sample
frame, max LM delay 1, 입력 frame과 출력 token/audio coverage의 1:1 계약을 확인한다. 이
GPU contract smoke가 통과하기 전에는 3,000 trial 전체 실행을 시작하지 않는다. Runner는
사용자 발화 뒤 digital zero를 붙여 `utterance_end + response_capture_ms`까지 streaming
window를 실제로 생성하고, trial마다 model stream과 RNG를 초기화하며 원자적 checkpoint로
재개한다.

텍스트 개발 snapshot은 audio나 model output을 포함하지 않고도 만들고 검증할 수 있다.

```bash
.venv/bin/python experiments/self_repair/scripts/dataset_v2/build_release.py \
  --kind text-development \
  --git-commit <40_HEX_CLEAN_GIT_COMMIT> \
  --output <TEXT_SNAPSHOT_DIRECTORY>
.venv/bin/python experiments/self_repair/scripts/dataset_v2/verify_release.py \
  <TEXT_SNAPSHOT_DIRECTORY>
```

`edge_private_smoke` 결과는 내부 calibration 전용이다. 현재 production 후보는 pinned
`kokoro_local_v1_0` + independent MFA alignment다. Kokoro는 결정적 설정에서 같은 요청을
반복하지 않으므로 source track은 600 targets × 1 candidate로 고정한다. 합성기는 후보마다
manifest를 checkpoint하며 `--resume` 재개를 지원한다.

Voice audition 재현 순서:

```bash
.venv/bin/python experiments/self_repair/scripts/dataset_v2/select_kokoro_voice_calibration.py
.venv/bin/python experiments/self_repair/scripts/dataset_v2/synthesize_candidates.py \
  --provider kokoro_local_v1_0 \
  --targets experiments/self_repair/dataset_v2/calibration/kokoro_voice_targets.jsonl \
  --output experiments/self_repair/dataset_v2/release_evidence/kokoro_voice_raw_candidates.jsonl \
  --audio-root experiments/self_repair/dataset_v2/artifacts/kokoro_voice_raw \
  --attempts 1 --resume
```

8GiB 로컬 host의 첫 일괄 smoke는 audio 10개를 모두 기록한 뒤 aggregate manifest flush 전에
종료됐다. immutable WAV/boundary와 frozen request hash로 private manifest를 명시적으로
recovery했고, 이후 합성기는 후보별 원자 checkpoint 방식으로 보완했다.

8GiB Mac에서는 여러 Kokoro process가 memory compression을 일으켜 오히려 느려질 수 있다.
Production 600개는 기본적으로 단일 process로 실행한다. 부득이하게 shard를 쓸 때는 서로
겹치지 않는 `--start-index`/`--limit` 범위와 별도 manifest를 사용하고, 완료 후 merge
validator가 600 target coverage와 모든 WAV hash를 확인하게 한다.

```bash
.venv/bin/python experiments/self_repair/scripts/dataset_v2/synthesize_candidates.py \
  --provider kokoro_local_v1_0 --attempts 1 --resume

.venv/bin/python experiments/self_repair/scripts/dataset_v2/merge_candidate_manifests.py \
  experiments/self_repair/dataset_v2/manifests/raw_candidates_shard_01.jsonl \
  experiments/self_repair/dataset_v2/manifests/raw_candidates_shard_02.jsonl \
  --output experiments/self_repair/dataset_v2/manifests/raw_candidates.jsonl

.venv/bin/python experiments/self_repair/scripts/dataset_v2/canonicalize_audio.py
.venv/bin/python experiments/self_repair/scripts/dataset_v2/align_from_boundaries.py
.venv/bin/python experiments/self_repair/scripts/dataset_v2/qc_candidates.py
.venv/bin/python experiments/self_repair/scripts/dataset_v2/summarize_production_audio.py
.venv/bin/python experiments/self_repair/scripts/dataset_v2/prepare_mfa_corpus.py
```

2026-08-26 실측은 raw/canonical 각 600개, 자동 QC 600/600 통과, 총 음성 5.756시간,
raw+canonical 1.853GiB다. 조건별 120개, 화자별 60개, 방향별 300개로 균형이다.
`align_from_boundaries.py` 결과는 Kokoro predicted duration을 이용한 extraction seed일 뿐
독립 정렬이 아니다. 이후 로컬 macOS arm64 MFA 2.2.4로 10개 voice를 각각 speaker로
분리해 600/600 TextGrid를 생성·import했고, 재실행한 자동 QC도 600/600 통과했다.
현재 accepted/prepared 수는 0이며 사람 검수 전에는 선택하지 않는다.
`prepare_mfa_corpus.py`는 canonical WAV를 복제하지 않고 speaker별 local hard link와 exact
`.lab` transcript 600쌍을 만들며 input manifest와 hash report를 생성한다.

macOS arm64 로컬 MFA 재현에는 `config/mfa-macos-arm64.yml`을 사용한다. MFA 2.2.4는 긴
PostgreSQL socket 경로에서 실패하므로 `MFA_ROOT_DIR`은 `/tmp/mfa22root`처럼 짧게 둔다.
기본 ARPA 사전에서 빠진 20개 단어형은 frozen G2P 결과
`config/mfa_english_us_arpa_oov.dict`로 보강하며 임의 발음을 추가하지 않는다.

```bash
micromamba create -f experiments/self_repair/dataset_v2/config/mfa-macos-arm64.yml
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa

.venv/bin/python experiments/self_repair/scripts/dataset_v2/build_mfa_dictionary.py \
  --base <MFA_ENGLISH_US_ARPA_DICT> \
  --output experiments/self_repair/dataset_v2/artifacts/mfa_input/english_us_arpa_augmented.dict

mfa align experiments/self_repair/dataset_v2/artifacts/mfa_input/corpus \
  experiments/self_repair/dataset_v2/artifacts/mfa_input/english_us_arpa_augmented.dict \
  english_us_arpa \
  experiments/self_repair/dataset_v2/artifacts/mfa_output/aligned \
  --clean -j 2

.venv/bin/python experiments/self_repair/scripts/dataset_v2/convert_mfa_textgrids.py \
  --tool-version 2.2.4 \
  --model-id <PINNED_ACOUSTIC_DICTIONARY_G2P_HASH_ID> \
  --alignment-run-id <FROZEN_ALIGNMENT_RUN_ID>
.venv/bin/python experiments/self_repair/scripts/dataset_v2/import_independent_alignment.py \
  --external experiments/self_repair/dataset_v2/artifacts/mfa_output/external_alignments.jsonl \
  --tool montreal_forced_aligner --tool-version 2.2.4 \
  --model-id <PINNED_ACOUSTIC_DICTIONARY_G2P_HASH_ID> \
  --alignment-run-id <FROZEN_ALIGNMENT_RUN_ID> --minimum-confidence 0.8
.venv/bin/python experiments/self_repair/scripts/dataset_v2/qc_candidates.py
```

MFA `alignment_analysis.csv`의 log likelihood와 phone-duration deviation은 calibrated
probability가 아니다. 따라서 converter는 이 값을 confidence로 위장하지 않고 600개 모두
사람 검수 대상으로 남긴다. 실제 hash와 분포는 `reports/mfa_local_alignment.json`에 있다.

승인 전에 다음 preflight를 실행하면 외부 호출 없이 정확한 요청 문자 수와 blocker를
확인한다. Authority 원본과 report는 `.gitignore` 대상인 `release_evidence/`에만 둔다.

```bash
cp experiments/self_repair/dataset_v2/config/production_authority.example.json \
  experiments/self_repair/dataset_v2/release_evidence/production_authority.json
# 승인자가 private authority JSON의 pending/빈 값을 직접 검토·기입한다.
.venv/bin/python experiments/self_repair/scripts/dataset_v2/production_preflight.py \
  --allow-blocked
```

현재 frozen Kokoro matrix는 600 requests × 1 deterministic candidate이며 provider API
비용과 credential이 없다. Preflight는 `kokoro==0.9.4`, model/config/10 voice SHA-256,
사람 text·voice 승인, 로컬 공간, RunPod/MFA 접근을 확인한다. Azure billable-character
계산기는 fallback provider 검증용으로만 남아 있다.

## 저장공간 실행 구성

사용자 승인에 따라 production audio와 MFA 정렬은 로컬에서 만들고, Moshi 평가는
RunPod에서 실행한다.

- 로컬: 현재 전체 working artifact 4.8GiB; core raw+canonical은 1.853GiB
- RunPod: Moshiko BF16 model과 3,000 response audio — 최소 40GiB workspace
- 로컬로 회수: manifest, report, annotation package, scored result
- RunPod에 유지: 대용량 response WAV와 model cache

현재 로컬 약 29GiB 여유 공간은 12GiB reserve gate를 통과한다. 따라서 50GiB
로컬 공간이나 별도 로컬 외장 디스크는 core audio 제작에 필수가 아니다.

## 중요한 문서

- `../DATASET_BUILD_PLAN.md`: 전체 단계와 Definition of Done
- `DECISIONS.md`: 고정값과 열린 gate
- `TTS_PROVIDER_REVIEW.md`: provider/권리/정렬 감사
- `ANALYSIS_PROTOCOL.md`: model output을 보기 전에 동결할 분석 규칙
- `ANNOTATION_GUIDE.md`: condition-blind human annotation 규칙
- `DATASET_CARD.md`: 용도, 한계, 안전 주의사항

## Artifact 정책

WAV/MP3, provider response, model output과 private blind map은 Git history에 직접 넣지 않는다.
Release manifest에는 상대 URI, SHA-256, lifecycle lineage와 license scope를 저장한다. Secret,
로컬 절대 경로, 사람 식별정보는 release package에 포함하지 않는다.
