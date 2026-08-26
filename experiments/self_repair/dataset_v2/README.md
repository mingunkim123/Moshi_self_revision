# Moshi self-repair matched-bundle dataset v2

이 디렉터리는 pretrained Moshi의 목적지 자기수정 능력을 평가하기 위한 controlled evaluation
set을 재현 가능하게 만드는 계약·데이터·검증 보고서를 담는다. 일반 학습 corpus가 아니다.

현재 상태는 **text/pipeline development**다. 공개 release용 TTS provider 권리와 Azure
credential, 독립 forced alignment 환경이 승인되기 전에는 audio production 또는
release-ready로 표시하지 않는다. 저장공간은 local TTS/QC와 remote evaluation으로 분리했다.

2026-08-26 현재 30개 blueprint, 300개 script, 60개 answer key와 600개 TTS rendition
target은 생성·검증되었다. 비공개 Edge calibration 180개도 완료했지만 22개가 clipping
QC에서 제외되었고 fast/slow voice의 자연 지연 구간이 겹치지 않았다. 따라서 4초 예시는
폐기하고 production provider에서 화자별 timing을 다시 동결해야 한다.
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

실제 audio 명령은 provider decision과 secret을 소스 코드에 남기지 않은 뒤 실행한다.
`edge_private_smoke` 결과는 내부 calibration 전용이다. 공개 production은 현재
`azure_speech_s0` + independent MFA alignment가 권장안이다.

승인 전에 다음 preflight를 실행하면 외부 호출 없이 정확한 요청 문자 수와 blocker를
확인한다. Authority 원본과 report는 `.gitignore` 대상인 `release_evidence/`에만 둔다.

```bash
cp experiments/self_repair/dataset_v2/config/production_authority.example.json \
  experiments/self_repair/dataset_v2/release_evidence/production_authority.json
# 승인자가 private authority JSON의 pending/빈 값을 직접 검토·기입한다.
.venv/bin/python experiments/self_repair/scripts/dataset_v2/production_preflight.py \
  --allow-blocked
```

현재 frozen matrix의 초기 3-candidate 정책은 1,800 requests와 1,820,700 Azure billable
characters이고, target당 총 5개 hard maximum은 3,000 requests와 3,034,500 characters다.
Azure는 outer `speak`/`voice` tag를 제외한 SSML body의 markup·공백·문장부호도 과금 문자에
포함하므로 transcript 길이만으로 비용을 계산하지 않는다. Preflight는 승인자가 Azure
portal에서 바로 확인한 million-character rate와 budget cap을 입력했을 때만 예산 gate를
통과시킨다. Credential은 존재 여부만 기록하고 값은 report에 저장하지 않는다.

## 저장공간 실행 구성

사용자 승인에 따라 production audio는 로컬에서 만들고, MFA 정렬과 Moshi 평가는
RunPod에서 실행한다.

- 로컬: raw/canonical candidates, accepted, prepared — 초기 약 8GiB, hard maximum 약 12GiB
- RunPod: Moshiko BF16 model, MFA, 3,000 response audio — 최소 40GiB workspace
- 로컬로 회수: manifest, report, annotation package, scored result
- RunPod에 유지: 대용량 response WAV와 model cache

현재 로컬 약 25GiB 여유 공간은 12GiB audio-production gate를 통과한다. 따라서 50GiB
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
