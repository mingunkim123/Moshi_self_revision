# Mechanistic stale-binding harness

이 디렉터리는 `MECHANISTIC_STALE_BINDING_RUNPOD.md`의 `[TARGET]` 명령을 구현한다. 목적은
Moshiko의 main temporal transformer에서 이전 값이 남는 위치와 수정값이 dependency/readout으로
전파되는 경로를 관찰·개입하는 것이다.

## 증거 수준

- `--synthetic`은 코드·통계·resume·패키징 계약만 검사한다. Moshiko 또는 Boston/Seattle에 관한
  실험 증거가 아니다.
- 실제 실행은 고정 revision과 검증된 WAV를 사용하고 `NO_TORCH_COMPILE=1`,
  `NO_CUDA_GRAPH=1`을 모델 생성 전에 설정해야 한다.
- 기존 v2는 exploratory/internal validation 전용이다. 정식 confirmation은 독립 검수된 다중도시
  데이터와 frozen selection이 모두 있을 때만 열린다.

## 로컬 검증

```bash
python3.12 -m venv .venv-mechanistic
.venv-mechanistic/bin/pip install -r experiments/self_repair/requirements-mechanistic.txt
.venv-mechanistic/bin/pip install -e ./moshi --no-deps
.venv-mechanistic/bin/pytest -q experiments/self_repair/mechanistic/tests
experiments/self_repair/mechanistic/runpod/run_local_validation.sh
```

## RunPod smoke

```bash
export MECH_EXPECTED_COMMIT=<40-hex-reviewed-harness-commit>
export PYTHON_BOOTSTRAP=python3.12
experiments/self_repair/mechanistic/runpod/setup.sh

export NO_TORCH_COMPILE=1
export NO_CUDA_GRAPH=1
export MECH_DATA_ROOT=/workspace/moshi/experiments/self_repair/dataset_v2
export MECH_RUN_ROOT=/workspace/mech-artifacts/<identity-derived-run-id>
export MECH_SCAN_SPEC=/workspace/mech-plans/residual-discovery-query_end.json
experiments/self_repair/mechanistic/runpod/runpod_smoke.sh
```

`setup.sh`는 exact clean commit, Python 3.12, pinned requirements와 Moshiko config/revision을
검증한다. 체크포인트를 다운로드하거나 GPU scan을 시작하지 않는다.

이 스크립트는 먼저 checkpoint를 import하지 않는 정적 산술·hash audit을 실행하고, CUDA가 실제로
보일 때에만 최대 4개 입력의 bounded canary를 실행한다. 전체 600개를 자동 Mimi encode하지 않는다.
Canary의 clean/repair 쌍은 `scenario_id`, `direction_id`, `speaker_id`, current/new value가 모두
같아야 하며, 가능한 경우 `clean_final` + `delayed_three_dependencies`를 우선한다. 나머지 canary
행도 이 matched group 밖에서 채우지 않는다.
Canary는 peak/total VRAM, activation byte 수, cell당 초, model-frame당 초, 전체 grid의 두 ETA와
보수적 저장공간 예약량을 JSON으로 남긴다. GPU/checkpoint가 없거나 대화 canary가 없으면 exit 3
(`NO_GO`)로 종료하며 유료 scan을 열지 않는다.

## 유료 scan readiness 계약

`scan-spec`은 산술 grid와 실제 CLI를 함께 고정한다. 최상위 `execution`은 `kind`, `role`,
`layers`, `anchors`, `donors`, `controls`, `components`, `limit_scenarios`, `selection_sha256`를
정확히 포함해야 한다. `scans`의 `expected_cell_count`는 manifest selector에서 계산되는 값과
일치해야 하며, `--include-generation`을 사용한 spec의
`generation.startup_modes`는 config의 두 required mode를 모두 포함한다.
`scans[0].donor_arms`는 generic active-arm 차원이다. residual/KV/path에서는
`execution.donors`, component에서는 `execution.controls`와 순서까지 같아야 하며 inactive CLI
default는 비용에 더하지 않는다. Layer/anchor/component 목록도 두 블록 사이에서 다르면 정적 단계가
중단된다.
common-handshake arm은 관측된 짧은 greeting이 아니라 `natural_max_ms` 전체를 비용으로 예약한다.
Conversation capture가 있으면 prepared WAV 길이가 아닌
`conversation_contract.target_end_frame_count`까지 replay/storage를 계산한다.

전체 scan을 열려면 다음 네 보고서와 full encoded manifest가 필요하다.

- exact model contract 및 그 `run_identity.json`
- strict open-loop report
- required startup mode별 conversation canary와 사람 자연스러움 검수
- bounded real-GPU canary measurements

`conversation_canary.json`에는 모드마다 `trial_count`, `truncated_count`, `cap_active_count`,
`exact_output_coverage_count`, `response_complete_count`, `text_tail_checked_count`,
`audio_tail_checked_count`, `human_flow_review_pass_count`를 기록한다. 모든 trial의 text/audio tail과
사람 흐름 검수가 완료되고 truncation/cap-active가 0이어야 한다. `tail_detection`은 policy
version과 calibration rule까지 기록한다. Readiness는 `policy_sha256` 필드를 제외한 canonical policy
내용을 다시 hash하고 frozen digest와 대조한다. 실제 GPU에서 decode한 forced-silence의 최대 dBFS가 -45 dBFS
threshold보다 낮다는 측정값이 있어야 한다. 이 측정 전에는 threshold를 “calibrated”라고 부르지 않는다.

```bash
# 예: discovery query_end residual stage를 checkpoint load 전에 동결한다.
.venv/bin/python experiments/self_repair/mechanistic/scripts/freeze_paid_scan_spec.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --kind residual --role discovery --layers 0:32 --anchors query_end \
  --donors clean_current --controls self,clean_current,clean_stale,shuffled \
  --components resid_post --full-replays-per-cell 3 \
  --readout-steps-per-cell 128 --output "$MECH_SCAN_SPEC"

# 정확한 정적 비용(모델을 load하지 않음)
.venv/bin/python experiments/self_repair/mechanistic/scripts/estimate_mechanistic_workload.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --scan-spec "$MECH_SCAN_SPEC" \
  --output "$MECH_RUN_ROOT/preflight/workload_estimate.json"

# 네 evidence를 현재 target identity에 묶고 GO/NO_GO를 발행한다.
.venv/bin/python experiments/self_repair/mechanistic/scripts/assemble_readiness_evidence.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --scan-spec "$MECH_SCAN_SPEC" \
  --model-contract "$MECH_RUN_ROOT/preflight/model_contract/model_contract.json" \
  --model-run-identity "$MECH_RUN_ROOT/preflight/model_contract/run_identity.json" \
  --open-loop "$MECH_RUN_ROOT/gpu_canary/open_loop_validation.json" \
  --conversation-canary "$MECH_RUN_ROOT/gpu_canary/conversation/conversation_canary.json" \
  --gpu-canary "$MECH_RUN_ROOT/gpu_canary/gpu_measurements.json" \
  --canary-manifest "$MECH_RUN_ROOT/gpu_canary/canary_trials.jsonl" \
  --canary-encoded-manifest "$MECH_RUN_ROOT/gpu_canary/encoded_manifest.jsonl" \
  --output "$MECH_RUN_ROOT/preflight/readiness_evidence.json"

.venv/bin/python experiments/self_repair/mechanistic/scripts/assess_mechanistic_readiness.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --scan-spec "$MECH_SCAN_SPEC" \
  --evidence "$MECH_RUN_ROOT/preflight/readiness_evidence.json" \
  --output "$MECH_RUN_ROOT/preflight/paid_scan_authorization.json"
```

발행 파일은 model repo/revision, Git commit, source+encoded data, manifest, config, scan-spec,
evidence와 assessment를 SHA-256으로 묶는다. 실제 non-synthetic scan은 checkpoint 생성 전에 이를
다시 검증하며, 다음 두 인자가 없거나 하나라도 달라지면 중단한다.

```text
--scan-spec "$MECH_SCAN_SPEC"
--readiness-go "$MECH_RUN_ROOT/preflight/paid_scan_authorization.json"
```

`--resume`은 canonical cell identity를 먼저 확인한다. 이미 끝난 cell만 있는 경우 checkpoint조차
load하지 않으며, 일부만 남았어도 기존 cell에 대한 donor/recipient replay는 실행하지 않는다.

## 대화·probe·확증 계약

Full-duplex canary는 `common_handshake_then_request`와 `greeting_suppressed` 두 mode를 모두
실행한다. 첫 mode는 Moshiko의 실제 첫 인사가 terminal punctuation으로 끝난 뒤 20 frames/1.6초
quiet가 확인될 때까지 text/audio를 보존하고, 6 frames/480 ms prepared lead-in과 user WAV를 같은
continuous Mimi stream에 이어 붙인다. `natural_model_start`는 greeting-confounded diagnostic-only다.
두 required mode 모두
frame 0부터 user end 후 40초까지 저장하며, tail 계약과 두 사람의 독립 자연스러움
검수가 끝나기 전에는 real `conversation_canary.json`을 발행하지 않는다. 첫 pass 상태는
`awaiting_double_blind_human_review`이며, reviewer 2명의 판정이 충돌하면 두 reviewer와 다른
독립 adjudicator가 필요하다. Synthetic 실행의 `synthetic_conversation_canary.json`은 real
readiness evidence로 인정되지 않는다.

Probe grid는 `fit_probes.py --probe-grid --sites ... --layers ... --anchors ...`로 exact
site×layer×semantic-anchor tensor를 각각 학습한다. 그리드 점수는 진단용이며 causal
결론으로 쓰지 않는다. 각 cell은 그 exact tensor 하나만 flatten하며 layer/time 평균을 내지 않고,
scenario-grouped CV를 사용한다. Frozen probe는 이미 동결된 discovery selection과 exact
coordinate가 일치하고 non-head feature일 때만 만든다. Probe 성능으로 site를 고르거나
internal/formal role에서 refit하지 않는다.

Empirical confirmation은 먼저 `run_confirmatory_patches.py --plan-only`로 pristine
`planned_cells.jsonl`을 만든다. 그 상태에서 `freeze_analysis_plan.py`가 selection, config,
cell universe와 bootstrap/Holm/SESOI 정의를 self-hashed spec으로 묶는다. Internal validation은
`internal_analysis_plan.template.json`, formal confirmation은 `analysis_plan.template.json`을
각자 결과 전에 동결하고 서로 다른 report directory에 보존한다. 결과를 쓴 뒤에는
`analyze_mechanistic_results.py --config ... --analysis-spec ... --expected-cells ...`만 허용되며,
missing/extra/failed/relabelled cell이 하나라도 있으면 중단한다.

로컬 validation script는 paid-readiness 승인, 실제 full-duplex 대화, empirical probe grid,
empirical analysis freeze 또는 release packaging을 실행하지 않는다. 이 경로들은 unit/integration
fixture로 검증되며, 실제 checkpoint·GPU·사람 검수 evidence가 갖춰진 RunPod에서만 열린다.

실제 스캔 순서, gate와 해석 제한은 상위
`experiments/self_repair/MECHANISTIC_STALE_BINDING_RUNPOD.md`를 따른다.
