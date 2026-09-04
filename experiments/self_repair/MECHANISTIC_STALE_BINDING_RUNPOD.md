# Moshi mechanistic stale-binding experiment: RunPod end-to-end runbook

- Status: **구현 완료·로컬 검증 완료, 실제 GPU/사람 검수 대기**
- Starting code baseline: `55c6a4456a084fb4f836bbf6eab5797e8a8ee5b0`
- Target repository/branch: `https://github.com/mingunkim123/Moshi_self_revision`
  (`codex/mechanistic-stale-binding-harness`)
- Pinned model for comparison with the existing pilot:
  `kyutai/moshiko-pytorch-bf16@2bfc9ae6e89079a5cc7ed2a68436010d91a3d289`

## 먼저 읽을 결론

이 실험은 **모델 구조상 가능**하다. Moshiko의 main Temporal Transformer는 32개 layer,
layer당 32개 attention head, hidden size 4,096이고 layer residual과 streaming KV cache에
접근할 수 있다. 이 branch에는 activation capture, head/KV/path intervention, strict
teacher-forced open-loop, probe, 보고서와 패키징 하네스가 구현돼 있다. 로컬 toy/synthetic
검증은 완료됐지만, 실제 Moshiko checkpoint 결과는 아직 만들지 않았다.

아래 두 종류의 명령을 구분한다.

- **`[EXISTS]`**: 현재 저장소에 존재하고 실행할 수 있다.
- **`[TARGET]`**: 이 branch에 구현된 mechanistic harness 명령이다. exact commit을 checkout한
  RunPod에서 실행한다.

가장 중요한 세 가지 제한도 먼저 고정한다.

1. 기존 v2는 Boston↔Seattle 두 값만 사용한다. old와 current가 완전히 반대이므로 이
   데이터만으로는 일반적인 `old value`와 `current value` 표현을 분리할 수 없다. v2는
   engineering smoke와 exploratory localization에만 쓰고, 강한 주장은 최소 4개 도시의
   fully-crossed control set에서 확인한다.
2. 현재 600개 음성은 자동 QC와 MFA를 통과했지만 사람 검수 기록이 빠진
   `release_eligible=false` provisional artifact다. confirmatory/publication 결론에 쓰지 않는다.
3. seed 17 behavioral pilot에서는 assistant가 사용자 음성보다 먼저 발화한 비율이 100%였고,
   primary window에서 target-only 증거는 3.7%뿐이었다. 현재 설정으로 남은 2,400개를 바로
   생성하지 않는다. 먼저 assistant feedback을 고정한 open-loop diagnostic을 통과시킨다.

## 완료의 정의

“17번째 layer의 특정 head가 Boston을 기억한다”는 문장은 probe 하나로 성립하지 않는다.
이 런북에서 완료란 다음 증거 사슬을 모두 남긴 상태다.

1. 동일 입력의 replay, self-patch, cache reset이 정상인 계측 harness가 있다.
2. clean과 repair의 final-current readout 차이가 재현된다.
3. discovery split에서 layer/time을 넓게 찾고, 후보 head/path를 좁힌다.
4. 잠가 둔 v2 internal validation과 새 formal-confirmation split에서 target donor의 rescue와
   stale donor의 corruption이 모두 사전 지정 방향으로 나타난다.
5. wrong-destination(clean-stale), same-value, shuffled-donor, no-op 같은 control로 일반적인 activation 손상을
   배제한다.
6. 최소 4개 도시의 crossed set에서도 old/current 역할이 분리된다.
7. 최종 후보만 정상 full-duplex generation에서 재검증하고, 들리는 발화까지 좋아지는지
   별도로 보고한다.
8. 실행 identity, raw scalar 결과, 통계, 그림, 실패 row, artifact hash를 보존한다.

Probe만 통과하면 “정보가 선형적으로 decode된다”, causal patch가 통과하면 “이 readout에
인과적으로 기여한다”, full-duplex behavior까지 통과하면 비로소 “이 task/distribution의
spoken stale-binding behavior에 기여한다”고 쓴다.

## 전체 흐름과 중단 게이트

| 단계 | 핵심 산출물 | GO 조건 | 실패하면 |
|---|---|---|---|
| 0. 설계 동결 | config, split, metric SHA | identity가 모두 40자리 SHA/hash로 고정 | 실행 금지 |
| 1. harness 구현 | 계측·open-loop·분석 코드와 tests | 모든 contract test 통과 | 구현 수정 |
| 2. RunPod 준비 | environment report | GPU/VRAM/disk/model shape 통과 | Pod/환경 교체 |
| 3. artifact 검증 | input hash manifest | 600/600 WAV와 manifest 검증 | 재업로드 |
| 4. Mimi cache | encoded user codes | 재인코딩 hash 일치 | codec/reset 수정 |
| 5. open-loop | replay/no-op report | feedback 동일, replay 안정, self-patch ≈ 0 | causal scan 금지 |
| 6. readout | baseline margins | clean 자격시험과 repair gap 통과 | readout/자극 수정 |
| 7. discovery | residual/probe/head/KV/path maps | control 대비 일관된 후보 | null result 보고 |
| 8. selection freeze | frozen candidate JSON + SHA | layer/head/window/가설 family 잠금 | validation 열람 금지 |
| 9. v2 internal validation | 12-scenario effects/CI/p-values | frozen 4-DID/Holm/SESOI gate | exploratory로 제한 |
| 10. formal multivalue confirmation | held-out ordered-pair 결과 | old/current 독립 조작에서 재현 | Boston–Seattle 축으로 제한 |
| 11. full-duplex bridge | WAV/text/human labels | audible behavior도 개선 | logit surrogate로 제한 |
| 12. 보고·보관 | Markdown, SVG, CSV/JSON, hashes | verifier와 checksum 통과 | package 금지 |

---

## Step 0. 연구 질문과 metric을 실행 전에 동결한다

### 0.1 무엇을 stale binding으로 부를지

Causal Transformer에서는 correction 뒤의 Seattle 정보가 과거 Boston frame의 hidden state를
소급해 바꿀 수 없다. KV cache에 Boston 정보가 남는 것도 정상이다. 진단 대상은 다음이다.

> repair 이후의 dependency/query state에서 retracted value가 current-value readout에
> 과도하게 사용되는가, 그리고 특정 layer/head/path 개입이 그 사용을 인과적으로 바꾸는가?

따라서 “Boston이 decode된다”나 “Boston attention weight가 높다”만으로 stale binding이라고
판정하지 않는다.

### 0.2 primary readout

Sampling 전 float32 text logits로 candidate sequence 전체의 길이 정규화 margin을 계산한다.

```text
M = mean_j log p(target_token_j | fixed_prefix, target_<j)
  - mean_j log p(stale_token_j  | fixed_prefix, stale_<j)
```

- SentencePiece의 leading-space 처리와 canonical verbalizer를 config에 고정한다.
- Primary는 canonical city name 하나만 사용한다. Alias sensitivity를 추가하면 도시마다 같은
  개수의 frozen alias를 쓰고 `logmeanexp`로 묶는다. Raw `logsumexp`로 alias 수가 많은 도시를
  보상하지 않는다.
- root destination과 D1/D2/D3 dependency readout을 분리한다.
- target/stale 후보의 token 수가 다르면 nats/token margin을 primary로, summed log likelihood를
  sensitivity analysis로 보고한다.
- sampled token, temperature/top-k 적용 뒤 logits, 첫 token 하나만으로 판정하지 않는다.

Patch primary effect는 raw nats/token으로 기록한다.

```text
delta_rescue(S) = M(repair <- clean-current at site S) - M(repair)
delta_stale(S)  = M(repair <- clean-stale   at site S) - M(repair)
DID(S)          = delta_rescue(S) - delta_stale(S)
```

`DID`를 value-specific confirmatory primary estimand로 둔다. Rescue와 stale-induction은 각각
기술적으로 함께 공개하지만 현재 tracked analysis template은 둘을 별도 pass/fail gate로 판정하지
않는다. 이것만으로는 repair run 안에 있던 endogenous stale state를 찾았다고 할 수 없다. 더 강한
주장을 하려면 다음 중 하나를 **별도 preregistered intervention/analysis extension**으로 구현하고
discovery에서 동결해야 한다.

```text
E_transfer(S) = M(clean-current <- repair at S)
              - M(clean-current <- mention-only at S)

E_erase(S)    = M(repair with frozen old-subspace/path erasure at S) - M(repair)
```

`E_transfer < 0` 또는 `E_erase > 0`가 matched cue/mention control보다 사전 동결된 분석에서
재현돼야 “stale binding의 유지/readout에 기여”라는 표현을 쓴다. 현재 네 DID template만 실행한
결과는 이 조건을 충족했다고 간주하지 않으며, “이 site의 destination-value intervention이 readout을
인과적으로 바꾼다”로 제한한다.

Normalized recovery는 보조 지표다.

```text
NR = delta_rescue / (M(clean-current) - M(repair))
```

분모가 사전 지정 tolerance보다 작은 pair는 NR에서 제외하되 raw effect에서는 빼지 않고,
NR을 0–1로 clamp하지 않는다.

### 0.3 split과 통계 단위

기존 [`analysis_folds.jsonl`](dataset_v2/assignments/analysis_folds.jsonl)을 mechanistic 모델
선택용으로 다음처럼 사용한다. 기존 파일의 `inferential_role` 자체를 변경하지 않는다.

- folds 1–3: v2 mechanistic discovery, 18 scenarios
- folds 4–5: v2 internal mechanistic validation, 12 scenarios
- 한 scenario의 모든 condition, direction, speaker, frame은 같은 split에 둔다.
- frame, direction, speaker, generation seed를 독립 표본 수로 세지 않는다.
- scenario가 primary cluster다. scenario 안의 direction×speaker는 동일 가중 평균한다.
- 95% CI는 scenario-cluster bootstrap 10,000회, seed `20260826`으로 계산한다.
- confirmation에서는 frozen `DID`, rescue, stale-induction, endogenous test를 하나의 명시된
  family로 두고 two-sided p-value를 Holm 보정한 뒤 예상 부호를 별도로 요구한다.
- 각 endpoint의 `hypothesis_id`, `family_id`, 방향, statistic, scenario 수, raw/adjusted p,
  CI 종류, SESOI, pass/fail을 machine-readable table에 남긴다.
- 12개 v2 internal-validation scenario의 exhaustive sign-flip은 scenario effect의 교환가능성/대칭성
  가정이 필요한 sensitivity analysis다. all-scenario effects와 cluster CI를 primary uncertainty로
  보고하고 sign-flip을 무조건 “exact test”라고 부르지 않는다.

Seed-17 pilot에서 이미 30개 scenario의 behavioral output을 분석했으므로 folds 4–5를 완전히
미관측인 confirmatory data라고 부르지 않는다. 다만 activation/site selection에는 쓰지 않고
internal mechanistic validation으로 잠가 둔다. Formal confirmation은 Step 17의 새로 검수된
multivalue/scenario/pair role manifest에서 수행한다. 현재 fold는 speaker holdout도 아니므로 별도
speaker-held-out split 없이 “새 speaker에 일반화”한다고 쓰지 않는다.

### 0.4 실행 identity

각 run은 아래 항목을 `run_identity.json`에 기록한다.

- 40자리 clean Git commit
- model repository와 resolved 40자리 revision
- tokenizer/Mimi/model file identity
- experiment config, analysis protocol, manifest, readout config SHA-256
- PyTorch/CUDA/cuDNN/SDPA backend, GPU 이름과 VRAM
- dtype, deterministic flags, random seeds
- open-loop feedback policy와 semantic-frame alignment policy
- instrumentation schema/version 및 modified-source hashes

Git working tree가 dirty이거나 requested model revision과 resolved revision이 다르면 STOP한다.

---

## Step 1. 구현된 mechanistic harness를 로컬에서 검증한다 `[TARGET]`

### 1.1 Definition of Ready 파일 구조

다음 파일들은 이 branch에 구현돼 있다. 이후 모든 `[TARGET]` 명령은 이 구조가 포함된 exact
commit을 전제로 한다.

```text
experiments/self_repair/mechanistic/
  README.md
  config/mechanistic.json
  config/readouts.json
  config/multivalue_cities.json
  manifests/
  reports/
  results/
  scripts/
    build_mech_manifest.py
    build_anchor_map.py
    build_multivalue_controls.py
    simulate_multivalue_power.py
    validate_multivalue_controls.py
    encode_user_audio.py
    validate_mechanistic_contract.py
    validate_open_loop.py
    estimate_mechanistic_workload.py
    select_gpu_canary_manifest.py
    run_bounded_gpu_canary.py
    assemble_readiness_evidence.py
    assess_mechanistic_readiness.py
    verify_paid_scan_authorization.py
    capture_activations.py
    score_readouts.py
    fit_probes.py
    scan_residual_patches.py
    scan_component_patches.py
    scan_kv_patches.py
    run_path_patches.py
    freeze_mechanistic_selection.py
    run_confirmatory_patches.py
    run_full_duplex_validation.py
    analyze_mechanistic_results.py
    render_mechanistic_report.py
    verify_mechanistic_run.py
    package_mechanistic_results.py
  tests/
  runpod/setup.sh
  runpod/runpod_smoke.sh
experiments/self_repair/requirements-mechanistic.txt
```

`results/`, raw activations, WAV, and private blind maps는 `.gitignore`에 추가하고, 작은 schema,
config, report, CSV/JSON summary, SVG만 Git에 넣는다.

### 1.2 필요한 모델 계측 지점

Main Temporal Transformer부터 계측한다. Depth Transformer는 audio codebook의 같은-frame 생성용이므로
첫 localization 대상에서 제외한다.

- layer별 `resid_pre`, `attn_out`, `resid_mid`, `mlp_out`, `resid_post`
- attention SDPA 결과이자 output projection 전의 head별 `z: [B,H,T,128]`
- q/k/v의 pre-RoPE와 post-RoPE 값
- KV ring cache의 tensor와 logical absolute-position map
- sampling 전 text logits

Layer residual은 일반 hook으로 잡을 수 있지만, 현재 attention 구현은 head 출력을 즉시 concat하고
`W_O`를 적용한다. head patch/ablation에는 SDPA 뒤, concat 전 callback seam을 추가해야 한다.
기본 callback은 no-op이어야 하며 production inference 결과를 바꾸면 안 된다.

현재 `LMGen._step()`은 `@torch.no_grad()`이고 sampled assistant text/audio를 다음 frame 입력에
다시 쓴다. 다음 API를 분리 구현한다.

- 계산된 logits/token과 feedback에 쓸 token을 분리
- assistant text feedback을 null/PAD 또는 frozen common trace로 강제
- assistant audio feedback도 paired conditions에서 byte-identical하게 강제
- gradient attribution용 별도 full-sequence/eager path
- mutable LM/KV/cache/offset state의 tensor-level deep clone/restore

### 1.3 최소 contract tests

RunPod full scan 전에 작은 fixture로 모두 통과해야 한다.

- hook을 끈 eager 결과와 기존 runner 결과가 사전 tolerance 안에서 일치
- hook 호출 수, layer/head index, tensor shape/dtype/device가 예상과 일치
- 동일 input을 두 번 실행한 logits와 saved activation이 일치
- `receiver <- receiver` self-patch와 identity hook가 baseline을 바꾸지 않음
- branch point의 residual과 필요한 streaming/KV state 전체를 이식한 full-state transplant가
  donor suffix/readout을 재현; 한 layer/frame의 부분 patch에는 이 조건을 요구하지 않음
- 지정하지 않은 layer/frame/cache 값은 바뀌지 않음
- trial reset 뒤 첫 trial 재실행 결과가 동일
- KV ring slot ↔ absolute frame mapping이 wrap 전후 모두 맞음
- 중단 후 resume가 completed cell을 중복 실행하지 않음
- NaN/OOM/validation error도 failure row로 원자적으로 남음

Instrumentation에서는 모델을 만들기 **전에** 두 환경 변수를 모두 설정한다.

```bash
export NO_TORCH_COMPILE=1
export NO_CUDA_GRAPH=1
```

CUDA graph가 켜져 있으면 Python hook가 한 번만 capture되거나 stale buffer를 볼 수 있다.

---

## Step 2. confirmatory용 control 자극을 준비한다 `[TARGET]`

### 2.1 기존 v2를 어디까지 쓸 수 있는가

현재 v2는 30 scenarios × 2 directions × 5 conditions × 2 speakers = 600 inputs이고,
5 generation seeds를 쓰면 3,000 behavioral trials다. semantic timing과 D1–D3 dependency가 있어
두 도시 smoke에는 유용하다.

그러나 confirmatory stale/current 분리를 위해 다음 control을 같은 scenario, voice, closing suffix,
가능하면 같은 absolute frame 길이로 추가한다.

- `clean_current`: new city만 active
- `clean_stale`: old city만 active
- `repair_old_to_new`: old가 명시적으로 retract됨
- `mention_only`: old를 과거 후보로 언급하지만 current는 new인 position/length-matched control
- `cue_length_control`: “actually/no”와 길이는 같지만 값 변경이 없는 control
- same-new/different-old donor
- same-old/different-new donor

최소 4개 도시로 ordered old→new pair를 fully cross하고, 최소 2 speakers를 쓴다. 도시별
token 길이와 prior를 확인하고 value-pair holdout을 둔다. 기존 2-city 데이터만 끝냈다면 결과
표현은 “Boston–Seattle target axis”로 제한한다.

도시 수가 `K`이면 repair pair는 scenario/condition/speaker마다 `K × (K - 1)`개다. `K=4`이면
12 ordered pairs다. Clean-current `K`개와 mention/cue controls의 exact row 수를 config에서 계산해
expected-count gate로 고정하고, 빠진 pair를 사후 가중치로 보정하지 않는다.

### 2.2 사람 검수 gate

새 audio는 기존 v2와 동일하게 canonicalization, independent alignment, 자동 QC, condition-blind
사람 이중 청취 기록을 거친다. manifest에 pass/fail, reviewer ID, timestamp, WAV SHA-256이 없으면
engineering run은 가능해도 confirmatory run은 금지한다.

`build_multivalue_controls.py`는 city 목록과 ordered-pair matrix를 만들고, 기존 v2의 script/audio
production 단계를 일반화해 source WAV와 timing manifest를 생성해야 한다. 실제 city 목록은
clean recognition과 tokenizer-length screen을 통과한 뒤 `multivalue_cities.json`에 고정한다.
Canonical outputs는 `/workspace/multivalue-controls/prepared_stimuli.jsonl`,
`role_manifest.jsonl`, `audio/`로 고정한다.
`validate_multivalue_controls.py`는 value/pair/condition/speaker coverage, lexical leakage, timing,
audio hash, review record를 fail-closed로 검사한다. 이 두 `[TARGET]` 도구가 없거나 사람 검수
gate가 열려 있으면 Step 17을 실행할 준비가 되지 않은 것이다.

새 role manifest는 scenario template 묶음과 ordered city-pair 묶음을 discovery/formal-confirmation
사이에 함께 격리한다. 같은 WAV, paraphrase template, donor, ordered pair가 양쪽에 나타나면 안 된다.
기존 `analysis_folds.jsonl`은 이 새 데이터의 split을 대신하지 않는다.

City clean-recognition/tokenization screen은 multivalue discovery/formal data에 절대 들어가지 않는
별도 calibration scenarios와 voices에서만 한다. City eligibility, role manifest, expected counts,
class coverage를 repair outcome을 보기 전에 freeze한다. Validator는 모든 retained city가
calibration과 formal partition에서 old와 new 역할로 모두 나타나는지 확인한다. 따라서 일반화
주장도 이 사전 screening을 통과한 city population에 조건부다.

`K≥4`, speaker 2명만으로 power가 보장되지는 않는다. Scenario가 독립 cluster이므로 생성 전에
formal scenario-cluster 수, ordered-pair allocation, 예상 ICC, SESOI/target MDE, 목표 power를
config에 고정하고 simulation을 실행한다. 미달이면 row 반복이 아니라 독립 scenario 수를 늘린다.

다음 명령의 실제 실행 시점은 Step 4의 RunPod environment 설치 뒤다. 실행 순서는
`city/split freeze → power → build → 사람 청취 → validate`이며 사람 청취 review는 자동화하지
않는다.

```bash
# [TARGET] city 목록과 role split을 config에 동결한 뒤, audio 생성 전에 실행한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/simulate_multivalue_power.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --city-config experiments/self_repair/mechanistic/config/multivalue_cities.json \
  --output /workspace/multivalue-controls/power_design.json

sha256sum /workspace/multivalue-controls/power_design.json

# Power gate가 통과한 뒤 recording script/target/review template를 먼저 만든다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/build_multivalue_controls.py \
  --city-config experiments/self_repair/mechanistic/config/multivalue_cities.json \
  --scenario-blueprints experiments/self_repair/dataset_v2/blueprints/scenarios.jsonl \
  --output-root /workspace/multivalue-controls

# recording_targets.csv에 맞춰 WAV를 만들고 independent timing.jsonl과 reviews.jsonl을 작성한다.
# 새 음성 생성은 이 harness/local-validation 범위 밖이며 자동으로 통과 처리되지 않는다.
# 모든 WAV/timing/review가 준비된 뒤 builder를 같은 인자로 다시 실행해 prepared_stimuli.jsonl을 만든다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/build_multivalue_controls.py \
  --city-config experiments/self_repair/mechanistic/config/multivalue_cities.json \
  --scenario-blueprints experiments/self_repair/dataset_v2/blueprints/scenarios.jsonl \
  --output-root /workspace/multivalue-controls

# 마지막으로 intervention-blind 사람 이중 청취와 불일치 조정까지 fail-closed 검증한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/validate_multivalue_controls.py \
  --input-root /workspace/multivalue-controls \
  --require-independent-alignment \
  --require-double-listen-review
```

첫 builder 호출의 정상 상태는 `awaiting_audio_alignment_and_human_review`다. 두 번째 호출은 정확한
mono 24 kHz PCM16, Mimi-frame alignment, WAV hash, 독립 timing과 통과한 review가 모두 있을 때만
`reviewed_audio_materialized`가 되며 `prepared_stimuli.jsonl`을 쓴다. 현재 tracked city config는
모든 후보를 `pending`/`eligible=false`로 두므로 별도 clean-recognition screen 기록이 없으면 첫
호출부터 의도적으로 중단한다.

---

## Step 3. RunPod Pod와 persistent storage를 준비한다

권장 사양은 A100/H100 80 GB다. A40/A6000/L40S 48 GB도 batch 1과 선택적 capture로 가능하지만
full QKV/gradient scan은 더 자주 offload/OOM된다. 기존 behavioral inference의 24 GB 요구량을
mechanistic full scan 요구량으로 착각하지 않는다.

- Persistent volume: 최소 100 GB, 권장 150–200 GB
- Volume mount: `/workspace`
- Container: 지원되는 CUDA/PyTorch 이미지이며 `python3.12`가 설치된 환경
- SSH 또는 RunPod web terminal 활성화
- HF token이 필요한 경우 environment secret로만 주입; shell history/config/Git에 쓰지 않음

Dense residual을 모두 저장할 때 대략적인 BF16 용량은 다음과 같다.

```text
bytes ≈ 32 layers × frames × 4096 × 2
```

816 frames면 trial당 약 214 MB, 600 trials면 약 128 GB다. 그래서 모든 trial의 full trace를
저장하지 않는다. 모든 layer에서는 사전 지정 anchor frame만 저장하고, head/QKV는 후보 layer만,
patch run에서는 scalar metric을 즉시 기록한다.

Pod를 만들고 다음을 확인한다.

```bash
nvidia-smi
df -h /workspace
python3 --version
```

80 GB GPU가 없어도 smoke는 가능하지만, pilot에서 peak VRAM과 cell/sec를 측정한 뒤 full ETA와
disk budget을 다시 산정한다.

---

## Step 4. 저장소를 clone하고 exact environment를 만든다

이 문서를 작성한 로컬 working copy에서는 `origin`이 Kyutai upstream이고 `self-revision`이
실험 fork다. 아래처럼 fork URL을 직접 clone한 RunPod checkout에서는 그 fork가 `origin`이 된다.

재사용 Pod에 `/workspace/moshi`가 이미 있으면 그 위에 clone하지 않는다. 기존 directory의 remote,
HEAD, dirty state를 검증해 그대로 쓰거나, 새 빈 directory를 선택한다.

```bash
cd /workspace
git clone https://github.com/mingunkim123/Moshi_self_revision.git moshi
cd /workspace/moshi
git remote -v

# mechanistic harness를 구현·검증한 정확한 40자리 commit으로 바꾼다.
export MECH_CODE_COMMIT="<MECHANISTIC_COMMIT_40_HEX>"
git fetch origin
git checkout --detach "$MECH_CODE_COMMIT"
test "$(git rev-parse HEAD)" = "$MECH_CODE_COMMIT"

export MECH_REPO_ROOT=/workspace/moshi
export MECH_DATA_ROOT=/workspace/moshi/experiments/self_repair/dataset_v2
export MECH_RUN_ID="<IDENTITY_DERIVED_RUN_ID>"
export MECH_RUN_ROOT="/workspace/mech-artifacts/$MECH_RUN_ID"
export HF_HOME=/workspace/hf-cache
export NO_TORCH_COMPILE=1
export NO_CUDA_GRAPH=1

mkdir -p "$MECH_RUN_ROOT"
test -z "$(git status --porcelain --untracked-files=no)"
git rev-parse HEAD
```

Clone한 저장소에서 remote 이름은 보통 `origin`이지만 URL은 이 fork다. 출력 SHA를 run ID에
기록하고, mechanistic harness가 들어간 정확한 commit인지 확인한다.

환경을 만든다.

```bash
cd "$MECH_REPO_ROOT"
export MECH_EXPECTED_COMMIT="$MECH_CODE_COMMIT"
export PYTHON_BOOTSTRAP=python3.12
experiments/self_repair/mechanistic/runpod/setup.sh
```

실행 중 package를 임의로 더 설치하지 않는다. 필요 dependency는 requirements에 version을
고정하고 새 clean commit에서 다시 시작한다.

---

## Step 5. Git에 없는 audio/manifest를 RunPod로 옮긴다

`.gitignore`가 `dataset_v2/artifacts`, `manifests`, `evaluation`을 제외하므로 GitHub clone만으로는
v2 실험을 실행할 수 없다. 로컬에 있는 아래 archive를 RunPod `/workspace`에 upload한다.

| 파일 | 용도 | SHA-256 |
|---|---|---|
| `moshi_v2_provisional_runpod_payload_b8d30fd.tar.gz` | 600 prepared WAV + old manifest/config | `0227f0935556ea938c81234c0895d5e307e160efa5ae53655eb126c3eb95ce8d` |
| `moshi_v2_seed17_pilot_b977391.tar.gz` | corrected 3,000-row manifest | `b3827212937f75dc9d37b250fa5aa3fce133b0493177cdaa2abd5277a4bab23b` |
| `dataset_v2/manifests/provisional_prepared_stimuli.jsonl` | unit spans, MFA provenance, prefix metadata | `55a66b4fae34f64de4a9488d9c6358fc77ca105f7c9b237d52b5a70f67dab8c4` |
| `moshi_v2_seed17_results_b977391.tar.gz` | 기존 seed-17 비교 결과, 선택 사항 | `79f5baa57b8cf84b363a49949b5fa0448a5b2905b3434967a4f10290b209646a` |

세 번째 metadata manifest는 기존 두 archive에 들어 있지 않지만 D1/D2/D3 anchor를 만드는 데
필수다. 예를 들어 SSH가 열려 있으면 **로컬 repository root**에서 다음처럼 전송한다. RunPod가
일반적인 custom SSH port를 제공한다고 가정한 예시다.

```bash
scp -P <SSH_PORT> moshi_v2_provisional_runpod_payload_b8d30fd.tar.gz \
  root@<SSH_HOST>:/workspace/
scp -P <SSH_PORT> moshi_v2_seed17_pilot_b977391.tar.gz \
  root@<SSH_HOST>:/workspace/
scp -P <SSH_PORT> \
  experiments/self_repair/dataset_v2/manifests/provisional_prepared_stimuli.jsonl \
  root@<SSH_HOST>:/workspace/
```

RunPod에서 hash를 직접 비교한다.

```bash
cd /workspace
echo "0227f0935556ea938c81234c0895d5e307e160efa5ae53655eb126c3eb95ce8d  moshi_v2_provisional_runpod_payload_b8d30fd.tar.gz" \
  | sha256sum --check -
echo "b3827212937f75dc9d37b250fa5aa3fce133b0493177cdaa2abd5277a4bab23b  moshi_v2_seed17_pilot_b977391.tar.gz" \
  | sha256sum --check -
echo "55a66b4fae34f64de4a9488d9c6358fc77ca105f7c9b237d52b5a70f67dab8c4  provisional_prepared_stimuli.jsonl" \
  | sha256sum --check -
```

기존 archive의 tracked config를 새 checkout 위에 덮어쓰지 않도록 임시 경로에 푼 뒤, ignored
artifact와 manifest만 복사한다.

```bash
export V2_PAYLOAD_ROOT="$(mktemp -d /workspace/v2-payload.XXXXXX)"
.venv/bin/python - <<'PY'
from pathlib import PurePosixPath
import tarfile

archives = {
    "/workspace/moshi_v2_provisional_runpod_payload_b8d30fd.tar.gz": {
        "count": 604,
        "wav_count": 600,
        "fixed": {
            "experiments/self_repair/dataset_v2/artifacts/provisional_prepared/",
            "experiments/self_repair/dataset_v2/evaluation/provisional_eval_trials.jsonl",
            "experiments/self_repair/dataset_v2/evaluation/RUNPOD_PROVISIONAL_COMMANDS.md",
            "experiments/self_repair/dataset_v2/config/eval.json",
        },
    },
    "/workspace/moshi_v2_seed17_pilot_b977391.tar.gz": {
        "count": 1,
        "wav_count": 0,
        "fixed": {
            "experiments/self_repair/dataset_v2/evaluation/"
            "provisional_eval_trials_seedpilot_b977391.jsonl",
        },
    },
}
wav_prefix = "experiments/self_repair/dataset_v2/artifacts/provisional_prepared/"
for archive, expected in archives.items():
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
    names = [member.name for member in members]
    assert len(names) == expected["count"], (archive, len(names))
    assert len(set(names)) == len(names), f"duplicate archive member: {archive}"
    for member in members:
        path = PurePosixPath(member.name)
        assert not path.is_absolute() and ".." not in path.parts, member.name
        assert member.isfile() or member.isdir(), f"link/special member: {member.name}"
    wav = {name for name in names if name.startswith(wav_prefix) and name.endswith(".wav")}
    other = set(names) - wav
    assert len(wav) == expected["wav_count"], (archive, len(wav))
    assert other == expected["fixed"], (archive, sorted(other ^ expected["fixed"]))
    print(f"archive allowlist passed: {archive} ({len(names)} members)")
PY
tar -xzf /workspace/moshi_v2_provisional_runpod_payload_b8d30fd.tar.gz \
  -C "$V2_PAYLOAD_ROOT"
tar -xzf /workspace/moshi_v2_seed17_pilot_b977391.tar.gz \
  -C "$V2_PAYLOAD_ROOT"

mkdir -p "$MECH_REPO_ROOT/experiments/self_repair/dataset_v2/artifacts/provisional_prepared"
mkdir -p "$MECH_REPO_ROOT/experiments/self_repair/dataset_v2/evaluation"
cp -a "$V2_PAYLOAD_ROOT/experiments/self_repair/dataset_v2/artifacts/provisional_prepared/." \
  "$MECH_REPO_ROOT/experiments/self_repair/dataset_v2/artifacts/provisional_prepared/"
cp "$V2_PAYLOAD_ROOT/experiments/self_repair/dataset_v2/evaluation/provisional_eval_trials_seedpilot_b977391.jsonl" \
  "$MECH_REPO_ROOT/experiments/self_repair/dataset_v2/evaluation/"
```

위 검사는 첫 archive의 604개 member(600 WAV + directory + 고정된 3개 파일), 두 번째 archive의
단일 corrected manifest, 중복·절대경로·`..`·link/special member 부재를 정확히 확인한다. 하나라도
다르면 assertion으로 extract 전에 중단한다.

Archive 내부에 해당 corrected manifest가 없으면 두 번째 archive가 올바르게 풀렸는지 확인한다.
새 mechanistic run에서는 old manifest의 `code_commit`을 그대로 재사용하지 말고, audio source
identity를 참조해 **현재 clean harness commit으로 새 mechanistic manifest**를 만든다.

```bash
mkdir -p "$MECH_RUN_ROOT/manifests"

# [TARGET]
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/build_mech_manifest.py \
  --source-eval-manifest \
    experiments/self_repair/dataset_v2/evaluation/provisional_eval_trials_seedpilot_b977391.jsonl \
  --prepared-manifest /workspace/provisional_prepared_stimuli.jsonl \
  --analysis-folds experiments/self_repair/dataset_v2/assignments/analysis_folds.jsonl \
  --audio-root "$MECH_DATA_ROOT" \
  --seeds 17 \
  --data-status exploratory_provisional \
  --output "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl"
```

Prepared manifest 안의 과거 로컬 absolute URI는 실행 path로 신뢰하지 않는다. Builder는
`prepared_stimulus_id`, WAV filename, SHA-256으로 `/workspace`의 실제 WAV를 다시 bind하고 새로운
상대 URI를 기록해야 한다. 새 multivalue control archive가 있다면 별도 source manifest와 SHA-256도 upload하고, 같은 tool로
`confirmatory_multivalue` manifest를 만든다. Runtime manifest를 tracked source directory에 쓰지
않아야 checkout이 clean한 상태로 유지된다.

---

## Step 6. 기존 자산과 GPU 계약을 preflight한다

먼저 현재 존재하는 test와 behavioral dry-run을 실행한다.

```bash
cd "$MECH_REPO_ROOT"

# [EXISTS]
.venv/bin/python -m unittest discover \
  -s experiments/self_repair/tests/dataset_v2 -v

# [EXISTS] old behavioral manifest/input 계약만 검증한다.
.venv/bin/python experiments/self_repair/scripts/dataset_v2/run_eval_v2.py \
  --input experiments/self_repair/dataset_v2/evaluation/provisional_eval_trials_seedpilot_b977391.jsonl \
  --output experiments/self_repair/dataset_v2/evaluation/preflight_results.jsonl \
  --generation-config experiments/self_repair/dataset_v2/config/eval.json \
  --artifact-root experiments/self_repair/dataset_v2 \
  --response-root /workspace/preflight-responses \
  --only-seed 17 \
  --dry-run
```

이 dry-run은 old manifest의 runner/config/audio hash만 검사하며 현재 mechanistic code identity나
model execution을 증명하지 않는다. 이어서 구현된 contract validator를 실행한다.

```bash
# [TARGET]
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/validate_mechanistic_contract.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --input-artifact-root "$MECH_DATA_ROOT" \
  --output-root "$MECH_RUN_ROOT/preflight/model_contract" \
  --model-repo kyutai/moshiko-pytorch-bf16 \
  --model-revision 2bfc9ae6e89079a5cc7ed2a68436010d91a3d289
```

GO 조건:

- 32 layers, 32 heads, hidden size 4,096, head dim 128
- Mimi 24 kHz, 1,920 samples/frame, 80 ms/frame, max LM delay 1
- expected input 600/600, missing/duplicate/hash mismatch 0
- compile과 CUDA graph가 모두 비활성화됨
- run identity/environment report가 생성됨
- self-patch/no-op/cache-reset contract test가 모두 통과함

하나라도 실패하면 full scan을 시작하지 않는다.

---

## Step 7. bounded user WAV만 먼저 Mimi code로 변환한다 `[TARGET]`

반복 patch마다 audio를 다시 encode하면 시간도 낭비되고 codec state leakage를 놓치기 쉽다.
WAV hash와 run identity를 key로 user/conversation/assistant-silence 8-codebook tensor를 한 번 만들고
원자적으로 기록한 content-addressed NPZ로 저장한다. 아직 600개 전체를 encode하지 않는다. 먼저
outcome과 무관한 deterministic selector로 최대 4개 clean/repair canary만 고른다.

```bash
mkdir -p "$MECH_RUN_ROOT/gpu_canary"
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/select_gpu_canary_manifest.py \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --output "$MECH_RUN_ROOT/gpu_canary/canary_trials.jsonl" \
  --max-trials 4 \
  --role discovery

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/encode_user_audio.py \
  --manifest "$MECH_RUN_ROOT/gpu_canary/canary_trials.jsonl" \
  --input-artifact-root "$MECH_DATA_ROOT" \
  --output-root "$MECH_RUN_ROOT/gpu_canary/encoded" \
  --output-manifest "$MECH_RUN_ROOT/gpu_canary/encoded_manifest.jsonl" \
  --model-revision 2bfc9ae6e89079a5cc7ed2a68436010d91a3d289 \
  --resume
```

각 manifest row에는 source WAV SHA, sample/frame 수, tensor shape/dtype/hash, Mimi revision,
prefix silence를 넣는다. 일부 입력을 두 번 encode해 byte-identical tensor인지 확인하고,
모든 trial 사이에서 codec/model streaming state를 reset한다.

이 단계의 GO 조건은 canary expected coverage 100%, duplicate/missing/hash mismatch 0,
repeated-encode mismatch 0이다. 전체 manifest encode는 Step 11의 GPU·대화·사람 검수 뒤에만 한다.

---

## Step 8. semantic event를 80 ms frame으로 매핑한다 `[TARGET]`

Prepared manifest의 timing provenance를 구분해서 사용한다.

- root/cue/new/repeated-old/closing 값은 `prepared_timing`에 이미 480 ms prefix가 반영돼 있으므로
  다시 더하지 않는다.
- D1/D2/D3/N1/N2/N3의 `alignment.unit_spans`는 content-relative다. 여기에
  `preparation.prefix_ms_actual`을 정확히 한 번 더한다.

```text
audio_start_frame          = floor(onset_ms / 80)
audio_end_frame_exclusive  = ceil(offset_ms / 80)
last_overlapping_frame     = ceil(offset_ms / 80) - 1
```

여기서 audio frame 번호를 hidden-state index로 바로 간주하지 않는다. 기존 streaming path는 첫
audio code에서 delay 초기화를 위해 LM step을 추가로 호출한다. Harness가 각 step마다
`consumed_audio_frame`, `lm_step`, `hidden_absolute_position`, `delay_slot`을 기록한
`frame_trace.jsonl`을 만들고, “event의 마지막 겹침 frame을 소비한 직후 state”를 실제 trace로
찾아야 한다. 이 mapping을 synthetic impulse/short fixture로 단위 테스트한다.

Primary anchor는 결과를 보기 전에 동결한다.

- `old_end`
- `cue_end`
- `new_end`
- `new_end+160ms`, `new_end+320ms`, `new_end+640ms`
- `D1_end`, `D2_end`, `D3_end`
- `query_end`와 fixed post-user readout frames

LM delay는 분석자가 켜고 끄는 옵션이 아니다. Harness는 checkpoint에서 stream별 delay vector를
읽고 `output_f[k] = feedback_{f+d_k}[k]`의 역스케줄과 최초 prime step을 적용한다. 각 frame trace에
delay vector와 실제 source frame을 남기고, config의 `max_lm_delay=1`과 다르면 중단한다. repair cue 전체를
하나로 묶으면 new city와 뒤의 repeated old city가 섞이므로 `new_value`와
`repeated_old` span을 반드시 분리한다. clean/repair는 길이가 다르므로 같은 raw frame 번호가
아니라 semantic anchor끼리 patch한다.

```bash
# [TARGET]
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/build_anchor_map.py \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --prepared-manifest /workspace/provisional_prepared_stimuli.jsonl \
  --output "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --frame-trace-output "$MECH_RUN_ROOT/frame_trace.jsonl"
```

산출물: `frame_trace.jsonl`과 `anchor_map.jsonl`. 모든 anchor가 `[0, frame_count)` 안에 있고
frame↔ms 역변환 오차가 80 ms 이하여야 한다. 각 primary anchor의 ±1 frame sensitivity도 별도
label로 실행한다.

---

## Step 9. strict open-loop replay를 검증한다 `[TARGET]`

주 분석에서는 user Mimi codes만 조건에 따라 바꾸고 assistant-side history는 모든 pair에서
동일하게 고정한다.

권장 primary policy:

- assistant text feedback: embedding이 0이 되는 null token 또는 검증된 PAD
- assistant audio feedback: 공통 null codebook token
- logits는 계산하지만 sampled token은 feedback history에 쓰지 않음

Sensitivity policies:

- text PAD + Mimi가 encode한 digital-silence trace
- 모든 condition에 동일한 frozen natural assistant trace

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/validate_open_loop.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --encoded-manifest "$MECH_RUN_ROOT/gpu_canary/encoded_manifest.jsonl" \
  --output "$MECH_RUN_ROOT/gpu_canary/open_loop_validation.json"
```

GO 조건:

- paired clean/repair의 assistant feedback tensor hash가 byte-identical
- feedback history에 sampled non-null text/audio token이 없음
- 동일 input replay의 logits/hidden state가 사전 tolerance 이내
- `receiver <- receiver`와 identity callback effect가 tolerance 이내
- readout 후보마다 같은 query-time snapshot을 deep-clone하며 target→stale와 stale→target 실행
  순서를 바꿔도 margin이 같음
- trial reset 뒤 첫 trial과 재실행 결과가 일치
- 첫 audio frame의 delay handling과 모든 time index가 unit test와 일치

Tolerance는 labeled outcome을 보기 전에 numeric smoke로 정하고 config에 freeze한다. 실패하면
아래 activation/probe/patch 단계로 가지 않는다.

---

## Step 10. causal readout과 baseline gap을 자격시험한다 `[TARGET]`

다음 readout을 fixed prefix와 teacher forcing으로 점수화한다.

- root: “The current destination is …”
- D1: “The sightseeing city is …”
- D2: “The dining city is …”
- D3: “The hotel city is …”

문구는 예시다. 실제 verbalizer와 token IDs는 `readouts.json`에 고정한다. 다른 문구를 결과를
본 뒤 선택하면 해당 결과는 exploratory로 강등한다.

Moshi의 user channel은 text가 아니라 audio다. 위 문장을 새 user-text prompt로 주입하지 않는다.
`query_end` state와 전체 streaming/KV state를 deep-clone한 뒤, model self-text channel의 고정된
readout prefix와 candidate continuation을 teacher-force한다. 그동안 user side에는 같은 digital
silence code, assistant audio side에는 같은 frozen null code sequence를 준다.

`readouts.json`은 80 ms frame별로 prefix token, candidate token, PAD 위치를 적은 emission
schedule을 가진다. Primary schedule은 clean-only calibration으로 하나를 골라 labeled repair
결과를 보기 전에 freeze한다. Timing sensitivity는 미리 고정한 여러 offset/PAD schedule의
likelihood를 `logmeanexp`로 평균하며 가장 잘 나온 schedule을 사후 선택하지 않는다. Target와
stale branch는 반드시 같은 prefix 직후의 원본 snapshot에서 각각 시작하고, 한 후보가 바꾼
text/KV history를 다른 후보가 이어받지 않는다. Null-context city-prior margin을 뺀 값도
sensitivity metric으로 기록한다.

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/score_readouts.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --readouts "$MECH_RUN_ROOT/preflight/model_contract/readouts.bound.json" \
  --manifest "$MECH_RUN_ROOT/gpu_canary/canary_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/gpu_canary/encoded_manifest.jsonl" \
  --anchors "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --role discovery \
  --output "$MECH_RUN_ROOT/gpu_canary/baseline_readout.jsonl"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/capture_activations.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/gpu_canary/canary_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/gpu_canary/encoded_manifest.jsonl" \
  --anchor-map "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --role discovery \
  --sites logits,resid_post \
  --layers 0,15,31 \
  --anchors query_end \
  --output-root "$MECH_RUN_ROOT/gpu_canary/capability_capture" \
  --resume
```

여기서 얻는 수치는 bounded capability smoke일 뿐 discovery 결과가 아니다. 전체 discovery
readout/capture는 Step 11의 유료 실행 authorization이 발행된 뒤 같은 frozen 인자로 다시 실행한다.

GO 권고 기준:

- clean current-value sign accuracy ≥80%
- Boston→Seattle과 Seattle→Boston 방향별 sign accuracy ≥80%
- 한 도시 prior만으로 성공하지 않음
- clean-current와 repair의 margin gap CI가 0을 벗어나고, causal rescue를 평가할 충분한 pair가 있음
- root와 D1–D3 dependency readout 사이의 효과 방향이 사전 정의와 일치함
- target/stale branch 순서를 뒤집어도 margin이 tolerance 안에서 동일

Clean/repair gap이 없으면 “repair failure 원인”을 찾을 대상이 없다. 이 경우 표본을 behavior로
사후 선택하지 말고 representation-dynamics study로 질문을 낮추거나 readout/자극을 수정한다.

---

## Step 11. smoke를 먼저 끝낸다 `[TARGET]`

Full grid 전에 다음 최소 조합만 실행한다.

- 1 scenario, 양방향
- clean + delayed-three, 1 speaker
- old/new/query anchor 각 1개
- early/middle/late layer 각 1개
- self/current/wrong-target/shuffled donor
- residual 1종, 후보 head 1개, K/V 1개

Bounded canary selector는 clean/repair를 `scenario_id`, `direction_id`, `speaker_id`, current/new
value까지 일치시킨다. 가능하면 `clean_final` + `delayed_three_dependencies` 쌍을 먼저 선택하고,
추가 행도 같은 matched group 안에서만 최대 4개까지 채운다. Selection sidecar는 이 grouping과
source/canary manifest SHA-256을 함께 고정한다.

이 smoke에는 folds 4–5와 새 formal-confirmation data를 절대 넣지 않는다. Patch effect의 기대
부호를 확인하는 test는 정답을 아는 analytic toy/constructed fixture에서만 수행한다. 실제 한
scenario가 원하는 부호를 낼 때까지 구현을 조정하지 않는다.

유료 작업 전에 각 예정 grid를 immutable JSON으로 freeze한다. `freeze_paid_scan_spec.py`는 checkpoint나
GPU backend를 import하지 않고 정적 산술용 `trial_selector`, `recipient_selector`, `scans`, `storage`와
실제 명령을 byte-level로 대조할 `execution`을 함께 만든다. Full-duplex 비용까지 같은 spec에 넣을
때만 `--include-generation`을 명시한다. `execution`에는 `kind`, `role`, `layers`, `anchors`, `donors`,
`controls`, `components`, `limit_scenarios`, `selection_sha256`가 빠짐없이 있어야 한다.
`scans[].expected_cell_count`가 manifest에서 다시 계산한 수와 다르면 checkpoint를 load하지 않는다.
한 paid-scan spec에는 실행 grid 하나만 둔다. 그 grid의 `donor_arms`는 generic active-arm
차원으로, residual/KV/path에서는 `execution.donors`, component에서는 `execution.controls`와
순서까지 같아야 한다. 반대쪽 inactive CLI default는 hash에는 묶되 cell 수에 중복 가산하지 않는다.

32 layers × 세 anchor × 전체 discovery repair를 한 번에 합치면 default cell budget을 넘는다. 따라서
anchor별 세 spec을 결과를 보기 전에 모두 동결하고 각 9,216-cell stage를 별도 authorization으로
연다. `128` readout steps는 bound token 길이보다 크게 잡은 사전 비용 예약값이며 관측 step 수가
아니다.

```bash
mkdir -p /workspace/mech-plans
for anchor in old_end new_end query_end; do
  .venv/bin/python \
    experiments/self_repair/mechanistic/scripts/freeze_paid_scan_spec.py \
    --config experiments/self_repair/mechanistic/config/mechanistic.json \
    --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
    --kind residual \
    --role discovery \
    --layers 0:32 \
    --anchors "$anchor" \
    --donors clean_current \
    --controls self,clean_current,clean_stale,shuffled \
    --components resid_post \
    --full-replays-per-cell 3 \
    --readout-steps-per-cell 128 \
    --output "/workspace/mech-plans/residual-discovery-$anchor.json"
done

export MECH_SCAN_SPEC=/workspace/mech-plans/residual-discovery-query_end.json

# 정적 hash/산술 → 최대 4개 입력 real-GPU canary 순서다. 전체 600개를 자동 encode하지 않는다.
experiments/self_repair/mechanistic/runpod/runpod_smoke.sh
```

첫 실행은 bounded GPU 측정을 마친 뒤 대화 증거가 아직 없으므로 exit 3 `NO_GO`로 끝나는 것이
정상이다. 이어서 같은 canary 중 repair 한 개를 다섯 seed와 두 필수 startup mode로 실행한다.
`common_handshake_then_request`는 모델의 실제 첫 인사가 terminal punctuation으로 끝난 뒤
text/audio quiet 20 frames(1.6초)를 확인하고, 준비된 6-frame(480 ms) lead-in과 user 요청을 동일한
continuous Mimi stream에 붙인다. `greeting_suppressed`는 별도 실험 arm이며, 두 mode 모두 model
frame 0부터 user 종료 뒤 40초까지 full audio/text를 보존한다. 사전선택 단계에서는 결과를 모르는
identity/no-op pair만 사용한다.

```bash
export MECH_CONVERSATION_ROOT="$MECH_RUN_ROOT/gpu_canary/conversation"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/run_full_duplex_validation.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/gpu_canary/canary_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/gpu_canary/encoded_manifest.jsonl" \
  --anchors "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --input-artifact-root "$MECH_DATA_ROOT" \
  --primary-intervention identity_noop \
  --donor-arms none \
  --seeds 17,29,42,101,2026 \
  --limit-trials 1 \
  --output-root "$MECH_CONVERSATION_ROOT"
```

이 첫 호출은 `validation.jsonl`, HMAC pseudonym으로 이름 붙인 full/primary WAV와
`blind_review_template.jsonl`만 만든다. `conversation_canary.json`은 아직 만들지 않는다. 두 독립
검수자가 template의 immutable field는 그대로 둔 채 서로 다른 `reviewer_id`와 아래 일곱 boolean을
각자 채운다.

```text
natural_flow, primary_response_scorable, final_target_correct, stale_state_error,
d1_binding_correct, d2_binding_correct, d3_binding_correct
```

두 검수 결과를 한 `reviews.jsonl`에 모은 뒤 같은 명령에
`--resume --reviews "$MECH_CONVERSATION_ROOT/reviews.jsonl"`를 추가한다. 판정이 하나라도 다르면
이 호출은 `adjudication_template.jsonl`을 쓰고 중단한다. 두 검수자와 다른 제3자가 그 파일의
`adjudicator_id`와 일곱 boolean을 채운 후
`--adjudications "$MECH_CONVERSATION_ROOT/adjudications.jsonl"`까지 추가해 다시 실행한다.

모든 clip이 이중 검수되고 충돌이 조정된 real-checkpoint run만
`$MECH_CONVERSATION_ROOT/conversation_canary.json`을 만든다. Synthetic 파일은 이름부터
`synthetic_conversation_canary.json`이며 readiness evidence로 거부된다.

스크립트는 CUDA가 없으면 static estimate만 남기고 exit 3으로 중단한다. CUDA가 있으면 exact pinned
checkpoint로 bounded identity-patch cell을 실행하여 다음 수치를
`gpu_canary/gpu_measurements.json`에 기록한다.

- peak/total VRAM bytes
- 실제 capture activation bytes
- elapsed seconds, mean cell seconds, seconds per model frame
- 전체 선언 grid의 cell-based/frame-based GPU-hour ETA
- 전체 activation/storage reserved bytes

Conversation evidence의 audio activity policy는 version, detector, Mimi frame size, -45 dBFS
threshold, forced-silence calibration rule을 모두 포함한다. Readiness는 선언된 digest를 신뢰하지 않고
`policy_sha256`를 제외한 policy object를 canonical JSON으로 다시 SHA-256하며, 실제 GPU
forced-silence decode maximum이 threshold 미만인 수치가 없으면 `NO_GO`다.

같은 경로의 canary 결과를 재사용해 새 evidence라고 부르지 않는다. 다시 측정하려면 새
identity-specific canary output root를 사용한다. 모델의 두 startup mode 대화 canary와 사람 흐름
검수가 아직 없다면 `NO_GO`로 끝나는 것이 정상이다. 특히 모드별 4회 이상, exact output coverage
100%, text/audio tail 검사 100%, response cap에서 활동 중인 run 0, truncation 0, 사람 자연스러움
승인 100%가 필요하다. Frozen `frame_rms_dbfs` policy와 실제 GPU forced-silence decode의 최대 dBFS도
기록하고, 그 측정 전에는 threshold를 calibrated라고 부르지 않는다.

GO 조건: finite output, deterministic replay, self-patch no-op, cache position 일치,
atomic row write, kill/restart 후 resume 중복 0과 위 수치/대화 gate가 전부 통과한다. Full encoded
manifest는 bounded canary를 사람이 검토한 뒤 별도 명시적 명령으로 만들며 smoke script가 600개를
몰래 encode하지 않는다.

사람 검수가 통과한 뒤에만 600개 전체를 명시적으로 encode한다. 기존 canary와 다른 output
manifest에 쓰고, 그 다음 source/code/model/config/scan hash와 모든 canary evidence를 하나의
authorization에 묶는다.

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/encode_user_audio.py \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --input-artifact-root "$MECH_DATA_ROOT" \
  --output-root "$MECH_RUN_ROOT/encoded_user" \
  --output-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --model-revision 2bfc9ae6e89079a5cc7ed2a68436010d91a3d289 \
  --resume

export MECH_CONVERSATION_CANARY="$MECH_CONVERSATION_ROOT/conversation_canary.json"
export MECH_FULL_ENCODED_MANIFEST="$MECH_RUN_ROOT/encoded_user_manifest.jsonl"

# 세 stage가 같은 bounded 측정을 재사용하더라도 target scan-spec hash가 다르므로
# evidence/authorization은 anchor마다 별도로 발행한다.
for anchor in old_end new_end query_end; do
  spec="/workspace/mech-plans/residual-discovery-$anchor.json"
  gate="$MECH_RUN_ROOT/preflight/residual-$anchor"
  mkdir -p "$gate"

  .venv/bin/python \
    experiments/self_repair/mechanistic/scripts/assemble_readiness_evidence.py \
    --config experiments/self_repair/mechanistic/config/mechanistic.json \
    --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
    --encoded-manifest "$MECH_FULL_ENCODED_MANIFEST" \
    --scan-spec "$spec" \
    --model-contract "$MECH_RUN_ROOT/preflight/model_contract/model_contract.json" \
    --model-run-identity "$MECH_RUN_ROOT/preflight/model_contract/run_identity.json" \
    --open-loop "$MECH_RUN_ROOT/gpu_canary/open_loop_validation.json" \
    --conversation-canary "$MECH_CONVERSATION_CANARY" \
    --gpu-canary "$MECH_RUN_ROOT/gpu_canary/gpu_measurements.json" \
    --canary-manifest "$MECH_RUN_ROOT/gpu_canary/canary_trials.jsonl" \
    --canary-encoded-manifest "$MECH_RUN_ROOT/gpu_canary/encoded_manifest.jsonl" \
    --output "$gate/readiness_evidence.json"

  .venv/bin/python \
    experiments/self_repair/mechanistic/scripts/assess_mechanistic_readiness.py \
    --config experiments/self_repair/mechanistic/config/mechanistic.json \
    --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
    --encoded-manifest "$MECH_FULL_ENCODED_MANIFEST" \
    --scan-spec "$spec" \
    --evidence "$gate/readiness_evidence.json" \
    --output "$gate/paid_scan_authorization.json"
done
```

Authorization이 `GO`일 때만 전체 discovery baseline/capture를 실행한다. 여기에도 frozen manifest,
readout, anchor를 그대로 쓴다.

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/score_readouts.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --readouts "$MECH_RUN_ROOT/preflight/model_contract/readouts.bound.json" \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_FULL_ENCODED_MANIFEST" \
  --anchors "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --role discovery \
  --output "$MECH_RUN_ROOT/baseline_readout.jsonl"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/capture_activations.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_FULL_ENCODED_MANIFEST" \
  --anchor-map "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --role discovery \
  --sites logits,resid_post \
  --layers 0:32 \
  --anchors old_end,new_end,D1_end,D2_end,D3_end,query_end \
  --output-root "$MECH_RUN_ROOT/discovery_baseline" \
  --resume
```

```text
estimated GPU hours = number_of_patch_cells × mean_suffix_seconds / 3600
```

기존 seed-17 free-running 600개는 약 4.99 GPU-hours였지만 mechanistic 비용은 trial 수보다
patch cell 수와 replay suffix 길이에 좌우되므로 이 값을 그대로 외삽하지 않는다.

모든 gate를 통과하면 `assemble_readiness_evidence.py`와
`assess_mechanistic_readiness.py`가 `paid_scan_authorization.json`을 발행한다. 이 파일은 Git commit,
model repo/revision, source+encoded data, manifest, config, scan spec, 모든 evidence와 assessment를
SHA-256으로 묶는다. 단순 JSON의 `passed=true`는 인정하지 않는다. 실제 scan은 checkpoint 생성 전에
내용 hash와 현재 파일을 다시 대조한다. 정확한 명령에는 다음 두 인자를 항상 추가한다.

```text
--scan-spec "$MECH_SCAN_SPEC"
--readiness-go "$MECH_RUN_ROOT/preflight/paid_scan_authorization.json"
```

둘 중 하나가 없거나 파일/CLI가 한 글자라도 바뀌면 `NO_GO`다. `--resume`은 cell identity를 먼저
검증하며 이미 끝난 cell은 donor/recipient replay 전에 건너뛴다. 전 cell이 끝났다면 checkpoint도
load하지 않는다.

---

## Step 12. residual-stream coarse localization을 실행한다 `[TARGET]`

Discovery folds 1–3에서만 32 layers × frozen semantic anchors를 훑는다. 처음부터 1,024 heads ×
모든 frame을 실행하지 않는다.

Recipient는 repair run이다. 주요 donor는 같은 scenario/speaker/current destination의
clean-current run이다. 같은 text bundle의 same-direction clean-current는 speaker matched다.
반대 destination의 clean-stale donor는 speaker가 달라질 수 있으므로 manifest에서 일치 여부를
확인하고, 맞지 않으면 matched donor라고 부르지 않으며 새 control set을 사용한다.

`old_end`, `new_end`, `query_end` 세 grid는 결과를 보기 전에 이미 Step 11에서 동결됐다. 각 명령은
동일한 32 layers × 288 discovery repair recipients × 1 clean-current arm = 9,216 cells다. 먼저
`--plan-only`로 exact cell identity를 만들고, planned count/hash가 spec과 일치하는지 확인한 뒤 같은
인자에서 `--plan-only`만 제거해 실행한다. 한 stage의 결과를 보고 뒤 stage의 grid를 바꾸지 않는다.

```bash
for anchor in old_end new_end query_end; do
  spec="/workspace/mech-plans/residual-discovery-$anchor.json"
  gate="$MECH_RUN_ROOT/preflight/residual-$anchor/paid_scan_authorization.json"
  out="$MECH_RUN_ROOT/discovery/residual-$anchor"

  .venv/bin/python \
    experiments/self_repair/mechanistic/scripts/scan_residual_patches.py \
    --config experiments/self_repair/mechanistic/config/mechanistic.json \
    --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
    --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
    --anchor-map "$MECH_RUN_ROOT/anchor_map.jsonl" \
    --readouts "$MECH_RUN_ROOT/preflight/model_contract/readouts.bound.json" \
    --role discovery \
    --layers 0:32 \
    --anchors "$anchor" \
    --donors clean_current \
    --controls self,clean_current,clean_stale,shuffled \
    --scan-spec "$spec" \
    --readiness-go "$gate" \
    --output-root "$out" \
    --plan-only

  .venv/bin/python \
    experiments/self_repair/mechanistic/scripts/scan_residual_patches.py \
    --config experiments/self_repair/mechanistic/config/mechanistic.json \
    --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
    --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
    --anchor-map "$MECH_RUN_ROOT/anchor_map.jsonl" \
    --readouts "$MECH_RUN_ROOT/preflight/model_contract/readouts.bound.json" \
    --role discovery \
    --layers 0:32 \
    --anchors "$anchor" \
    --donors clean_current \
    --controls self,clean_current,clean_stale,shuffled \
    --scan-spec "$spec" \
    --readiness-go "$gate" \
    --output-root "$out" \
    --resume
done
```

Donor assignment는 실행 전에 recipient별 same scenario/voice/value/anchor 우선순위, 동률 처리,
여러 donor 평균 방식, shuffled-donor derangement 제약과 RNG seed까지 manifest에 고정한다. Donor
반복은 추가 독립 표본이 아니라 같은 scenario 안의 반복 측정이다.

Primary는 anchor의 마지막 한 frame을 patch한다. 2-frame sensitivity는 semantic하게 정렬된 두
frame tensor를 순서대로 바꾸며 activation을 mean-pool해서 한 vector를 주입하지 않는다. 후보
layer/time에서만 더 긴 ordered span으로 넓힌다. ±1-frame sensitivity가 primary anchor의 실패를
대체하지 않는다. Discovery
heatmap은 탐색적이며 selection p-value를 붙이지 않는다. 전체 grid 자체를 하나의 주장으로
검정하려면 scenario permutation max-T FWER를 쓴다.

Gradient attribution/AtP는 full-sequence eager gradient path를 별도로 검증한 뒤 patch cell의
우선순위를 정하는 선택적 prescreen으로만 쓸 수 있다. Attribution 상위 site도 exact activation
patch를 거쳐야 하며, attribution 단독 결과를 causal localization으로 보고하지 않는다.

이 coarse primary grid는 `clean_current` donor만 billable arm으로 실행한다. 아래 controls는 Step 14의
사전 동결된 component grid와 Step 16/17 confirmatory grid에서 실행한다. Coarse grid의 inactive
`--controls` 목록은 spec identity에는 묶이지만 cell 수에는 포함되지 않는다.

필수 controls:

- self/no-op
- same-value random donor
- wrong destination donor
- clean-stale(wrong-destination) donor
- shuffled scenario/speaker donor
- activation norm 및 out-of-distribution 진단

---

## Step 13. probe는 진단용으로만 학습한다 `[TARGET]`

Probe task를 섞지 않는다. 현재 하네스가 직접 만드는 primary probe label은 manifest의
`new_value`, 즉 K-class `current_destination`이다. 아래 다른 질문은 같은 label을 재해석하지 말고
별도의 사전 선언 label manifest와 독립 probe artifact를 추가한 뒤에만 실행한다.

- `current_destination`
- `old_value_identity` (K-class)
- `old_was_mentioned` (boolean)
- `is_retracted`
- D1/D2/D3의 `bound_to_current`

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/fit_probes.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --capture-root "$MECH_RUN_ROOT/discovery_baseline" \
  --role discovery \
  --group-by scenario_id \
  --probe-grid \
  --sites resid_post \
  --layers 0:32 \
  --anchors old_end,new_end,query_end \
  --alpha 1.0 \
  --output-root "$MECH_RUN_ROOT/discovery/probes"
```

이 명령은 `site × layer × semantic-anchor`별 L2 ridge probe를 따로 만들며 tensor를 layer/time에
걸쳐 평균하지 않는다. `alpha=1.0`, scenario-grouped CV와 seed는 결과를 보기 전에 고정한다. 각
cell의 class mapping, fold prediction, 정확도, source capture/hash가
`probe_grid_metrics.jsonl`에 남는다. Probe는 진단용이며 causal selection의 점수로 쓸 수 없다.

필수 비교는 majority, shuffled-label, Mimi/layer-0, timing/duration-only, random projection,
parameter-count-matched control, clean→repair cross-condition transfer다. 한 vector의 여러 frame을
독립 sample로 부풀리지 않는다. v2만 썼다면 “old memory를 찾았다”가 아니라
“Boston-vs-Seattle axis가 decode되었다”고 쓴다.

Discovery causal scan이 끝나면 Step 16의 frozen mechanistic selection과 정확히 같은
site/layer/anchor가 이 probe grid에 존재하고 head-specific이 아닐 때만 `--site-selection`과
`--freeze-output`으로 coefficients/class mapping을 동결한다. Head/KV/path selection처럼 현재
probe feature 계약과 맞지 않으면 probe를 억지로 다른 residual에 붙이지 않고 endpoint를
`not_applicable`로 남긴다. Step 16/17에서는 동결된 probe만 한 번 적용하며, validation/formal
heatmap으로 새 site를 고르지 않는다.

---

## Step 14. 후보 layer에서 attention/MLP/head를 좁힌다 `[TARGET]`

Residual scan에서 선택한 소수 layer/time만 분해한다.

1. aggregate attention output
2. MLP output
3. output projection 전 head `z`
4. mean ablation 및 donor patch

Head grid까지 controls 다섯 개를 한꺼번에 실행하면 기본 10,000-cell budget을 넘는다. Residual
결과에서 primary `clean_current` site를 먼저 scenario-balanced mean으로 동결하고, aggregate
attention/MLP controls(2,880 cells)와 head-z primary(9,216 cells)를 결과를 보기 전에 서로 다른
spec으로 동결한다. `--upstream-selection`은 이전 stage의 layer/anchor/head만 hash-bind하며 새
component와 active arm을 결과와 무관하게 명시할 수 있게 한다.

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/freeze_mechanistic_selection.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --discovery-root "$MECH_RUN_ROOT/discovery" \
  --output "$MECH_RUN_ROOT/discovery/residual_candidates.json"

# 이후 discovery stage가 공통으로 쓰는 source-specific authorization helper.
authorize_v2_stage() {
  local spec="$1"
  local gate="$2"
  mkdir -p "$gate"
  .venv/bin/python \
    experiments/self_repair/mechanistic/scripts/assemble_readiness_evidence.py \
    --config experiments/self_repair/mechanistic/config/mechanistic.json \
    --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
    --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
    --scan-spec "$spec" \
    --model-contract "$MECH_RUN_ROOT/preflight/model_contract/model_contract.json" \
    --model-run-identity "$MECH_RUN_ROOT/preflight/model_contract/run_identity.json" \
    --open-loop "$MECH_RUN_ROOT/gpu_canary/open_loop_validation.json" \
    --conversation-canary "$MECH_CONVERSATION_CANARY" \
    --gpu-canary "$MECH_RUN_ROOT/gpu_canary/gpu_measurements.json" \
    --canary-manifest "$MECH_RUN_ROOT/gpu_canary/canary_trials.jsonl" \
    --canary-encoded-manifest "$MECH_RUN_ROOT/gpu_canary/encoded_manifest.jsonl" \
    --output "$gate/readiness_evidence.json"
  .venv/bin/python \
    experiments/self_repair/mechanistic/scripts/assess_mechanistic_readiness.py \
    --config experiments/self_repair/mechanistic/config/mechanistic.json \
    --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
    --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
    --scan-spec "$spec" \
    --evidence "$gate/readiness_evidence.json" \
    --output "$gate/paid_scan_authorization.json"
}

export MECH_COMPONENT_AGG_SPEC=/workspace/mech-plans/component-aggregate-discovery.json
export MECH_COMPONENT_HEAD_SPEC=/workspace/mech-plans/component-head-discovery.json

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/freeze_paid_scan_spec.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --kind component --role discovery \
  --upstream-selection "$MECH_RUN_ROOT/discovery/residual_candidates.json" \
  --donors clean_current,self,shuffled \
  --controls clean_current,clean_stale,self,same_value_random,shuffled \
  --components attn_out,mlp_out \
  --full-replays-per-cell 3 --readout-steps-per-cell 128 \
  --output "$MECH_COMPONENT_AGG_SPEC"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/freeze_paid_scan_spec.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --kind component --role discovery \
  --upstream-selection "$MECH_RUN_ROOT/discovery/residual_candidates.json" \
  --donors clean_current,self,shuffled \
  --controls clean_current \
  --components head_z \
  --full-replays-per-cell 3 --readout-steps-per-cell 128 \
  --output "$MECH_COMPONENT_HEAD_SPEC"

authorize_v2_stage "$MECH_COMPONENT_AGG_SPEC" \
  "$MECH_RUN_ROOT/preflight/component-aggregate"
authorize_v2_stage "$MECH_COMPONENT_HEAD_SPEC" \
  "$MECH_RUN_ROOT/preflight/component-head"

# 두 명령 모두 먼저 --plan-only, 그 다음 동일 인자로 --resume 실행한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/scan_component_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --anchor-map "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --readouts "$MECH_RUN_ROOT/preflight/model_contract/readouts.bound.json" \
  --role discovery \
  --selection "$MECH_RUN_ROOT/discovery/residual_candidates.json" \
  --donors clean_current,self,shuffled \
  --controls clean_current,clean_stale,self,same_value_random,shuffled \
  --components attn_out,mlp_out \
  --scan-spec "$MECH_COMPONENT_AGG_SPEC" \
  --readiness-go "$MECH_RUN_ROOT/preflight/component-aggregate/paid_scan_authorization.json" \
  --output-root "$MECH_RUN_ROOT/discovery/components/aggregate" \
  --plan-only

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/scan_component_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --anchor-map "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --readouts "$MECH_RUN_ROOT/preflight/model_contract/readouts.bound.json" \
  --role discovery \
  --selection "$MECH_RUN_ROOT/discovery/residual_candidates.json" \
  --donors clean_current,self,shuffled \
  --controls clean_current,clean_stale,self,same_value_random,shuffled \
  --components attn_out,mlp_out \
  --scan-spec "$MECH_COMPONENT_AGG_SPEC" \
  --readiness-go "$MECH_RUN_ROOT/preflight/component-aggregate/paid_scan_authorization.json" \
  --output-root "$MECH_RUN_ROOT/discovery/components/aggregate" \
  --resume

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/scan_component_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --anchor-map "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --readouts "$MECH_RUN_ROOT/preflight/model_contract/readouts.bound.json" \
  --role discovery \
  --selection "$MECH_RUN_ROOT/discovery/residual_candidates.json" \
  --donors clean_current,self,shuffled \
  --controls clean_current \
  --components head_z \
  --scan-spec "$MECH_COMPONENT_HEAD_SPEC" \
  --readiness-go "$MECH_RUN_ROOT/preflight/component-head/paid_scan_authorization.json" \
  --output-root "$MECH_RUN_ROOT/discovery/components/heads" \
  --plan-only

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/scan_component_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --anchor-map "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --readouts "$MECH_RUN_ROOT/preflight/model_contract/readouts.bound.json" \
  --role discovery \
  --selection "$MECH_RUN_ROOT/discovery/residual_candidates.json" \
  --donors clean_current,self,shuffled \
  --controls clean_current \
  --components head_z \
  --scan-spec "$MECH_COMPONENT_HEAD_SPEC" \
  --readiness-go "$MECH_RUN_ROOT/preflight/component-head/paid_scan_authorization.json" \
  --output-root "$MECH_RUN_ROOT/discovery/components/heads" \
  --resume
```

각 row는 `[recipient, donor, layer, head, source_window, target_window, relation, baseline_M,
patched_M, delta_M, hashes]`를 가져야 한다. aggregate attention과 개별 head effect가 구조적으로
모순되는지 확인하고, probe score가 아니라 causal DID와 controls로 후보를 순위화한다.

Attention probability를 시각화하려면 동일 mask/RoPE/scaling으로 q·k에서 다시 계산하거나
instrumented SDPA가 반환한 값을 쓴다. Weight map은 “attention이 배분됐다”는 기술 통계일 뿐,
head output patch 없이 causal use의 증거로 쓰지 않는다.

---

## Step 15. KV와 path patching으로 propagation을 검사한다 `[TARGET]`

후보 head에서 먼저 **component selection에 고정된 정확한 semantic anchor**의 K-only / V-only /
K+V를 비교한다. 이 단계의 paid spec은 upstream selection의 layer/head/anchor를 그대로 상속하므로,
old/new/D1–D3를 한 spec에서 소급 탐색하지 않는다. 여러 source window의 시간 전파를 비교하려면
각 anchor를 결과 전에 별도 upstream selection/paid spec으로 선언해야 한다. 아래 path 단계는 동결된
single K/V writer와 `query_end` residual mediator의 제한된 writer→KV→query-residual 경로만
검사한다. 특정 query head reader라고 부르려면 별도로 `head_z` mediator/head를 결과 전에 동결해야
하며, 아래 residual mediator 결과를 head-level 증거로 해석하지 않는다.

```bash
# Aggregate/head 결과의 primary clean-current effect에서 한 exact component/head를 동결한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/freeze_mechanistic_selection.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --discovery-root "$MECH_RUN_ROOT/discovery/components" \
  --components head_z \
  --output "$MECH_RUN_ROOT/discovery/component_candidates.json"

export MECH_KV_SCAN_SPEC=/workspace/mech-plans/kv-discovery.json
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/freeze_paid_scan_spec.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --kind kv --role discovery \
  --upstream-selection "$MECH_RUN_ROOT/discovery/component_candidates.json" \
  --donors clean_current \
  --controls self,clean_current,clean_stale,shuffled \
  --components k_only,v_only,kv \
  --full-replays-per-cell 3 --readout-steps-per-cell 128 \
  --output "$MECH_KV_SCAN_SPEC"

authorize_v2_stage "$MECH_KV_SCAN_SPEC" "$MECH_RUN_ROOT/preflight/kv"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/scan_kv_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --anchor-map "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --readouts "$MECH_RUN_ROOT/preflight/model_contract/readouts.bound.json" \
  --role discovery \
  --selection "$MECH_RUN_ROOT/discovery/component_candidates.json" \
  --donors clean_current \
  --controls self,clean_current,clean_stale,shuffled \
  --modes k_only,v_only,kv \
  --scan-spec "$MECH_KV_SCAN_SPEC" \
  --readiness-go "$MECH_RUN_ROOT/preflight/kv/paid_scan_authorization.json" \
  --output-root "$MECH_RUN_ROOT/discovery/kv" \
  --plan-only

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/scan_kv_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --anchor-map "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --readouts "$MECH_RUN_ROOT/preflight/model_contract/readouts.bound.json" \
  --role discovery \
  --selection "$MECH_RUN_ROOT/discovery/component_candidates.json" \
  --donors clean_current \
  --controls self,clean_current,clean_stale,shuffled \
  --modes k_only,v_only,kv \
  --scan-spec "$MECH_KV_SCAN_SPEC" \
  --readiness-go "$MECH_RUN_ROOT/preflight/kv/paid_scan_authorization.json" \
  --output-root "$MECH_RUN_ROOT/discovery/kv" \
  --resume

# Path writer는 joint kv가 아닌 k_only/v_only 중 하나여야 한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/freeze_mechanistic_selection.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --discovery-root "$MECH_RUN_ROOT/discovery/kv" \
  --components k_only,v_only \
  --output "$MECH_RUN_ROOT/discovery/kv_writer_candidates.json"

export MECH_PATH_MEDIATOR_LAYER="$(.venv/bin/python - \
  "$MECH_RUN_ROOT/discovery/kv_writer_candidates.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["layer"])
PY
)"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/freeze_path_selection.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --writer-selection "$MECH_RUN_ROOT/discovery/kv_writer_candidates.json" \
  --mediator-site resid_post \
  --mediator-layer "$MECH_PATH_MEDIATOR_LAYER" \
  --mediator-anchor query_end \
  --mediator-head none \
  --output "$MECH_RUN_ROOT/discovery/path_selection.json"

export MECH_PATH_SCAN_SPEC=/workspace/mech-plans/path-discovery.json
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/freeze_paid_scan_spec.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --role discovery \
  --selection "$MECH_RUN_ROOT/discovery/path_selection.json" \
  --controls self,clean_current,clean_stale,shuffled \
  --full-replays-per-cell 3 --readout-steps-per-cell 128 \
  --output "$MECH_PATH_SCAN_SPEC"

authorize_v2_stage "$MECH_PATH_SCAN_SPEC" "$MECH_RUN_ROOT/preflight/path"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/run_path_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --anchor-map "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --readouts "$MECH_RUN_ROOT/preflight/model_contract/readouts.bound.json" \
  --role discovery \
  --selection "$MECH_RUN_ROOT/discovery/path_selection.json" \
  --donors clean_current \
  --controls self,clean_current,clean_stale,shuffled \
  --scan-spec "$MECH_PATH_SCAN_SPEC" \
  --readiness-go "$MECH_RUN_ROOT/preflight/path/paid_scan_authorization.json" \
  --output-root "$MECH_RUN_ROOT/discovery/paths" \
  --plan-only

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/run_path_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --anchor-map "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --readouts "$MECH_RUN_ROOT/preflight/model_contract/readouts.bound.json" \
  --role discovery \
  --selection "$MECH_RUN_ROOT/discovery/path_selection.json" \
  --donors clean_current \
  --controls self,clean_current,clean_stale,shuffled \
  --scan-spec "$MECH_PATH_SCAN_SPEC" \
  --readiness-go "$MECH_RUN_ROOT/preflight/path/paid_scan_authorization.json" \
  --output-root "$MECH_RUN_ROOT/discovery/paths" \
  --resume
```

중요한 RoPE 주의점: cache의 K는 이미 RoPE가 적용돼 있다. 다른 absolute frame의 post-RoPE K를
그대로 복사하면 value가 아니라 position까지 바뀐다. pre-RoPE K를 capture한 뒤 receiver position의
RoPE를 다시 적용하거나, donor/receiver anchor가 같은 absolute position인 전용 자극을 쓴다.
V도 시간·음향 matching이 필요하다.

“dependency까지 correction이 전파됐다”는 주장은 cue/new-value writer에서 D-unit/query reader로
제한한 path patch가 holdout에서 재현될 때만 한다. attention weight만으로 propagation을 주장하지
않는다.

---

## Step 16. 후보를 freeze하고 v2 internal validation을 한 번만 연다 `[TARGET]`

Discovery에서 다음을 `mechanistic_frozen_selection.json`에 기록한다. 현재 구현은 여러 site를
사후 조합하지 않고, scenario별 동일 가중의 canonical `clean_current` causal effect가 가장 큰
**단일 실행 가능한 site**를 고정한다. 동률은 component/layer/head/anchor/cell ID의 사전 정의된
lexicographic 순서로 해소한다. Path 결과는 propagation secondary evidence로 보존하지만 현재
full-duplex erasure 계약과 직접 호환되지 않으므로 최종 primary 후보군에서는 제외한다.

- primary single tensor site와 deterministic tie-break 규칙
- layer/head/site/window/donor/receiver/readout
- 예상 effect 방향
- primary/secondary hypothesis family와 Holm 규칙
- no-op로 정한 numeric tolerance와 smallest effect of interest, 사용한 no-op samples와 MAD estimator
- 모든 config/input/code SHA

SESOI 예시는 `max(0.05 nats/token, 5 × no-op MAD)`지만 값은 validation을 보기 전에 고정한다.

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/freeze_mechanistic_selection.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --discovery-root "$MECH_RUN_ROOT/discovery" \
  --components resid_post,attn_out,mlp_out,head_z,k_only,v_only,kv \
  --output "$MECH_RUN_ROOT/mechanistic_frozen_selection.json"

sha256sum "$MECH_RUN_ROOT/mechanistic_frozen_selection.json"
```

Hash를 lab note/commit에 남긴 뒤 folds 4–5의 **mechanistic activation/patch 결과**를 처음 연다.
이 12 scenarios의 seed-17 behavioral summary는 이미 관찰됐으므로 formal confirmation이라고
부르지 않는다. 결과를 보고 site/window/family를 바꾸면 그 run 전체를 exploratory로 표기하고
새 독립 holdout을 만든다.

```bash
# Frozen primary와 네 specificity/no-op control을 하나의 exact confirmatory grid로 먼저 동결한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/freeze_paid_scan_spec.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --role internal_validation \
  --selection "$MECH_RUN_ROOT/mechanistic_frozen_selection.json" \
  --confirmation-control-arms clean_stale,self,same_value_random,shuffled \
  --full-replays-per-cell 3 \
  --readout-steps-per-cell 128 \
  --output "$MECH_RUN_ROOT/internal_validation/paid_scan_spec.json"

mkdir -p "$MECH_RUN_ROOT/internal_validation/preflight"
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/assemble_readiness_evidence.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --scan-spec "$MECH_RUN_ROOT/internal_validation/paid_scan_spec.json" \
  --model-contract "$MECH_RUN_ROOT/preflight/model_contract/model_contract.json" \
  --model-run-identity "$MECH_RUN_ROOT/preflight/model_contract/run_identity.json" \
  --open-loop "$MECH_RUN_ROOT/gpu_canary/open_loop_validation.json" \
  --conversation-canary "$MECH_CONVERSATION_CANARY" \
  --gpu-canary "$MECH_RUN_ROOT/gpu_canary/gpu_measurements.json" \
  --canary-manifest "$MECH_RUN_ROOT/gpu_canary/canary_trials.jsonl" \
  --canary-encoded-manifest "$MECH_RUN_ROOT/gpu_canary/encoded_manifest.jsonl" \
  --output "$MECH_RUN_ROOT/internal_validation/preflight/readiness_evidence.json"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/assess_mechanistic_readiness.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --scan-spec "$MECH_RUN_ROOT/internal_validation/paid_scan_spec.json" \
  --evidence "$MECH_RUN_ROOT/internal_validation/preflight/readiness_evidence.json" \
  --output "$MECH_RUN_ROOT/internal_validation/preflight/paid_scan_authorization.json"

# Frozen readout와 probe도 새 site 탐색 없이 같은 split에 한 번 적용한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/score_readouts.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --readouts "$MECH_RUN_ROOT/preflight/model_contract/readouts.bound.json" \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --anchors "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --role internal_validation \
  --folds 4,5 \
  --output "$MECH_RUN_ROOT/internal_validation/baseline_readout.jsonl"

# 결과가 없는 pristine patch root에서 exact cell universe를 materialize한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/run_confirmatory_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --selection "$MECH_RUN_ROOT/mechanistic_frozen_selection.json" \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --anchors "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --readouts "$MECH_RUN_ROOT/preflight/model_contract/readouts.bound.json" \
  --baseline-readout "$MECH_RUN_ROOT/internal_validation/baseline_readout.jsonl" \
  --scan-spec "$MECH_RUN_ROOT/internal_validation/paid_scan_spec.json" \
  --readiness-go "$MECH_RUN_ROOT/internal_validation/preflight/paid_scan_authorization.json" \
  --control-arms clean_stale,self,same_value_random,shuffled \
  --role internal_validation \
  --folds 4,5 \
  --output-root "$MECH_RUN_ROOT/internal_validation" \
  --plan-only

# 첫 internal result cell보다 먼저 exact cell universe와 통계 family를 동결한다.
mkdir -p "$MECH_RUN_ROOT/internal_validation/analysis"
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/freeze_analysis_plan.py \
  --run-root "$MECH_RUN_ROOT" \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --template experiments/self_repair/mechanistic/config/internal_analysis_plan.template.json \
  --selection "$MECH_RUN_ROOT/mechanistic_frozen_selection.json" \
  --planned-cells "$MECH_RUN_ROOT/internal_validation/planned_cells.jsonl" \
  --output-spec "$MECH_RUN_ROOT/internal_validation/analysis/analysis_spec.json" \
  --output-expected-cells "$MECH_RUN_ROOT/internal_validation/analysis/expected_cells.json"

# Probe endpoint는 Step 13 설명처럼 causal selection과 exact coordinate가 호환될 때만 실행한다.
# 먼저 discovery와 internal-validation 양쪽에서 그 한 tensor coordinate만 별도 capture한다.
export MECH_PROBE_COORD_FILE="$MECH_RUN_ROOT/internal_validation/probe_coordinate.txt"
if .venv/bin/python - \
  "$MECH_RUN_ROOT/mechanistic_frozen_selection.json" > "$MECH_PROBE_COORD_FILE" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    selection = json.load(handle)
if selection.get("head") is not None or selection.get("component") not in {
    "resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"
}:
    raise SystemExit("selected causal site is not compatible with a non-head probe")
print(selection["component"])
print(selection["layer"])
print(selection["anchor"])
PY
then
  mapfile -t MECH_PROBE_COORD < "$MECH_PROBE_COORD_FILE"
  export MECH_PROBE_SITE="${MECH_PROBE_COORD[0]}"
  export MECH_PROBE_LAYER="${MECH_PROBE_COORD[1]}"
  export MECH_PROBE_ANCHOR="${MECH_PROBE_COORD[2]}"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/capture_activations.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --anchor-map "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --role discovery \
  --sites "$MECH_PROBE_SITE" \
  --layers "$MECH_PROBE_LAYER" \
  --anchors "$MECH_PROBE_ANCHOR" \
  --output-root "$MECH_RUN_ROOT/discovery/frozen_probe_capture" \
  --resume

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/fit_probes.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --capture-root "$MECH_RUN_ROOT/discovery/frozen_probe_capture" \
  --role discovery \
  --site-selection "$MECH_RUN_ROOT/mechanistic_frozen_selection.json" \
  --probe-grid \
  --sites "$MECH_PROBE_SITE" \
  --layers "$MECH_PROBE_LAYER" \
  --anchors "$MECH_PROBE_ANCHOR" \
  --freeze-output "$MECH_RUN_ROOT/discovery/current_destination_probe_frozen.json" \
  --output-root "$MECH_RUN_ROOT/discovery/frozen_probe_fit"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/capture_activations.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --anchor-map "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --role internal_validation \
  --sites "$MECH_PROBE_SITE" \
  --layers "$MECH_PROBE_LAYER" \
  --anchors "$MECH_PROBE_ANCHOR" \
  --output-root "$MECH_RUN_ROOT/internal_validation/probe_capture" \
  --resume

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/fit_probes.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --capture-root "$MECH_RUN_ROOT/internal_validation/probe_capture" \
  --role internal_validation \
  --site-selection "$MECH_RUN_ROOT/mechanistic_frozen_selection.json" \
  --frozen-probe "$MECH_RUN_ROOT/discovery/current_destination_probe_frozen.json" \
  --output-root "$MECH_RUN_ROOT/internal_validation/probes"
else
  unset MECH_PROBE_SITE MECH_PROBE_LAYER MECH_PROBE_ANCHOR
  echo "probe endpoint not_applicable: frozen primary is head/KV/path-specific"
fi

# 같은 planned_cells.jsonl을 유지한 채 모델 실행만 연다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/run_confirmatory_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --selection "$MECH_RUN_ROOT/mechanistic_frozen_selection.json" \
  --manifest "$MECH_RUN_ROOT/manifests/mechanistic_trials.jsonl" \
  --encoded-manifest "$MECH_RUN_ROOT/encoded_user_manifest.jsonl" \
  --anchors "$MECH_RUN_ROOT/anchor_map.jsonl" \
  --readouts "$MECH_RUN_ROOT/preflight/model_contract/readouts.bound.json" \
  --baseline-readout "$MECH_RUN_ROOT/internal_validation/baseline_readout.jsonl" \
  --scan-spec "$MECH_RUN_ROOT/internal_validation/paid_scan_spec.json" \
  --readiness-go "$MECH_RUN_ROOT/internal_validation/preflight/paid_scan_authorization.json" \
  --control-arms clean_stale,self,same_value_random,shuffled \
  --role internal_validation \
  --folds 4,5 \
  --output-root "$MECH_RUN_ROOT/internal_validation" \
  --resume

# Internal-validation 결과도 사전 동결된 4-term DID와 specificity family로 즉시 분석한다.
# Formal report와 섞이지 않도록 root report를 role-specific directory로 이동한다.
test ! -e "$MECH_RUN_ROOT/reports"
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/analyze_mechanistic_results.py \
  --run-root "$MECH_RUN_ROOT" \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --analysis-spec "$MECH_RUN_ROOT/internal_validation/analysis/analysis_spec.json" \
  --expected-cells "$MECH_RUN_ROOT/internal_validation/analysis/expected_cells.json"
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/render_mechanistic_report.py \
  --run-root "$MECH_RUN_ROOT"
mv "$MECH_RUN_ROOT/reports" "$MECH_RUN_ROOT/internal_validation/analysis_reports"
```

Internal causal gate:

- primary `clean_current - self` 4-term DID가 양수이고, two-sided scenario-cluster sign-flip의
  Holm-adjusted p ≤ .05이며 percentile scenario-cluster bootstrap CI lower bound가 frozen
  SESOI 0.1 nats/token보다 큼
- specificity family의 `clean_current - clean_stale`, `clean_current - same_value_random`,
  `clean_current - shuffled` 4-term DID 각각도 예상 부호, family 내 Holm p, 같은 frozen SESOI를 통과
- analyzer가 사전 동결된 모든 scenario contrast와 raw cell을 공개하고, 방향별 별도 효과나
  nonlinear max-control contrast를 계산하지 않았는데 통과했다고 주장하지 않음
- 사전 지정 cell이 모두 완료됨. OOM/NaN은 같은 identity로만 재시도하고 미해결이면 해당
  hypothesis를 `unevaluable`로 두며 available-case 분석이나 imputation을 하지 않음

Step 10의 clean capability/baseline gap, 개별 `delta_rescue`/`delta_stale`, `E_transfer`/`E_erase`,
방향별 sign은 중요한 기술·후속 진단이지만 현재 frozen analyzer의 네 DID registry와는 별도다.
별도 preregistered template과 구현 없이 이를 이 단계의 자동 GO 판정으로 보고하지 않는다.

통과하지 않아도 결과를 버리지 않고 null/unstable finding으로 보고한다.

---

## Step 17. 새 다중 도시 set에서 formal confirmation한다 `[TARGET]`

Step 2의 사람 검수를 마친 최소 4-city fully-crossed set에서 **새 site를 탐색하지 않고** frozen
circuit을 적용한다. 별도 immutable role manifest가 discovery와 formal confirmation의 whole
scenario template 및 whole ordered pair를 격리해야 한다.

```bash
export MECH_FORMAL_DATA_ROOT=/workspace/multivalue-controls
export MECH_FORMAL_ROOT="$MECH_RUN_ROOT/formal_confirmation"

# Reviewed multivalue source를 portable mechanistic manifest로 bind하고 먼저 fail-closed 검증한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/build_mech_manifest.py \
  --prepared-manifest "$MECH_FORMAL_DATA_ROOT/prepared_stimuli.jsonl" \
  --role-manifest "$MECH_FORMAL_DATA_ROOT/role_manifest.jsonl" \
  --audio-root "$MECH_FORMAL_DATA_ROOT" \
  --data-status reviewed_multivalue \
  --output "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/validate_multivalue_controls.py \
  --input-root "$MECH_FORMAL_DATA_ROOT" \
  --mechanistic-manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --require-independent-alignment \
  --require-double-listen-review

sha256sum "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl"
sha256sum "$MECH_FORMAL_DATA_ROOT/role_manifest.jsonl"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/build_anchor_map.py \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --prepared-manifest "$MECH_FORMAL_DATA_ROOT/prepared_stimuli.jsonl" \
  --output "$MECH_FORMAL_ROOT/anchor_map.jsonl" \
  --frame-trace-output "$MECH_FORMAL_ROOT/frame_trace.jsonl"

# 먼저 CPU-only source/identity 계약만 검사한다. 이 단계는 checkpoint를 load하지 않는다.
mkdir -p "$MECH_FORMAL_ROOT/preflight" "$MECH_FORMAL_ROOT/gpu_canary"
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/validate_mechanistic_contract.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --input-artifact-root "$MECH_FORMAL_DATA_ROOT" \
  --output-root "$MECH_FORMAL_ROOT/preflight/static_contract" \
  --model-repo kyutai/moshiko-pytorch-bf16 \
  --model-revision 2bfc9ae6e89079a5cc7ed2a68436010d91a3d289 \
  --dry-run

# CUDA가 확인된 후 exact model/readout contract를 한 번 bind한다.
nvidia-smi
.venv/bin/python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)'
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/validate_mechanistic_contract.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --input-artifact-root "$MECH_FORMAL_DATA_ROOT" \
  --output-root "$MECH_FORMAL_ROOT/preflight/model_contract" \
  --model-repo kyutai/moshiko-pytorch-bf16 \
  --model-revision 2bfc9ae6e89079a5cc7ed2a68436010d91a3d289

# Site/path는 절대 바꾸지 않고, 새 city verbalizer token ID만 formal bound readout으로 재결합한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/rebind_mechanistic_selection.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --selection "$MECH_RUN_ROOT/mechanistic_frozen_selection.json" \
  --readouts "$MECH_FORMAL_ROOT/preflight/model_contract/readouts.bound.json" \
  --output "$MECH_FORMAL_ROOT/formal_frozen_selection.json"

# Frozen primary + 사전 지정 controls의 formal paid grid를 결과 전에 동결한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/freeze_paid_scan_spec.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --role formal_confirmation \
  --selection "$MECH_FORMAL_ROOT/formal_frozen_selection.json" \
  --confirmation-control-arms clean_stale,self,same_value_random,shuffled \
  --full-replays-per-cell 3 \
  --readout-steps-per-cell 128 \
  --output "$MECH_FORMAL_ROOT/paid_scan_spec.json"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/estimate_mechanistic_workload.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --scan-spec "$MECH_FORMAL_ROOT/paid_scan_spec.json" \
  --output "$MECH_FORMAL_ROOT/preflight/workload_estimate.json"

# Formal source hash에 묶인 최대 4-row canary만 encode/replay한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/select_gpu_canary_manifest.py \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --output "$MECH_FORMAL_ROOT/gpu_canary/canary_trials.jsonl" \
  --max-trials 4 \
  --role formal_confirmation

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/encode_user_audio.py \
  --manifest "$MECH_FORMAL_ROOT/gpu_canary/canary_trials.jsonl" \
  --input-artifact-root "$MECH_FORMAL_DATA_ROOT" \
  --output-root "$MECH_FORMAL_ROOT/gpu_canary/encoded" \
  --output-manifest "$MECH_FORMAL_ROOT/gpu_canary/encoded_manifest.jsonl" \
  --model-revision 2bfc9ae6e89079a5cc7ed2a68436010d91a3d289 \
  --resume

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/validate_open_loop.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --encoded-manifest "$MECH_FORMAL_ROOT/gpu_canary/encoded_manifest.jsonl" \
  --output "$MECH_FORMAL_ROOT/gpu_canary/open_loop_validation.json"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/run_bounded_gpu_canary.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_FORMAL_ROOT/gpu_canary/canary_trials.jsonl" \
  --input-artifact-root "$MECH_FORMAL_DATA_ROOT" \
  --workload-estimate "$MECH_FORMAL_ROOT/preflight/workload_estimate.json" \
  --output "$MECH_FORMAL_ROOT/gpu_canary/gpu_measurements.json" \
  --layer 0

export MECH_FORMAL_FLOW_ROOT="$MECH_FORMAL_ROOT/gpu_canary/conversation"
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/run_full_duplex_validation.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_FORMAL_ROOT/gpu_canary/canary_trials.jsonl" \
  --encoded-manifest "$MECH_FORMAL_ROOT/gpu_canary/encoded_manifest.jsonl" \
  --anchors "$MECH_FORMAL_ROOT/anchor_map.jsonl" \
  --input-artifact-root "$MECH_FORMAL_DATA_ROOT" \
  --primary-intervention identity_noop \
  --donor-arms none \
  --seeds 17,29,42,101,2026 \
  --limit-trials 1 \
  --output-root "$MECH_FORMAL_FLOW_ROOT"
```

이 full-duplex 호출도 Step 11과 같이 두 startup mode에서 실제 첫 인사와 user 요청,
응답 끝을 모두 저장하고 blind template만 만든다. Formal source로 다시 두 사람이
독립 검수하고 불일치를 제3자가 조정한 후, 같은 명령에 `--resume --reviews ...`와 필요시
`--adjudications ...`를 추가한다. Formal manifest hash의
`conversation_canary.json`이 없으면 다음 full encode와 scan은 실행하지 않는다.

```bash
test -f "$MECH_FORMAL_FLOW_ROOT/conversation_canary.json"

# Bounded GPU/대화 gate가 열린 후에만 formal set 전체를 encode한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/encode_user_audio.py \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --input-artifact-root "$MECH_FORMAL_DATA_ROOT" \
  --output-root "$MECH_FORMAL_ROOT/encoded_user" \
  --output-manifest "$MECH_FORMAL_ROOT/encoded_user_manifest.jsonl" \
  --model-revision 2bfc9ae6e89079a5cc7ed2a68436010d91a3d289 \
  --resume

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/assemble_readiness_evidence.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --encoded-manifest "$MECH_FORMAL_ROOT/encoded_user_manifest.jsonl" \
  --scan-spec "$MECH_FORMAL_ROOT/paid_scan_spec.json" \
  --model-contract "$MECH_FORMAL_ROOT/preflight/model_contract/model_contract.json" \
  --model-run-identity "$MECH_FORMAL_ROOT/preflight/model_contract/run_identity.json" \
  --open-loop "$MECH_FORMAL_ROOT/gpu_canary/open_loop_validation.json" \
  --conversation-canary "$MECH_FORMAL_FLOW_ROOT/conversation_canary.json" \
  --gpu-canary "$MECH_FORMAL_ROOT/gpu_canary/gpu_measurements.json" \
  --canary-manifest "$MECH_FORMAL_ROOT/gpu_canary/canary_trials.jsonl" \
  --canary-encoded-manifest "$MECH_FORMAL_ROOT/gpu_canary/encoded_manifest.jsonl" \
  --output "$MECH_FORMAL_ROOT/preflight/readiness_evidence.json"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/assess_mechanistic_readiness.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --encoded-manifest "$MECH_FORMAL_ROOT/encoded_user_manifest.jsonl" \
  --scan-spec "$MECH_FORMAL_ROOT/paid_scan_spec.json" \
  --evidence "$MECH_FORMAL_ROOT/preflight/readiness_evidence.json" \
  --output "$MECH_FORMAL_ROOT/preflight/paid_scan_authorization.json"

# Formal rows에서는 transported frozen readout/site만 사용한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/score_readouts.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --readouts "$MECH_FORMAL_ROOT/preflight/model_contract/readouts.bound.json" \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --role-manifest "$MECH_FORMAL_DATA_ROOT/role_manifest.jsonl" \
  --encoded-manifest "$MECH_FORMAL_ROOT/encoded_user_manifest.jsonl" \
  --anchors "$MECH_FORMAL_ROOT/anchor_map.jsonl" \
  --role formal_confirmation \
  --output "$MECH_FORMAL_ROOT/baseline_readout.jsonl"

# Result cell을 하나도 생성하지 않고 formal cell universe를 먼저 고정한다.
mkdir -p "$MECH_FORMAL_ROOT/patches" "$MECH_RUN_ROOT/analysis"
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/run_confirmatory_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --selection "$MECH_FORMAL_ROOT/formal_frozen_selection.json" \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --role-manifest "$MECH_FORMAL_DATA_ROOT/role_manifest.jsonl" \
  --encoded-manifest "$MECH_FORMAL_ROOT/encoded_user_manifest.jsonl" \
  --anchors "$MECH_FORMAL_ROOT/anchor_map.jsonl" \
  --readouts "$MECH_FORMAL_ROOT/preflight/model_contract/readouts.bound.json" \
  --baseline-readout "$MECH_FORMAL_ROOT/baseline_readout.jsonl" \
  --scan-spec "$MECH_FORMAL_ROOT/paid_scan_spec.json" \
  --readiness-go "$MECH_FORMAL_ROOT/preflight/paid_scan_authorization.json" \
  --control-arms clean_stale,self,same_value_random,shuffled \
  --role formal_confirmation \
  --output-root "$MECH_FORMAL_ROOT/patches" \
  --plan-only

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/freeze_analysis_plan.py \
  --run-root "$MECH_RUN_ROOT" \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --template experiments/self_repair/mechanistic/config/analysis_plan.template.json \
  --selection "$MECH_FORMAL_ROOT/formal_frozen_selection.json" \
  --planned-cells "$MECH_FORMAL_ROOT/patches/planned_cells.jsonl" \
  --output-spec "$MECH_RUN_ROOT/analysis/analysis_spec.json" \
  --output-expected-cells "$MECH_RUN_ROOT/analysis/expected_cells.json"
```

Probe는 causal claim이 아닌 선택적 진단 endpoint다. Frozen causal site가 non-head tensor와 호환될
때만 calibration role에서 K-class classifier를 fit/freeze하고 formal role에 한 번 적용한다.
Probe 성능으로 site를 다시 고르지 않는다. 아래 조건문은 transported selection 자체에서 exact
coordinate를 다시 읽으며, head/KV/path-specific primary면 probe endpoint를 `not_applicable`로 남긴다.

```bash
export MECH_FORMAL_PROBE_COORD_FILE="$MECH_FORMAL_ROOT/probe_coordinate.txt"
if .venv/bin/python - \
  "$MECH_FORMAL_ROOT/formal_frozen_selection.json" > "$MECH_FORMAL_PROBE_COORD_FILE" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    selection = json.load(handle)
if selection.get("head") is not None or selection.get("component") not in {
    "resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"
}:
    raise SystemExit("transported causal site is not compatible with a non-head probe")
print(selection["component"])
print(selection["layer"])
print(selection["anchor"])
PY
then
  mapfile -t MECH_FORMAL_PROBE_COORD < "$MECH_FORMAL_PROBE_COORD_FILE"
  export MECH_PROBE_SITE="${MECH_FORMAL_PROBE_COORD[0]}"
  export MECH_PROBE_LAYER="${MECH_FORMAL_PROBE_COORD[1]}"
  export MECH_PROBE_ANCHOR="${MECH_FORMAL_PROBE_COORD[2]}"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/capture_activations.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --encoded-manifest "$MECH_FORMAL_ROOT/encoded_user_manifest.jsonl" \
  --anchor-map "$MECH_FORMAL_ROOT/anchor_map.jsonl" \
  --role multivalue_calibration \
  --sites "$MECH_PROBE_SITE" \
  --layers "$MECH_PROBE_LAYER" \
  --anchors "$MECH_PROBE_ANCHOR" \
  --output-root "$MECH_FORMAL_ROOT/probe_calibration_capture" \
  --resume

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/fit_probes.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --role-manifest "$MECH_FORMAL_DATA_ROOT/role_manifest.jsonl" \
  --capture-root "$MECH_FORMAL_ROOT/probe_calibration_capture" \
  --role multivalue_calibration \
  --site-selection "$MECH_FORMAL_ROOT/formal_frozen_selection.json" \
  --probe-grid \
  --sites "$MECH_PROBE_SITE" \
  --layers "$MECH_PROBE_LAYER" \
  --anchors "$MECH_PROBE_ANCHOR" \
  --freeze-output "$MECH_FORMAL_ROOT/multivalue_probe_frozen.json" \
  --output-root "$MECH_FORMAL_ROOT/probe_calibration"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/capture_activations.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --encoded-manifest "$MECH_FORMAL_ROOT/encoded_user_manifest.jsonl" \
  --anchor-map "$MECH_FORMAL_ROOT/anchor_map.jsonl" \
  --role formal_confirmation \
  --sites "$MECH_PROBE_SITE" \
  --layers "$MECH_PROBE_LAYER" \
  --anchors "$MECH_PROBE_ANCHOR" \
  --output-root "$MECH_FORMAL_ROOT/probe_capture" \
  --resume

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/fit_probes.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --role-manifest "$MECH_FORMAL_DATA_ROOT/role_manifest.jsonl" \
  --capture-root "$MECH_FORMAL_ROOT/probe_capture" \
  --role formal_confirmation \
  --site-selection "$MECH_FORMAL_ROOT/formal_frozen_selection.json" \
  --frozen-probe "$MECH_FORMAL_ROOT/multivalue_probe_frozen.json" \
  --output-root "$MECH_FORMAL_ROOT/probes"

sha256sum "$MECH_FORMAL_ROOT/multivalue_probe_frozen.json"
else
  echo "formal K-class probe endpoint not_applicable: frozen primary is head/KV/path-specific"
fi

# Frozen plan/spec/authorization을 전혀 바꾸지 않고 formal patch를 실행한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/run_confirmatory_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --selection "$MECH_FORMAL_ROOT/formal_frozen_selection.json" \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --role-manifest "$MECH_FORMAL_DATA_ROOT/role_manifest.jsonl" \
  --encoded-manifest "$MECH_FORMAL_ROOT/encoded_user_manifest.jsonl" \
  --anchors "$MECH_FORMAL_ROOT/anchor_map.jsonl" \
  --readouts "$MECH_FORMAL_ROOT/preflight/model_contract/readouts.bound.json" \
  --baseline-readout "$MECH_FORMAL_ROOT/baseline_readout.jsonl" \
  --scan-spec "$MECH_FORMAL_ROOT/paid_scan_spec.json" \
  --readiness-go "$MECH_FORMAL_ROOT/preflight/paid_scan_authorization.json" \
  --control-arms clean_stale,self,same_value_random,shuffled \
  --role formal_confirmation \
  --output-root "$MECH_FORMAL_ROOT/patches" \
  --resume
```

`unseen ordered pair`는 구성 도시 각각은 discovery에서 봤지만 특정 old→new 조합을 보지 않은
것이다. `unseen city`는 도시 identity 자체를 전혀 보지 않은 더 강한 별도 split이다. 둘을 섞어
표현하지 않고 old-city와 new-city별 macro effect와 CI를 함께 낸다.

GO 조건:

- 도시별 clean readout sign accuracy ≥80%
- same-new/different-old와 same-old/different-new counterfactual이 모두 존재
- held-out ordered pair에서 rescue/stale induction 방향이 유지
- token length/prior sensitivity에서 결론이 유지
- 호환되는 non-head primary일 때 calibration role에서 한 번 fit한 K-class
  `current_destination` probe가 formal role에 refit 없이 적용됨. `old_value_identity`,
  `old_was_mentioned`, `is_retracted`는 현재 manifest/probe schema가 독립 causal task로 구현하지
  않으므로 GO 조건으로 주장하지 않고 후속 endpoint로 남김
- Step 16과 동일한 frozen 4-DID, Holm/SESOI, exact missing-cell gate 통과. Clean capability,
  endogenous transfer/erasure, 방향별 sign 같은 추가 endpoint는 별도 preregistration 없이 formal
  GO를 통과한 것으로 보고하지 않음

실패하면 일반 stale-binding mechanism이 아니라 Boston–Seattle 또는 특정 lexical-value circuit으로
결론을 낮춘다.

---

## Step 18. 최종 후보만 정상 full-duplex에서 재검증한다 `[TARGET]`

Open-loop는 원인 진단의 주 분석이고, full-duplex는 ecological bridge다. 현재 runner의 frozen
primary pair는 다음 두 조건이다.

- 같은 repair snapshot에서 시작한 unpatched/no-op branch
- 같은 repair snapshot의 frozen `E_erase` 또는 candidate ablation branch — primary

`--donor-arms conditional_on_feedback_divergence`는 donor 결과 arm을 추가하는 옵션이 아니라,
clean/wrong donor를 해석할 수 있는 조건을 provenance에 고정하는 정책 이름이다. 현재 구현은 그런
donor output을 생성하지 않는다. 따라서 clean-current/wrong-target full-duplex donor 결과를 이
명령에서 얻었다고 보고하지 않는다. 그 둘은 별도 runner와 사전 동결 spec을 추가한 뒤에만
exploratory로 실행할 수 있다.

기존과 같은 5 seeds `[17, 29, 42, 101, 2026]`를 쓰고 response WAV, inner text, frame timing을
저장한다. seed를 독립 n으로 세지 않고 rendition별 successes/5로 먼저 묶는다.

Primary ecological bridge는 **동일한 repair streaming state를 intervention 직전에 두 개로
deep-clone**하고 한 branch에만 frozen within-recipient erasure/ablation을 적용한 뒤 둘 다
자연스럽게 free-run한다. 이렇게 해야 clean과 repair의 assistant history 차이를 donor effect로
오인하지 않는다.

각 condition pair의 `first_feedback_divergence_frame`을 기록한다. Clean/wrong donor arm은 patch
frame이 그 divergence보다 이를 때만 해석하고, 이미 history가 갈라졌으면 `unevaluable` 또는
exploratory로 둔다. Patch 시점까지 assistant feedback을 인위적으로 같게 만든 실행은 open-loop
sensitivity이지 “normal full-duplex”라고 부르지 않는다.

```bash
export MECH_FINAL_INPUT_ROOT="$MECH_RUN_ROOT/full_duplex/formal_canary_inputs"
export MECH_FINAL_CONVERSATION_ROOT="$MECH_RUN_ROOT/full_duplex/formal_bridge_001"
mkdir -p "$MECH_FINAL_INPUT_ROOT"

# Formal role 안에서 outcome을 보지 않는 deterministic matched pair를 최대 4개로 고정한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/select_gpu_canary_manifest.py \
  --manifest "$MECH_RUN_ROOT/manifests/multivalue_trials.jsonl" \
  --output "$MECH_FINAL_INPUT_ROOT/canary_trials.jsonl" \
  --max-trials 4 \
  --role formal_confirmation

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/encode_user_audio.py \
  --manifest "$MECH_FINAL_INPUT_ROOT/canary_trials.jsonl" \
  --input-artifact-root /workspace/multivalue-controls \
  --output-root "$MECH_FINAL_INPUT_ROOT/encoded" \
  --output-manifest "$MECH_FINAL_INPUT_ROOT/encoded_manifest.jsonl" \
  --model-revision 2bfc9ae6e89079a5cc7ed2a68436010d91a3d289 \
  --resume

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/run_full_duplex_validation.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --selection "$MECH_FORMAL_ROOT/formal_frozen_selection.json" \
  --manifest "$MECH_FINAL_INPUT_ROOT/canary_trials.jsonl" \
  --encoded-manifest "$MECH_FINAL_INPUT_ROOT/encoded_manifest.jsonl" \
  --anchors "$MECH_RUN_ROOT/formal_confirmation/anchor_map.jsonl" \
  --input-artifact-root /workspace/multivalue-controls \
  --primary-intervention within_repair_erasure \
  --donor-arms conditional_on_feedback_divergence \
  --seeds 17,29,42,101,2026 \
  --limit-trials 1 \
  --max-paired-cells 10 \
  --output-root "$MECH_FINAL_CONVERSATION_ROOT"
```

첫 실행은 pristine output root에 blind review template만 만든다. Step 11과 같은 두 사람 독립
검수·필요시 제3자 조정을 마친 뒤 같은 명령에 `--resume --reviews`와 필요시
`--adjudications`를 추가한다. 이 CLI는 한 호출당 최대
8 repair trial, 기본 16 paired cells로 제한되며 위 frozen primary는 1 repair × 5 seeds × 2 modes =
10 cells다. 더 많은 formal scenario는 각각 새 identity-specific output root로 사전 선언해 shard하고,
결과를 본 뒤 성공 사례만 고르지 않는다. Formal 새 자산이 아직 없으면 이 단계는 실행하지 않으며,
기존 v2 internal-validation pair로 돌린 결과는 provisional ecological bridge로만 표기한다.

두 intervention-blind annotator가 `final_target_correct`, `stale_state_error`, D1–D3 binding을
독립 판정하고 불일치는 제3자가 조정한다. agreement도 보고한다. Primary behavioral estimand는
scenario 동일 가중의 paired `patched - unpatched` final-target accuracy difference, key secondary는
stale-state-error difference다. Five seeds는 먼저 rendition별 successes/5로 집계하고,
scenario-cluster CI, frozen SESOI와 multiplicity family를 적용한다.

Held-out run에서 clean ≥80%, immediate-repair ≥70%, scorable primary windows ≥90%를 다시
요구한다. `greeting_suppressed`와 `common_handshake_then_request`를 분리 보고하며,
`natural_model_start`는 diagnostic-only로 따로 둔다. Logit
margin만 좋아지고 audible response가 좋아지지 않으면 “text-logit surrogate에 causal
contribution”까지만 말한다.

기존 behavioral runner로 남은 2,400개를 재개하는 것은 이 단계의 turn-taking smoke와 capability
gate가 통과한 뒤에만 결정한다.

---

## Step 19. 분석·그림·보고서를 만든다 `[TARGET]`

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/analyze_mechanistic_results.py \
  --run-root "$MECH_RUN_ROOT" \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --analysis-spec "$MECH_RUN_ROOT/analysis/analysis_spec.json" \
  --expected-cells "$MECH_RUN_ROOT/analysis/expected_cells.json"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/render_mechanistic_report.py \
  --run-root "$MECH_RUN_ROOT"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/verify_mechanistic_run.py \
  --run-root "$MECH_RUN_ROOT"
```

최소 산출물:

```text
<run-root>/
  encoded_user_manifest.jsonl
  anchor_map.jsonl
  frame_trace.jsonl
  baseline_readout.jsonl
  artifact_sha256.json
  analysis/
    analysis_spec.json
    expected_cells.json
  preflight/model_contract/
    run_identity.json
    environment.json
    input_hash_manifest.jsonl
    model_contract.json
    readouts.bound.json
  gpu_canary/
    open_loop_validation.json
    gpu_measurements.json
    conversation/conversation_canary.json
  discovery_baseline/
    capture_manifest.jsonl
  discovery/
    residual-*/residual_patch_results.jsonl
    components/component_patch_results.jsonl
    kv/kv_patch_results.jsonl
    paths/path_patch_results.jsonl
    probe_metrics.json
  internal_validation/
    analysis/analysis_spec.json
    analysis/expected_cells.json
    analysis_reports/MECHANISTIC_RESULTS.md
    patch_results.jsonl
    planned_cells.jsonl
    failures.jsonl
    resume_summary.json
    probe_metrics.json
    baseline_readout.jsonl
    metrics.json
  formal_confirmation/
    patches/patch_results.jsonl
    patches/planned_cells.jsonl
    formal_frozen_selection.json
    paid_scan_spec.json
    baseline_readout.jsonl
    multivalue_probe_frozen.json
    probes/probe_metrics.json
    patches/failures.jsonl
    patches/resume_summary.json
    patches/metrics.json
  full_duplex/
    validation.jsonl
  reports/
    MECHANISTIC_RESULTS.md
    mechanistic_discovery_summary.json
    tables/all_scenario_effects.csv
    tables/multiplicity_registry.csv
    figures/01_baseline_margin.svg
    figures/02_probe_layer-time.svg
    figures/03_residual_patch.svg
    figures/04_frozen_confirmation.svg
    figures/05_temporal_propagation.svg
    figures/06_controls_and_noops.svg
```

모든 report JSON에는 `analysis_status`, analyzer version, counts, provenance, gates, limitations,
모든 input SHA를 넣는다. 성공 사례만 고르지 말고 `all_scenario_effects.csv`에 전체 scenario를
공개한다. `multiplicity_registry.csv`에는 hypothesis/family/direction/statistic/n/raw p/adjusted p/
CI type/SESOI/pass-fail을 기록한다. Confirmation 그림은 preregistered point만 forest로 그리고,
전체 confirmation heatmap을 추가하면 `POST-CONFIRMATORY EXPLORATORY` watermark를 붙여 site
선택에 사용하지 않는다.

### 결과를 읽는 순서

1. `preflight/model_contract/run_identity.json`: code/model/data가 의도한 실행인지 확인
2. `gpu_canary/open_loop_validation.json`: replay/self-patch gate 확인
3. baseline figure: clean capability와 repair gap 확인
4. discovery heatmap: 후보를 찾은 위치를 보되 확증 p-value로 해석하지 않음
5. internal/formal confirmation forest: preregistered effect, CI, Holm p, SESOI 확인
6. controls figure: no-op/clean-stale/same-value/shuffled effect 확인
7. temporal propagation: writer→dependency/query path가 이어지는지 확인
8. full-duplex table: 실제 audible behavior에 연결되는지 확인
9. limitations: provisional audio, speaker/value generalization 범위를 확인

### 허용되는 결론 문장

- Probe만: “L17 query-time activation에서 originally mentioned city가 held-out scenarios에서
  선형 decode되었다.”
- Causal confirmation까지: “Frozen L17Hk intervention이 target-vs-stale margin을 X
  nats/token만큼 변화시켰다(95% CI …, Holm p=…).”
- 모든 gate 통과: “결과는 L17Hk가 이 task/distribution에서 stale destination binding의
  유지·readout에 인과적으로 기여한다는 해석과 일치한다.”

“유일한 Boston memory head”, “Seattle이 과거 state를 덮어썼다”, “근본 원인”, 다른 언어·slot·
speaker로의 일반화는 별도 증거 없이 쓰지 않는다.

---

## Step 20. RunPod에서 결과를 바로 본다

Markdown과 SVG는 terminal에서 간단히 확인할 수 있다.

```bash
cd "$MECH_RUN_ROOT/reports"
sed -n '1,240p' MECHANISTIC_RESULTS.md
find figures -maxdepth 1 -type f -name '*.svg' -print | sort
```

로컬 브라우저로 볼 때는 RunPod terminal에서 read-only HTTP server를 열고, RunPod가 제공한
TCP/HTTP port mapping을 사용한다.

```bash
cd "$MECH_RUN_ROOT/reports"
python3 -m http.server 8000 --bind 0.0.0.0
```

민감한 artifact가 있으면 public port를 열지 말고 SSH port forwarding을 쓴다.

```bash
ssh -p <SSH_PORT> -L 8000:127.0.0.1:8000 root@<SSH_HOST>
```

그 뒤 로컬에서 `http://127.0.0.1:8000/`을 연다. Server를 열기 전에 report directory에 raw
audio, private blind map, credential이 없는지 확인한다.
HTTP server는 별도 terminal/tmux pane에서 실행한다. 같은 terminal을 계속 쓸 때는 `Ctrl-C`로
server를 중지한 뒤 Step 21로 간다.

---

## Step 21. 결과를 검증·압축·회수한다 `[TARGET]`

Verifier가 성공한 뒤 작은 보고서 package와 private large-artifact package를 분리한다.

대형 private tarball은 원본과 압축본이 한 volume에 동시에 있어 공간을 거의 두 배 쓸 수 있다.
먼저 `du -sh "$MECH_RUN_ROOT"`와 `df -h /workspace`로 여유 공간을 확인한다. 부족하면
scenario shard별로 압축해 즉시 object storage/local로 전송하거나, tar stream을 SSH로 직접
보내고 전체 archive를 같은 volume에 만들지 않는다.

```bash
cd "$MECH_REPO_ROOT"
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/package_mechanistic_results.py \
  --run-root "$MECH_RUN_ROOT" \
  --public-output /workspace/mechanistic-report-package.tar.gz \
  --private-output /workspace/mechanistic-large-artifacts.tar.gz

sha256sum /workspace/mechanistic-report-package.tar.gz
sha256sum /workspace/mechanistic-large-artifacts.tar.gz
```

로컬로 회수한다.

```bash
scp -P <SSH_PORT> \
  root@<SSH_HOST>:/workspace/mechanistic-report-package.tar.gz .
scp -P <SSH_PORT> \
  root@<SSH_HOST>:/workspace/mechanistic-large-artifacts.tar.gz .
```

GitHub에는 다음만 올린다.

- 코드, tests, pinned config/schema
- frozen selection과 run identity에서 secret/private URI를 제거한 사본
- 작은 CSV/JSON summary, final Markdown, SVG
- large/private artifact의 relative URI와 SHA-256 manifest

다음은 Git에 넣지 않는다.

- model weights와 HF cache
- WAV/MP3와 full activation/QKV/KV tensors
- credential, 절대 로컬 path, private blind map
- 수 GB JSONL 또는 safetensors shard

Commit 전 `git status --short`, `git diff --check`, secret scan, report verifier를 실행한다. 대형
artifact가 staging됐으면 commit하지 말고 `.gitignore` 정책부터 수정한다.

Harness 구현 commit은 `codex/mechanistic-stale-binding-harness` branch에서 검증한 뒤
`self-revision` fork의 같은 이름 branch에 올린다. 이 단계에서는 user가 보관한 untracked WAV,
archive, HTML, 연구 그림을 절대 stage하지 않는다. `git add`에는 이번 변경 파일만 explicit
allowlist로 적고 directory 전체를 넣지 않는다. 실제 GPU 결과는 나중에 별도의 reviewed result
commit으로 추가한다.

```bash
cd "$MECH_REPO_ROOT"
test "$(git branch --show-current)" = codex/mechanistic-stale-binding-harness

git add \
  .gitignore \
  experiments/self_repair/MECHANISTIC_STALE_BINDING_RUNPOD.md \
  experiments/self_repair/requirements-mechanistic.txt \
  experiments/self_repair/mechanistic/.gitignore \
  experiments/self_repair/mechanistic/LOCAL_VALIDATION.md \
  experiments/self_repair/mechanistic/README.md \
  experiments/self_repair/mechanistic/__init__.py \
  experiments/self_repair/mechanistic/config \
  experiments/self_repair/mechanistic/core.py \
  experiments/self_repair/mechanistic/runtime.py \
  experiments/self_repair/mechanistic/analysis_protocol.py \
  experiments/self_repair/mechanistic/audio_activity.py \
  experiments/self_repair/mechanistic/blinding.py \
  experiments/self_repair/mechanistic/causal_scan.py \
  experiments/self_repair/mechanistic/conversation.py \
  experiments/self_repair/mechanistic/probes.py \
  experiments/self_repair/mechanistic/readiness.py \
  experiments/self_repair/mechanistic/response_window.py \
  experiments/self_repair/mechanistic/verification.py \
  experiments/self_repair/mechanistic/runpod \
  experiments/self_repair/mechanistic/schemas \
  experiments/self_repair/mechanistic/scripts \
  experiments/self_repair/mechanistic/tests \
  experiments/self_repair/mechanistic/manifests/.gitkeep \
  experiments/self_repair/mechanistic/reports/.gitkeep \
  moshi/moshi/models/lm.py \
  moshi/moshi/modules/transformer.py

git diff --cached --name-only
git diff --cached --check
git status --short
git commit -m "Harden mechanistic stale-binding GPU workflow"
git remote -v
test "$(git remote get-url self-revision)" = \
  https://github.com/mingunkim123/Moshi_self_revision.git
git push -u self-revision codex/mechanistic-stale-binding-harness
```

실제 report를 GitHub에 추가할 때에는 packaging tool이 private URI/absolute path를 제거해 생성한
작은 public 파일만 별도 allowlist로 stage한다. 출력이 아직 `$MECH_RUN_ROOT/reports`에만 있다면
검증되지 않은 파일을 repository에 복사하지 않는다.

---

## Checkpoint/resume 계약

각 patch cell identity는 다음 hash로 만든다.

```text
model revision + code commit + input hash + open-loop policy hash
+ donor trial id + recipient trial id + component/layer/head
+ source/target frame span + readout hash
```

- 한 GPU shard는 하나의 output directory만 소유한다.
- scenario 단위로 shard한다.
- 각 cell은 먼저 atomic JSON으로 쓰고 검증 후 JSONL로 merge한다.
- 완료 row가 있으면 identity와 artifact hash를 검증한 뒤 skip한다.
- OOM, NaN, shape/time mismatch도 지우지 않고 failure row로 남긴다.
- 다른 identity의 결과를 같은 directory에 섞지 않는다.
- prefix branching 때 mutable streaming/KV state를 얕은 복사하지 않는다.
- prefix는 재실행할 수 있어도 완료 patch cell을 다시 실행하지 않는다.

## 자주 생기는 실패와 해석 오류

- `NO_CUDA_GRAPH=1`을 늦게 설정해 hook가 stale/미실행됨
- clean/repair의 raw frame 번호를 그대로 맞춰 acoustic/position을 patch함
- post-RoPE K를 다른 absolute position으로 직접 복사함
- seed/frame을 독립 n으로 세어 CI를 부당하게 좁힘
- discovery와 confirmation scenario가 섞임
- confirmation을 본 뒤 head/window/family를 바꿈
- BF16 activation으로 probe optimizer를 그대로 학습함
- NR의 거의 0인 denominator를 보고하지 않음
- generic activation damage를 destination-specific effect로 해석함
- Boston이 decode된다는 사실 자체를 stale error로 해석함
- assistant inner text 개선을 audible answer 개선으로 해석함
- provisional audio를 confirmatory/publication-ready라고 서술함
- seed-17 floor와 early greeting을 무시하고 남은 2,400개부터 생성함

## 관련 저장소 문서

- [`dataset_v2/README.md`](dataset_v2/README.md): v2 생성·검증·artifact 상태
- [`dataset_v2/ANALYSIS_PROTOCOL.md`](dataset_v2/ANALYSIS_PROTOCOL.md): 기존 behavioral 분석 규칙
- [`dataset_v2/reports/SEED17_PILOT_ANALYSIS.md`](dataset_v2/reports/SEED17_PILOT_ANALYSIS.md):
  floor/turn-taking pilot 결과
- [`dataset_v2/DATASET_CARD.md`](dataset_v2/DATASET_CARD.md): 데이터 용도와 한계
- [`moshi/moshi/models/lm.py`](../../moshi/moshi/models/lm.py): LMGen, streaming step, text logits
- [`moshi/moshi/modules/transformer.py`](../../moshi/moshi/modules/transformer.py): transformer layers,
  SDPA/head concat, RingKVCache
- [`moshi/moshi/utils/compile.py`](../../moshi/moshi/utils/compile.py): compile/CUDA graph switch

## 방법론 참고문헌

- [Moshi: a speech-text foundation model for real-time dialogue](https://kyutai.org/assets/pdfs/Moshi.pdf)
- [Towards Best Practices of Activation Patching in Language Models](https://arxiv.org/abs/2309.16042)
- [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255)
- [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593)
- [Localizing Model Behavior with Path Patching](https://arxiv.org/abs/2304.05969)
- [The Hydra Effect: Emergent Self-repair in Language Model Computations](https://arxiv.org/abs/2307.15771)
- [Causal Tracing / Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- [How do Language Models Bind Entities in Context?](https://arxiv.org/abs/2310.17191)
- [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/)

## 최종 체크리스트

- [ ] harness implementation/tests가 clean commit에 있음
- [ ] protocol, split, readout, tolerance, SESOI가 confirmation 전에 hash로 동결됨
- [ ] model/code/config/input identity가 모두 기록됨
- [ ] provisional와 confirmatory artifact가 섞이지 않음
- [ ] open-loop replay/self-patch/cache-reset gate 통과
- [ ] clean capability와 repair gap gate 통과
- [ ] residual → component/head → KV/path 순으로 discovery 수행
- [ ] frozen sites만 folds 4–5 internal validation에서 한 번 검정
- [ ] 새 role manifest의 4-city fully-crossed held-out-pair formal confirmation 수행
- [ ] full-duplex five-seed blinded behavioral bridge 수행
- [ ] scenario-cluster CI와 multiplicity correction 적용
- [ ] controls/failure rows/all-scenario table 포함
- [ ] report verifier와 SHA-256 package 검증 통과
- [ ] Git에는 작은 재현 artifact만 push
