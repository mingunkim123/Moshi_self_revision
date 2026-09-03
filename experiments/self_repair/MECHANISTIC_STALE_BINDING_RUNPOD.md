# Moshi mechanistic stale-binding experiment: RunPod end-to-end runbook

- Status: **실행 런북 겸 구현 사양**
- Last audited code baseline: `77452e214e63cfe35fe6dca2f6adf4fb838d49dc`
- Target repository: `https://github.com/mingunkim123/Moshi_self_revision` (`self-revision/main`)
- Pinned model for comparison with the existing pilot:
  `kyutai/moshiko-pytorch-bf16@2bfc9ae6e89079a5cc7ed2a68436010d91a3d289`

## 먼저 읽을 결론

이 실험은 **모델 구조상 가능**하다. Moshiko의 main Temporal Transformer는 32개 layer,
layer당 32개 attention head, hidden size 4,096이고 layer residual과 streaming KV cache에
접근할 수 있다. 다만 현재 저장소에는 behavioral runner만 있고 activation capture, head/KV
intervention, strict teacher-forced open-loop, probe, mechanistic report 코드는 없다.

따라서 이 문서의 뒤쪽 명령을 지금 checkout에서 그대로 실행하면 안 된다. 아래 두 종류의
명령을 구분한다.

- **`[EXISTS]`**: 현재 저장소에 존재하고 실행할 수 있다.
- **`[TARGET]`**: 이 문서에 정의한 mechanistic harness를 구현·테스트·commit한 뒤 실행한다.

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
5. wrong-time, same-value, shuffled-donor, no-op 같은 control로 일반적인 activation 손상을
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
| 9. v2 internal validation | 12-scenario effects/CI/p-values | 양방향 효과와 controls 통과 | exploratory로 제한 |
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
예상 방향이 맞는지 확인하는 component gate다. 이것만으로는 repair run 안에 있던 endogenous
stale state를 찾았다고 할 수 없으므로 다음 중 하나도 discovery에서 선택해 freeze한다.

```text
E_transfer(S) = M(clean-current <- repair at S)
              - M(clean-current <- mention-only at S)

E_erase(S)    = M(repair with frozen old-subspace/path erasure at S) - M(repair)
```

`E_transfer < 0` 또는 `E_erase > 0`가 matched cue/mention control보다 재현돼야 “stale binding의
유지/readout에 기여”라는 표현을 쓴다. 그렇지 않으면 “이 site의 destination-value intervention이
readout을 인과적으로 바꾼다”로 제한한다.

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

## Step 1. RunPod에 올리기 전에 mechanistic harness를 구현한다 `[TARGET]`

### 1.1 Definition of Ready 파일 구조

다음 파일들은 **현재 아직 존재하지 않는다**. 이후 모든 `[TARGET]` 명령은 이 구조가 commit된
상태를 전제로 한다.

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

# Power gate가 통과한 뒤에만 source audio와 timing manifest를 만든다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/build_multivalue_controls.py \
  --city-config experiments/self_repair/mechanistic/config/multivalue_cities.json \
  --scenario-blueprints experiments/self_repair/dataset_v2/blueprints/scenarios.jsonl \
  --output-root /workspace/multivalue-controls

# 이 사이에 intervention-blind 사람 이중 청취와 불일치 조정을 기록한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/validate_multivalue_controls.py \
  --input-root /workspace/multivalue-controls \
  --require-independent-alignment \
  --require-double-listen-review
```

---

## Step 3. RunPod Pod와 persistent storage를 준비한다

권장 사양은 A100/H100 80 GB다. A40/A6000/L40S 48 GB도 batch 1과 선택적 capture로 가능하지만
full QKV/gradient scan은 더 자주 offload/OOM된다. 기존 behavioral inference의 24 GB 요구량을
mechanistic full scan 요구량으로 착각하지 않는다.

- Persistent volume: 최소 100 GB, 권장 150–200 GB
- Volume mount: `/workspace`
- Container: 지원되는 CUDA/PyTorch 이미지, Python 3.10–3.14
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
export MECH_ARTIFACT_ROOT="/workspace/mech-artifacts/$MECH_RUN_ID"
export HF_HOME=/workspace/hf-cache
export NO_TORCH_COMPILE=1
export NO_CUDA_GRAPH=1

mkdir -p "$MECH_ARTIFACT_ROOT"
test -z "$(git status --porcelain --untracked-files=no)"
git rev-parse HEAD
```

Clone한 저장소에서 remote 이름은 보통 `origin`이지만 URL은 이 fork다. 출력 SHA를 run ID에
기록하고, mechanistic harness가 들어간 정확한 commit인지 확인한다.

환경을 만든다.

```bash
cd "$MECH_REPO_ROOT"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/python -m pip install -e ./moshi
.venv/bin/python -m pip install -r experiments/self_repair/requirements-v2.txt

# [TARGET] 구현 후 version-pinned 분석/계측 의존성을 설치한다.
.venv/bin/python -m pip install \
  -r experiments/self_repair/requirements-mechanistic.txt
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
tar -tzf /workspace/moshi_v2_provisional_runpod_payload_b8d30fd.tar.gz
tar -tzf /workspace/moshi_v2_seed17_pilot_b977391.tar.gz
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

`tar -tzf` 출력은 첫 archive가 `dataset_v2/artifacts/provisional_prepared/`, tracked config,
old evaluation note/manifest만, 두 번째 archive가 corrected manifest 하나만 포함하는지 확인한다.
다른 top-level path나 `..` member가 보이면 extract하지 않는다.

Archive 내부에 해당 corrected manifest가 없으면 두 번째 archive가 올바르게 풀렸는지 확인한다.
새 mechanistic run에서는 old manifest의 `code_commit`을 그대로 재사용하지 말고, audio source
identity를 참조해 **현재 clean harness commit으로 새 mechanistic manifest**를 만든다.

```bash
mkdir -p "$MECH_ARTIFACT_ROOT/manifests"

# [TARGET]
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/build_mech_manifest.py \
  --source-eval-manifest \
    experiments/self_repair/dataset_v2/evaluation/provisional_eval_trials_seedpilot_b977391.jsonl \
  --prepared-manifest /workspace/provisional_prepared_stimuli.jsonl \
  --analysis-folds experiments/self_repair/dataset_v2/assignments/analysis_folds.jsonl \
  --data-status exploratory_provisional \
  --output "$MECH_ARTIFACT_ROOT/manifests/mechanistic_trials.jsonl"
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
  --manifest "$MECH_ARTIFACT_ROOT/manifests/mechanistic_trials.jsonl" \
  --input-artifact-root "$MECH_DATA_ROOT" \
  --output-root "$MECH_ARTIFACT_ROOT/preflight" \
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

## Step 7. user WAV를 Mimi code로 한 번만 변환한다 `[TARGET]`

반복 patch마다 audio를 다시 encode하면 시간도 낭비되고 codec state leakage를 놓치기 쉽다.
WAV hash를 key로 user-side 8-codebook tensor를 한 번 만들고 safetensors shard로 저장한다.

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/encode_user_audio.py \
  --manifest "$MECH_ARTIFACT_ROOT/manifests/mechanistic_trials.jsonl" \
  --output-root "$MECH_ARTIFACT_ROOT/encoded_user" \
  --output-manifest "$MECH_ARTIFACT_ROOT/encoded_user_manifest.jsonl" \
  --model-revision 2bfc9ae6e89079a5cc7ed2a68436010d91a3d289 \
  --resume
```

각 manifest row에는 source WAV SHA, sample/frame 수, tensor shape/dtype/hash, Mimi revision,
prefix silence를 넣는다. 일부 입력을 두 번 encode해 byte-identical tensor인지 확인하고,
모든 trial 사이에서 codec/model streaming state를 reset한다.

GO 조건: expected coverage 100%, duplicate/missing/hash mismatch 0, repeated-encode mismatch 0.

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

LM delay 1 frame을 state/readout index에 어떻게 반영했는지 config에 명시한다. repair cue 전체를
하나로 묶으면 new city와 뒤의 repeated old city가 섞이므로 `new_value`와
`repeated_old` span을 반드시 분리한다. clean/repair는 길이가 다르므로 같은 raw frame 번호가
아니라 semantic anchor끼리 patch한다.

```bash
# [TARGET]
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/build_anchor_map.py \
  --manifest "$MECH_ARTIFACT_ROOT/manifests/mechanistic_trials.jsonl" \
  --prepared-manifest /workspace/provisional_prepared_stimuli.jsonl \
  --output "$MECH_ARTIFACT_ROOT/anchor_map.jsonl" \
  --frame-trace-output "$MECH_ARTIFACT_ROOT/frame_trace.jsonl"
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
  --encoded-manifest "$MECH_ARTIFACT_ROOT/encoded_user_manifest.jsonl" \
  --output "$MECH_ARTIFACT_ROOT/open_loop_validation.json"
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
  --readouts experiments/self_repair/mechanistic/config/readouts.json \
  --manifest "$MECH_ARTIFACT_ROOT/manifests/mechanistic_trials.jsonl" \
  --anchors "$MECH_ARTIFACT_ROOT/anchor_map.jsonl" \
  --role discovery \
  --output "$MECH_ARTIFACT_ROOT/baseline_readout.jsonl"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/capture_activations.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --role discovery \
  --sites logits,resid_post \
  --anchors query_end,D1_end,D2_end,D3_end \
  --output-root "$MECH_ARTIFACT_ROOT/discovery_baseline" \
  --resume
```

GO 권고 기준:

- clean current-value sign accuracy ≥80%
- Boston→Seattle과 Seattle→Boston 방향별 sign accuracy ≥80%
- 한 도시 prior만으로 성공하지 않음
- clean-current와 repair의 margin gap CI가 0을 벗어나고, causal rescue를 평가할 충분한 pair가 있음
- root와 dependency readout이 neutral relation을 무차별적으로 바꾸지 않음
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

이 smoke에는 folds 4–5와 새 formal-confirmation data를 절대 넣지 않는다. Patch effect의 기대
부호를 확인하는 test는 정답을 아는 analytic toy/constructed fixture에서만 수행한다. 실제 한
scenario가 원하는 부호를 낼 때까지 구현을 조정하지 않는다.

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/scan_residual_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --role smoke \
  --limit-scenarios 1 \
  --layers 0,15,31 \
  --anchors new_end,query_end \
  --controls self,current,wrong,shuffled \
  --output-root "$MECH_ARTIFACT_ROOT/smoke" \
  --resume
```

GO 조건: finite output, deterministic replay, self-patch no-op, cache position 일치,
atomic row write, kill/restart 후 resume 중복 0. Peak VRAM, cell/sec, suffix runtime, output bytes를
기록해 full budget을 산정한다.

```text
estimated GPU hours = number_of_patch_cells × mean_suffix_seconds / 3600
```

기존 seed-17 free-running 600개는 약 4.99 GPU-hours였지만 mechanistic 비용은 trial 수보다
patch cell 수와 replay suffix 길이에 좌우되므로 이 값을 그대로 외삽하지 않는다.

---

## Step 12. residual-stream coarse localization을 실행한다 `[TARGET]`

Discovery folds 1–3에서만 32 layers × frozen semantic anchors를 훑는다. 처음부터 1,024 heads ×
모든 frame을 실행하지 않는다.

Recipient는 repair run이다. 주요 donor는 같은 scenario/speaker/current destination의
clean-current run이다. 같은 text bundle의 same-direction clean-current는 speaker matched다.
반대 destination의 clean-stale donor는 speaker가 달라질 수 있으므로 manifest에서 일치 여부를
확인하고, 맞지 않으면 matched donor라고 부르지 않으며 새 control set을 사용한다.

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/scan_residual_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --role discovery \
  --layers 0:32 \
  --anchors old_end,cue_end,new_end,D1_end,D2_end,D3_end,query_end \
  --donors clean_current,clean_stale,self,same_value_random,shuffled \
  --output-root "$MECH_ARTIFACT_ROOT/discovery/residual" \
  --resume
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

필수 controls:

- self/no-op
- same-value random donor
- wrong destination donor
- wrong layer/time
- shuffled scenario/speaker donor
- neutral N1–N3 readout
- activation norm 및 out-of-distribution 진단

---

## Step 13. probe는 진단용으로만 학습한다 `[TARGET]`

Probe task를 섞지 않는다.

- `current_destination`
- `old_value_identity` (K-class)
- `old_was_mentioned` (boolean)
- `is_retracted`
- D1/D2/D3의 `bound_to_current`

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/fit_probes.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --role discovery \
  --group-by scenario_id \
  --output-root "$MECH_ARTIFACT_ROOT/discovery/probes"
```

Layer×anchor별 L2 linear/logistic probe를 따로 학습한다. Discovery 내부 nested group CV로
regularization을 선택하고, scaling은 train fold에만 fit한다. balanced accuracy, AUROC, Brier,
scenario-bootstrap CI를 낸다.

필수 비교는 majority, shuffled-label, Mimi/layer-0, timing/duration-only, random projection,
parameter-count-matched control, clean→repair cross-condition transfer다. 한 vector의 여러 frame을
독립 sample로 부풀리지 않는다. v2만 썼다면 “old memory를 찾았다”가 아니라
“Boston-vs-Seattle axis가 decode되었다”고 쓴다.

Discovery가 끝나면 선택된 task/layer/anchor, preprocessing, coefficients, class mapping, threshold를
selection JSON에 freeze한다. Step 16에서 folds 4–5에 정확히 한 번 적용하고 selected probe
endpoint를 별도 Holm family로 보고한다. Full confirmation heatmap을 새 site 선택에 쓰지 않는다.

---

## Step 14. 후보 layer에서 attention/MLP/head를 좁힌다 `[TARGET]`

Residual scan에서 선택한 소수 layer/time만 분해한다.

1. aggregate attention output
2. MLP output
3. output projection 전 head `z`
4. mean ablation 및 donor patch

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/scan_component_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --role discovery \
  --selection "$MECH_ARTIFACT_ROOT/discovery/residual_candidates.json" \
  --components attn_out,mlp_out,head_z \
  --controls self,current,wrong,same_value_random,shuffled \
  --output-root "$MECH_ARTIFACT_ROOT/discovery/components" \
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

후보 head에만 다음 순서로 실행한다.

1. old-value frame K-only / V-only / K+V
2. pre-repair D1–D3 K/V
3. new-value와 repair-cue K/V
4. query head가 어느 source window를 인과적으로 읽는지
5. writer→KV→query-reader의 제한 path

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/scan_kv_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --role discovery \
  --selection "$MECH_ARTIFACT_ROOT/discovery/component_candidates.json" \
  --modes k_only,v_only,kv \
  --output-root "$MECH_ARTIFACT_ROOT/discovery/kv" \
  --resume

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/run_path_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --role discovery \
  --selection "$MECH_ARTIFACT_ROOT/discovery/kv_candidates.json" \
  --output-root "$MECH_ARTIFACT_ROOT/discovery/paths" \
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

Discovery에서 다음을 `mechanistic_frozen_selection.json`에 기록한다.

- top K sites와 tie-break 규칙; 권장 K=4 이하
- primary single site/path 또는 joint circuit과 individual secondary sites
- joint intervention의 tensor site, 적용 순서, 조합 규칙. 서로 다른 serial layer의 full residual을
  차례로 덮어쓰는 구성은 앞 개입이 무효화될 수 있으므로 primary joint circuit으로 쓰지 않음
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
  --discovery-root "$MECH_ARTIFACT_ROOT/discovery" \
  --output "$MECH_ARTIFACT_ROOT/mechanistic_frozen_selection.json"

sha256sum "$MECH_ARTIFACT_ROOT/mechanistic_frozen_selection.json"
```

Hash를 lab note/commit에 남긴 뒤 folds 4–5의 **mechanistic activation/patch 결과**를 처음 연다.
이 12 scenarios의 seed-17 behavioral summary는 이미 관찰됐으므로 formal confirmation이라고
부르지 않는다. 결과를 보고 site/window/family를 바꾸면 그 run 전체를 exploratory로 표기하고
새 독립 holdout을 만든다.

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/run_confirmatory_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --selection "$MECH_ARTIFACT_ROOT/mechanistic_frozen_selection.json" \
  --role internal_validation \
  --folds 4,5 \
  --output-root "$MECH_ARTIFACT_ROOT/internal_validation" \
  --resume

# Frozen readout와 probe도 새 site 탐색 없이 같은 split에 한 번 적용한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/score_readouts.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --readouts experiments/self_repair/mechanistic/config/readouts.json \
  --manifest "$MECH_ARTIFACT_ROOT/manifests/mechanistic_trials.jsonl" \
  --role internal_validation \
  --folds 4,5 \
  --output "$MECH_ARTIFACT_ROOT/internal_validation/baseline_readout.jsonl"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/fit_probes.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --role internal_validation \
  --frozen-selection "$MECH_ARTIFACT_ROOT/mechanistic_frozen_selection.json" \
  --output-root "$MECH_ARTIFACT_ROOT/internal_validation/probes"
```

Internal causal gate:

- frozen clean sign accuracy가 다시 ≥80%이고, baseline gap
  `G = M(clean-current) - M(repair)`의 scenario-cluster CI가 0보다 큼
- primary value-specific `DID`의 Holm-adjusted two-sided p < .05, 예상 부호가 양수이며 frozen
  adjusted/simultaneous CI lower bound가 frozen positive SESOI보다 큼
- `delta_rescue` CI lower bound가 positive SESOI보다 크고, `delta_stale` CI upper bound가
  negative SESOI보다 작음
- frozen endogenous test `E_transfer < 0` 또는 `E_erase > 0`가 matched control 대비 재현됨
- 두 방향에서 effect sign이 일치
- specificity contrast `C_specific = DID_primary - max_c(abs(DID_control_c))`의 CI가 0보다 큼.
  여기서 control set은 neutral/same-value/wrong-time/shuffled로 미리 고정
- bootstrap CI가 pointwise인지 simultaneous인지 명시. Familywise interval 주장을 하면 max-T
  simultaneous CI를 사용
- all-scenario raw effects와 cluster CI가 primary이며 exhaustive sign-flip/paired-t/wild-cluster는
  가정을 명시한 sensitivity analysis
- 사전 지정 cell이 모두 완료됨. OOM/NaN은 같은 identity로만 재시도하고 미해결이면 해당
  hypothesis를 `unevaluable`로 두며 available-case 분석이나 imputation을 하지 않음

통과하지 않아도 결과를 버리지 않고 null/unstable finding으로 보고한다.

---

## Step 17. 새 다중 도시 set에서 formal confirmation한다 `[TARGET]`

Step 2의 사람 검수를 마친 최소 4-city fully-crossed set에서 **새 site를 탐색하지 않고** frozen
circuit을 적용한다. 별도 immutable role manifest가 discovery와 formal confirmation의 whole
scenario template 및 whole ordered pair를 격리해야 한다.

```bash
# Reviewed multivalue source를 portable mechanistic manifest로 bind하고 먼저 fail-closed 검증한다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/build_mech_manifest.py \
  --prepared-manifest /workspace/multivalue-controls/prepared_stimuli.jsonl \
  --role-manifest /workspace/multivalue-controls/role_manifest.jsonl \
  --data-status reviewed_multivalue \
  --output "$MECH_ARTIFACT_ROOT/manifests/multivalue_trials.jsonl"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/validate_multivalue_controls.py \
  --input-root /workspace/multivalue-controls \
  --mechanistic-manifest "$MECH_ARTIFACT_ROOT/manifests/multivalue_trials.jsonl" \
  --require-independent-alignment \
  --require-double-listen-review

sha256sum "$MECH_ARTIFACT_ROOT/manifests/multivalue_trials.jsonl"
sha256sum /workspace/multivalue-controls/role_manifest.jsonl

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/encode_user_audio.py \
  --manifest "$MECH_ARTIFACT_ROOT/manifests/multivalue_trials.jsonl" \
  --output-root "$MECH_ARTIFACT_ROOT/formal_confirmation/encoded_user" \
  --output-manifest \
    "$MECH_ARTIFACT_ROOT/formal_confirmation/encoded_user_manifest.jsonl" \
  --model-revision 2bfc9ae6e89079a5cc7ed2a68436010d91a3d289 \
  --resume

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/build_anchor_map.py \
  --manifest "$MECH_ARTIFACT_ROOT/manifests/multivalue_trials.jsonl" \
  --prepared-manifest /workspace/multivalue-controls/prepared_stimuli.jsonl \
  --output "$MECH_ARTIFACT_ROOT/formal_confirmation/anchor_map.jsonl" \
  --frame-trace-output "$MECH_ARTIFACT_ROOT/formal_confirmation/frame_trace.jsonl"

# Frozen site에서만 K-class probe를 calibration role에 fit하고 두 번째 immutable hash를 만든다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/fit_probes.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_ARTIFACT_ROOT/manifests/multivalue_trials.jsonl" \
  --role-manifest /workspace/multivalue-controls/role_manifest.jsonl \
  --role multivalue_calibration \
  --site-selection "$MECH_ARTIFACT_ROOT/mechanistic_frozen_selection.json" \
  --freeze-output \
    "$MECH_ARTIFACT_ROOT/formal_confirmation/multivalue_probe_frozen.json" \
  --output-root "$MECH_ARTIFACT_ROOT/formal_confirmation/probe_calibration"

sha256sum "$MECH_ARTIFACT_ROOT/formal_confirmation/multivalue_probe_frozen.json"

# Formal rows에서는 frozen readout/probe만 적용하며 새 layer/anchor/classifier를 선택하지 않는다.
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/score_readouts.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --readouts experiments/self_repair/mechanistic/config/readouts.json \
  --manifest "$MECH_ARTIFACT_ROOT/manifests/multivalue_trials.jsonl" \
  --role-manifest /workspace/multivalue-controls/role_manifest.jsonl \
  --role formal_confirmation \
  --output "$MECH_ARTIFACT_ROOT/formal_confirmation/baseline_readout.jsonl"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/fit_probes.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --manifest "$MECH_ARTIFACT_ROOT/manifests/multivalue_trials.jsonl" \
  --role-manifest /workspace/multivalue-controls/role_manifest.jsonl \
  --role formal_confirmation \
  --frozen-probe \
    "$MECH_ARTIFACT_ROOT/formal_confirmation/multivalue_probe_frozen.json" \
  --output-root "$MECH_ARTIFACT_ROOT/formal_confirmation/probes"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/run_confirmatory_patches.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --selection "$MECH_ARTIFACT_ROOT/mechanistic_frozen_selection.json" \
  --manifest "$MECH_ARTIFACT_ROOT/manifests/multivalue_trials.jsonl" \
  --role-manifest /workspace/multivalue-controls/role_manifest.jsonl \
  --encoded-manifest \
    "$MECH_ARTIFACT_ROOT/formal_confirmation/encoded_user_manifest.jsonl" \
  --anchors "$MECH_ARTIFACT_ROOT/formal_confirmation/anchor_map.jsonl" \
  --baseline-readout "$MECH_ARTIFACT_ROOT/formal_confirmation/baseline_readout.jsonl" \
  --role formal_confirmation \
  --output-root "$MECH_ARTIFACT_ROOT/formal_confirmation" \
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
- `old_value_identity`, `old_was_mentioned`, `current_destination`, `is_retracted`를 독립적으로
  decode/patch 가능
- Step 16과 같은 baseline-gap, DID/component/endogenous/specificity/missing-cell/multiplicity gate 통과

실패하면 일반 stale-binding mechanism이 아니라 Boston–Seattle 또는 특정 lexical-value circuit으로
결론을 낮춘다.

---

## Step 18. 최종 후보만 정상 full-duplex에서 재검증한다 `[TARGET]`

Open-loop는 원인 진단의 주 분석이고, full-duplex는 ecological bridge다. 다음 네 조건만 top
candidate에 적용한다.

- 같은 repair snapshot에서 시작한 unpatched/no-op branch
- 같은 repair snapshot의 frozen `E_erase` 또는 candidate ablation branch — primary
- clean-current donor intervention — 조건부 exploratory
- wrong-target donor intervention — 조건부 exploratory

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
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/run_full_duplex_validation.py \
  --config experiments/self_repair/mechanistic/config/mechanistic.json \
  --selection "$MECH_ARTIFACT_ROOT/mechanistic_frozen_selection.json" \
  --primary-intervention within_repair_erasure \
  --donor-arms conditional_on_feedback_divergence \
  --seeds 17,29,42,101,2026 \
  --output-root "$MECH_ARTIFACT_ROOT/full_duplex" \
  --resume
```

두 intervention-blind annotator가 `final_target_correct`, `stale_state_error`, D1–D3 binding을
독립 판정하고 불일치는 제3자가 조정한다. agreement도 보고한다. Primary behavioral estimand는
scenario 동일 가중의 paired `patched - unpatched` final-target accuracy difference, key secondary는
stale-state-error difference다. Five seeds는 먼저 rendition별 successes/5로 집계하고,
scenario-cluster CI, frozen SESOI와 multiplicity family를 적용한다.

Held-out run에서 clean ≥80%, immediate-repair ≥70%, scorable primary windows ≥90%를 다시
요구한다. assistant greeting 억제 조건과 기존 자연 streaming 조건을 분리 보고한다. Logit
margin만 좋아지고 audible response가 좋아지지 않으면 “text-logit surrogate에 causal
contribution”까지만 말한다.

기존 behavioral runner로 남은 2,400개를 재개하는 것은 이 단계의 turn-taking smoke와 capability
gate가 통과한 뒤에만 결정한다.

---

## Step 19. 분석·그림·보고서를 만든다 `[TARGET]`

```bash
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/analyze_mechanistic_results.py \
  --run-root "$MECH_ARTIFACT_ROOT" \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260826

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/render_mechanistic_report.py \
  --run-root "$MECH_ARTIFACT_ROOT"

.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/verify_mechanistic_run.py \
  --run-root "$MECH_ARTIFACT_ROOT"
```

최소 산출물:

```text
<run-root>/
  run_identity.json
  environment.json
  input_hash_manifest.jsonl
  encoded_user_manifest.jsonl
  anchor_map.jsonl
  frame_trace.jsonl
  open_loop_validation.json
  baseline_readout.jsonl
  capture_manifest.jsonl
  failures.jsonl
  resume_summary.json
  artifact_sha256.json
  discovery/
    residual_patch_results.jsonl
    component_patch_results.jsonl
    kv_patch_results.jsonl
    path_patch_results.jsonl
    probe_metrics.json
  internal_validation/
    patch_results.jsonl
    probe_metrics.json
    baseline_readout.jsonl
    metrics.json
  formal_confirmation/
    patch_results.jsonl
    baseline_readout.jsonl
    multivalue_probe_frozen.json
    probe_metrics.json
    metrics.json
  full_duplex/
    validation.jsonl
  reports/
    MECHANISTIC_RESULTS.md
    mechanistic_discovery_summary.json
    mechanistic_frozen_selection.json
    mechanistic_confirmation_metrics.json
    tables/all_scenario_effects.csv
    tables/multiplicity_registry.csv
    figures/01_baseline_margin_by_condition.svg
    figures/02_probe_discovery_layer_time_heatmap.svg
    figures/03_residual_patch_heatmap_DISCOVERY.svg
    figures/04_preregistered_confirmation_points_forest.svg
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

1. `run_identity.json`: code/model/data가 의도한 실행인지 확인
2. `open_loop_validation.json`: replay/self-patch gate 확인
3. baseline figure: clean capability와 repair gap 확인
4. discovery heatmap: 후보를 찾은 위치를 보되 확증 p-value로 해석하지 않음
5. internal/formal confirmation forest: preregistered effect, CI, Holm p, SESOI 확인
6. controls figure: no-op/wrong-time/shuffled/neutral effect 확인
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
cd "$MECH_ARTIFACT_ROOT/reports"
sed -n '1,240p' MECHANISTIC_RESULTS.md
find figures -maxdepth 1 -type f -name '*.svg' -print | sort
```

로컬 브라우저로 볼 때는 RunPod terminal에서 read-only HTTP server를 열고, RunPod가 제공한
TCP/HTTP port mapping을 사용한다.

```bash
cd "$MECH_ARTIFACT_ROOT/reports"
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
먼저 `du -sh "$MECH_ARTIFACT_ROOT"`와 `df -h /workspace`로 여유 공간을 확인한다. 부족하면
scenario shard별로 압축해 즉시 object storage/local로 전송하거나, tar stream을 SSH로 직접
보내고 전체 archive를 같은 volume에 만들지 않는다.

```bash
cd "$MECH_REPO_ROOT"
.venv/bin/python \
  experiments/self_repair/mechanistic/scripts/package_mechanistic_results.py \
  --run-root "$MECH_ARTIFACT_ROOT" \
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

Fork에 결과용 branch를 올리는 예시는 다음과 같다. `git add`에는 검증된 small-artifact
allowlist만 명시하고 directory 전체를 무심코 stage하지 않는다.

```bash
cd "$MECH_REPO_ROOT"
git switch -c codex/mechanistic-stale-binding-results

git add \
  experiments/self_repair/mechanistic/config \
  experiments/self_repair/mechanistic/reports/MECHANISTIC_RESULTS.md \
  experiments/self_repair/mechanistic/reports/figures \
  experiments/self_repair/mechanistic/reports/tables \
  experiments/self_repair/mechanistic/reports/run_identity.public.json \
  experiments/self_repair/mechanistic/reports/artifact_sha256.public.json

git diff --cached --name-only
git diff --cached --check
git status --short
git commit -m "Report mechanistic stale-binding experiment"
git remote -v
git push -u origin codex/mechanistic-stale-binding-results
```

위 report 경로는 packaging tool이 private URI/absolute path를 제거해 생성해야 한다. 출력이 아직
`$MECH_ARTIFACT_ROOT/reports`에만 있다면 검증된 small files를 tracked report directory로 복사하는
별도 export step을 packaging tool에 구현한 뒤 stage한다.

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
