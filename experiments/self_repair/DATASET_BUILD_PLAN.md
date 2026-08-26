# 자기수정 matched-bundle 데이터셋 구축 계획

- 문서 상태: 실행 중 — Kokoro raw 600·10-speaker MFA·자동 QC 완료, human review gate 대기
- 작성일: 2026-08-26
- 적용 위치: `experiments/self_repair/`
- 핵심 산출물: 의미 설계도 30개, 생성 대본 300개, accepted audio 600개
- 선택 산출물: pause-based latency extension audio 120개

버전 용어는 기존 E1–E9 파일럿을 legacy v1, 이 문서에서 새로 만드는 matched-bundle
release를 dataset/schema v2.0.0으로 통일한다. 새 경로, `VERSION`, schema와 release tag는
모두 v2를 가리킨다.

## 실행 체크포인트 — 2026-08-26

| 단계 | 상태 | 검증 증거 |
|---|---|---|
| 설계·스키마·분석 계약 | 완료 | v2 config 3개, schema 6개, 분석/주석 계약 |
| 의미 설계도 | 완료 | 30/30 독립 agent review 승인, 구조·schema error 0 |
| 대본·답안 키 | 완료 | 300 scripts, 60 answer keys, byte-level 재생성 일치 |
| 화자 배정 | 완료 | 120 matched bundles, 600 rendition targets, 10 voices 균형 |
| 비공개 Edge calibration | 완료 | 180/180 합성·provider boundary mapping, 158 QC pass, 22 clipping 제외 |
| Kokoro voice calibration | 기술 완료 | 10/10 실제 로컬 합성·24k PCM16·token mapping·무클리핑 QC 통과, human double-listen 대기 |
| production preflight | 완료 | 600-target join·Kokoro model/config/voice hash·authority·local 12GiB/remote 40GiB·MFA gate 자동화 |
| storage architecture | 완료 | local TTS/QC/MFA + RunPod Moshi, 대용량 response remote 유지 |
| production TTS | raw/QC 완료 | 600/600 local Kokoro 합성·canonical 변환·자동 QC 통과, 5.756시간·raw+canonical 1.853GiB |
| 독립 정렬·선정 | 정렬 완료·사람 승인 대기 | local MFA 2.2.4 10-speaker TextGrid/import 600/600, OOV 0, 재자동 QC 600/600; calibrated confidence가 없어 human review 전 accepted 0/600 유지 |
| accepted 600·Moshi 3,000·annotation | 대기 | production audio 이후 실행 |
| human/pause extension | 대기 | core 결과와 별도 consent/예산 결정 이후 실행 |

비공개 calibration에서는 fast/slow voice 사이의 global timing overlap이 없었다. 따라서
원문의 4,000 ms 설명값을 production target으로 쓰지 않고 화자별 자연 발화 target을
승인된 provider에서 다시 측정한다. 상세 결과는
`dataset_v2/reports/edge_private_timing_calibration.json`에 기록한다.

## 0. 한눈에 보는 계획

이 데이터셋의 핵심은 600개 문장을 각각 쓰는 것이 아니다. 사람이 검토한 30개
의미 설계도만 만들고, 코드가 각 설계도를 양방향·5조건으로 재배열하여 300개
대본을 생성한다. 각 대본은 배정된 화자 2명이 읽으므로 최종 음성은 600개가 된다.

```text
30 scenarios
× 2 directions (A→B, B→A)
× 5 conditions
= 300 scripts

300 scripts
× 2 speakers
= 600 accepted audio files
```

분석 단위는 음성 하나가 아니라 `scenario × direction × speaker`로 묶인 5조건
`matched_audio_bundle`이다. 같은 matched audio bundle의 다섯 발화는 다음을 공유해야
한다.

- 최종 의도와 구조화된 `gold_state`
- D1·D2·D3 및 N1·N2·N3 전체 의미 정보
- repair 조건에서 동일한 repair cue 문구
- 화자 또는 TTS voice
- 녹음·합성 환경과 오디오 처리 방식

주요 독립변수는 repair 직전에 철회될 old value와 이미 연결된 dependent unit의
수 `0`, `1`, `3`이다. 세 delayed 조건에서는 repair 전과 후의 의미 단위 수를 각각
3개로 고정하고 실제 시간을 별도로 측정한다.

### 수량과 ID 용어

이 문서에서는 `bundle`이라는 단어를 단독으로 쓰지 않고 다음 네 수준을 구분한다.

| 단위 | ID | 정의 | 수량 |
|---|---|---|---:|
| text bundle | `text_bundle_id` | `scenario × direction` | 60 |
| script | `script_id` | `text_bundle × condition` | 300 |
| matched audio bundle | `matched_audio_bundle_id` | `text_bundle × source_track × speaker`의 5조건 묶음 | 트랙당 120 |
| rendition target | `rendition_target_id` | `script × source_track × speaker` | 트랙당 600 |

Azure처럼 비결정적 provider를 사용할 때의 원 설계는 후보 3개, 실패 시 최대 5개였다.
현재 Kokoro source track은 고정 설정에서 결정적이므로 rendition target당 후보 1개,
candidate audio 총 600개로 override한다. 같은 요청을 반복해 동일 waveform을 독립 후보처럼
세지 않는다.
`candidate_id`는 rendition target과 attempt 번호로 만들고, 최종
`accepted_audio_id`는 각 rendition target에서 선택된 파일 하나만 가리킨다. TTS와
human track은 같은 `script_id`를 공유해도 `source_track_id`가 달라 ID가 충돌하지 않는다.

### 핵심 연구 질문

1. 실제 지연 시간이 비슷할 때 stale dependency가 0개, 1개, 3개로 늘어나면 최종
   수정값 선택 성능이 하락하는가?
2. clean과 immediate 조건을 통과한 모델이 delayed 조건에서만 실패하는가?
3. dependency 수와 spoken semantic load를 고정한 pause-based 조작에서 elapsed delay가
   길어지면 성능이 하락하는가? 이 질문은 선택 extension에서 별도로 검증한다.

### 단계별 최종 흐름

```text
설계 동결 → 스키마/도구 골격 → 설계도 30개 → 대본 300개 자동 생성
→ 정적·의미 QC → 화자 균형 배정 → TTS smoke → 전체 합성/녹음
→ forced alignment → 후보 선택 → 오디오/메타데이터 QC
→ baseline/통계 → 사람 음성 검증 → 버전 고정·배포
```

## 1. 범위와 고정 원칙

### 1.1 v2 권장 범위

현재 저장소의 영어 Boston–Seattle 대조 실험과 이어지는 첫 버전은 다음처럼
통제하는 것을 권장한다.

- 언어: 영어
- root slot: `destination`
- value pair: `Boston ↔ Seattle`
- 30개 scenario: 같은 root 구조를 갖되 문맥과 D/N 의미 설계를 달리한 30개
- repair cue: `Sorry, I mean {new}, not {old}.`
- 핵심 audio track: TTS controlled-reading 600개
- 후속 검증: 동일 300개 대본의 human controlled-reading

이 기본안은 범용 학습 corpus가 아니라 pretrained Moshi의 내부 타당성을 우선한
`controlled evaluation stimulus set`이다. 모든 optional fold가 Boston/Seattle을 공유하므로
새 value pair나 root slot에 대한 lexical/semantic 일반화를 측정할 수 없다.

여러 도메인, root slot, value pair를 한 번에 섞으면 dependency 효과에 어휘 친숙도,
도메인 난이도, 양방향 자연성 차이가 추가될 수 있다. 따라서 다양한 slot/value의
일반화는 v2 controlled 결과를 확인한 뒤 별도 확장 세트로 추가한다. 실제 제작 전에 Step 0에서
이 기본안을 확정하거나 변경 내역을 결정 기록에 남긴다. 목적이 학습 corpus라면 이
기본안을 사용하지 않고 여러 value pair/root slot, value-pair holdout, root-slot holdout과
더 큰 표본 수를 새로 설계한다.

### 1.2 반드시 지킬 원칙

- 300개 대본을 LLM으로 각각 자유 생성하지 않는다.
- 사람은 30개 설계도의 의미 구조와 문장 자연성을 검토한다.
- 코드는 방향과 조건에 따라 검증된 단위의 순서만 재배열한다.
- 같은 text bundle과 matched audio bundle의 최종 의미는 완전히 같아야 한다.
- delayed 조건은 `Root + pre 3 units + Repair + post 3 units + Closing` 구조를 지킨다.
- 조건 label이 아니라 forced alignment로 측정한 실제 시간을 분석에 사용한다.
- TTS는 각 발화를 통문장으로 합성한다. 문장 조각을 이어 붙이지 않는다.
- 강한 time-stretch로 시간을 맞추지 않는다. 필요하면 문장 경계 pause만 미세 조절한다.
- candidate audio와 최종 audio를 구분하며 제외 사유를 모두 남긴다.
- 기존 E1–E9 파일과 결과는 변경하지 않고 v2 경로에 새 데이터를 만든다.

### 1.3 데이터 트랙과 수량의 해석

600개는 한 audio track의 accepted 수량이다. TTS 600개와 human 600개를 모두 만들면
총 1,200개이며, 두 트랙을 같은 데이터처럼 섞지 않고 canonical `source_track_id`로
구분한다. `source_type`은 `tts`/`human`을 나타내는 파생 필드로만 사용한다.

| 트랙 | 역할 | 권장 규모 | 공개 결론의 범위 |
|---|---|---:|---|
| TTS controlled | 파이프라인, metric, baseline, latency bin 검증 | 600 | 합성 음성에서의 통제 실험 |
| Human controlled-reading | 최종 validation/test | 600 | 통제된 사람 발화 |
| Human elicited-natural | 자연스러운 repair prosody 확장 | 별도 결정 | 통제가 약한 자연발화 확장 |
| Pause-based latency extension | 지연 효과 분리 | 120 | `dependency_count=0`에서의 pause-latency curve |

## 2. 실험 단위와 조건 정의

### 2.1 의미 설계도 한 개의 필수 요소

각 scenario는 다음 요소를 갖는다.

- `root_slot`: 수정되는 중심 정보의 slot
- `value_a`, `value_b`: 양방향으로 교환할 두 값
- `root_template`: old 또는 new value를 소개하는 문장
- `D1`, `D2`, `D3`: root 값에 의존하는 세 의미 단위
- `N1`, `N2`, `N3`: root 값과 무관하고 수정 후에도 유지되는 세 의미 단위
- `repair_template`: 모든 조건에서 공유하는 수정 표현
- `closing_prompt`: 모든 조건 끝에 붙는 root-invariant terminal response cue
- `gold_state_template`: 최종 root와 D/N 상태를 구조화하는 규칙
- `one_dependency_unit`: 해당 scenario의 delayed-one에서 repair 전에 둘 D 단위
- `one_dependency_pre_position`: delayed-one의 pre-repair 세 위치 중 D가 놓일 위치

Dependent unit은 root 값이 바뀌면 연결 대상도 바뀌어야 한다. Neutral unit은 root가
바뀌어도 의미와 참값이 그대로 유지되어야 한다.

```text
destination                  user
├── activity.location        ├── preference
├── food.location            ├── dietary_restriction
└── accommodation.location   └── budget
```

Neutral unit에는 도시명, 도시 별칭, 지역명, demonym, `there`, `that city`, `local`,
`while I am there`처럼 root를 다시 가리킬 수 있는 표현을 넣지 않는다. 자동 금칙어
검사와 별개로 두 명의 검토자가 의미 독립성을 판정한다.

각 unit에는 relation 이름만 쓰지 않고 다음 typed annotation을 둔다.

- `state_patch`: 해당 unit이 최종 상태에 추가하는 key/value 또는 root-bound placeholder
- `binding`: `root_dependent` 또는 `root_invariant`
- `balance_pair_id`: 길이·위치 균형을 위해 대응시키는 D/N pair
- `speech_act`: statement, request, question 등 담화행위
- `boundary_type`: `nonterminal` 또는 `terminal`

`gold_state`는 텍스트에서 다시 추측하지 않고 root 값에 각 `state_patch`를 적용해
계산한다. 두 방향과 다섯 조건의 patch multiset이 같은지 코드가 구조 비교한다.

### 2.2 Full-duplex 턴테이킹 통제

Moshi는 사용자 발화가 끝나기 전에도 응답할 수 있다. D가 질문이고 N이 진술이면
`delayed_three`는 repair 전에 답변을 유도하는 질문이 세 개, `delayed_neutral`은 0개가
되어 dependency 수와 조기 턴 개시를 구분할 수 없다.

Primary set에서는 D와 N을 가능한 한 같은 담화행위의 비종결 선언형 clause 또는
목록형 요청으로 쓴다. 특히 repair 전 unit에는 독립적인 질문표와 긴 종결 pause를 두지
않는다. 예를 들어 D를 `I will need activity recommendations for the trip`처럼 표현하고
N도 같은 억양·접속 구조로 이어 간다. TTS의 boundary/pause policy는 조건별로 바꾸지
않는다. 여섯 unit을 모두 nonterminal로 만들면 발화 종료가 불분명해지므로 모든 조건의
맨 끝에 동일한 root-invariant closing prompt `Could you help me plan all of that?`을 한 번
붙인다. Closing은 D/N 의미 단위 수에는 포함하지 않는다. 두 명의 검토자가 다음을
판정한다.

- D/N의 `speech_act`와 문장 경계 강도가 대응되는가?
- repair 전에 모델이 답을 시작하도록 유도하는 standalone question이 있는가?
- condition에 따라 pause나 종결 억양 수가 달라지는가?

모델 평가 시 `assistant_started_before_repair`, `assistant_speech_onset_ms`,
`pre_repair_assistant_audio_ms`를 저장한다. 이는 condition 이후에 생길 수 있는
mediator이므로 primary 모델의 단순 공변량으로 넣지 않는다. 조건별 조기 응답률 차이가
사전 tolerance를 넘으면 dependency-only 인과 해석 gate를 실패 처리하고, 별도
진단·mediation 또는 sensitivity 분석으로 보고한다.

### 2.3 다섯 조건의 생성 규칙

| 조건 | 생성 순서 | repair 전 dependent 수 | 용도 |
|---|---|---:|---|
| `clean_final` | `new root + N1 N2 N3 + D1 D2 D3 + C` | 해당 없음 | clean teacher/gate |
| `immediate_repair` | `old root + repair + N1 N2 N3 + D1 D2 D3 + C` | 0 | 즉각 수정 처리 확인 |
| `delayed_neutral` | `old root + N1 N2 N3 + repair + D1 D2 D3 + C` | 0 | 긴 시간·무 stale dependency |
| `delayed_one_dependency` | `old root + permute(Di, Nj, Nk) + repair + permute(Dj, Dk, Ni) + C` | 1 | stale dependency 1개 |
| `delayed_three_dependencies` | `old root + D1 D2 D3 + repair + N1 N2 N3 + C` | 3 | stale dependency 3개 |

`C`는 모든 조건에 동일한 terminal `closing_prompt`이며 dependency count에 포함하지 않는다.

Delayed-one의 `Di`는 고정하지 않고 다음처럼 counterbalance한다. 여기서 `Ni`는 선택된
`Di`와 같은 `balance_pair_id`를 가진 neutral unit이며 repair 뒤로 보내고, 나머지 두
neutral unit은 repair 전에 둔다.

- scenario 01–10: D1을 repair 전에 배치
- scenario 11–20: D2를 repair 전에 배치
- scenario 21–30: D3를 repair 전에 배치

`Di`의 pre-repair 위치도 1/2/3에 각각 10개를 배정한다. D identity × position의 3×3
cell은 각 3개 또는 4개가 되도록 고정 rotation table을 만든다. `one_dependency_unit`,
`one_dependency_pre_position`, 전체 pre/post order를 blueprint에 저장하고 validator가
10/10/10 및 cell 3–4개 조건을 검사한다. 양방향은 같은 rotation을 사용하며 분석에는
실제 unit offset부터 repair cue까지의 `stale_dependency_age_ms`도 저장한다.

### 2.4 primary contrast와 진단 조건

Primary contrast는 실제 latency가 통제된 세 delayed 조건의 비교다.

```text
delayed_neutral (0) ↔ delayed_one (1) ↔ delayed_three (3)
```

`clean_final`과 `immediate_repair`는 모델의 기본 이해 및 가장 짧은 수정 능력을
확인하는 gate다. 이 둘이 실패하면 delayed 차이를 stale dependency 효과로 해석하지
않는다.

## 3. 시간 정의와 통제 기준

### 3.1 저장할 정렬 이벤트

- `old_value_onset_ms`: 최초 old value 시작
- `old_value_offset_ms`: 최초 old value 종료
- `repair_cue_onset_ms`: `Sorry`, `No` 등 repair cue 시작
- `new_value_onset_ms`: 최종 new value 시작(clean root 또는 repair cue 안)
- `new_value_offset_ms`: 최종 new value 종료
- `repeated_old_onset_ms`: `not {old}` 안의 반복 old value 시작
- `repeated_old_offset_ms`: 반복 old value 종료
- `repair_cue_offset_ms`: 전체 repair cue 종료
- `closing_prompt_onset_ms`: 공통 closing prompt 시작
- `closing_prompt_offset_ms`: 공통 closing prompt 종료
- `utterance_end_ms`: 전체 사용자 발화 종료

계산식은 다음으로 통일한다.

```text
actual_latency_ms = repair_cue_onset_ms - old_value_offset_ms
post_final_value_duration_ms = utterance_end_ms - new_value_offset_ms
post_repair_duration_ms = post_final_value_duration_ms  # repair 조건의 호환 alias
post_cue_duration_ms = utterance_end_ms - repair_cue_offset_ms
```

Clean에서는 initial old value, repair cue와 repeated old 필드만 `null`이고, clean root의
`new_value_onset_ms`와 `new_value_offset_ms`는 반드시 저장한다. Immediate와 delayed에는
모든 이벤트를 저장한다. onset과 offset을 혼용하지 않도록 계산 함수와 JSON Schema에
공식을 명시한다.
Legacy E1–E9 timestamp를 v2 event로 직접 이관하지 않는다. 기존 영어 neural TTS의
`repair_onset_ms`는 도시명이 아니라 `I` 또는 `could` 같은 clause 시작일 수 있고,
`repair_marker_onset_ms`도 새 cue 정의와 완전히 같다고 보장할 수 없다. v2 audio는 모두
독립 정렬해 cue/new/repeated-old onset·offset을 새로 계산한다.
repair template이 old value를 반복하지 않으면 `repeated_old_*`는 `null`이지만
모든 repair 조건에서 `repair_cue_offset_ms`는 항상 저장한다. Clean에서는 cue 필드가
`null`이다. 모든 응답 window는 cue 시작과 cue 완료를 구분한다.

### 3.2 자연 발화 기반 timing calibration

4,000 ms는 원문의 설명용 예시이지 production 기준이 아니다. 세 개의 완전한 의미
단위를 4초 안에 읽으면 부자연스러운 속도가 필요할 수 있으므로 설계도 동결 전에
empirical calibration을 먼저 수행한다.

1. D/N 초안에서 예상 duration이 짧은·중간·긴 scenario를 고른다.
2. voice pool 전체를 짧은 calibration script로 읽혀 fastest/median/slowest voice를 찾는다.
3. 대표 voice로 `N N N`, `D N N`, `D D D` 및 세 post-repair 조합을 자연 설정에서
   합성한다.
4. forced alignment로 각 조합의 duration 분포와 공통 overlap 구간을 구한다.
5. 공통 overlap 안에서 `target_latency_ms`, `target_post_duration_ms`, tolerance를 정하고
   Step 0 명세에 동결한다.

전역 절대시간보다 같은 matched audio bundle 안에서 세 delayed 조건을 맞추는 것이
우선이다. TTS는 voice별, 사람 음성은 speaker별 자연 속도 calibration으로 target을 둘
수 있다. 원문의 ±200–300 ms는 tolerance 시작점으로만 사용하며 feasibility 결과와 함께
사전 확정한다. production 이후 조건별 threshold를 바꾸지 않는다. 실제 분석에는
`actual_latency_ms`, `post_repair_duration_ms`, 각 stale unit의 age를 연속 변수로 넣는다.

## 4. 저장소 구조와 데이터 계약

기존 self-repair 파일럿을 보존하기 위해 새 데이터는 `dataset_v2/`에 격리한다.

```text
experiments/self_repair/
├── DATASET_BUILD_PLAN.md
├── dataset_v2/
│   ├── README.md
│   ├── VERSION
│   ├── config/dataset.yaml
│   ├── schemas/
│   │   ├── blueprint.schema.json
│   │   ├── script.schema.json
│   │   ├── audio.schema.json
│   │   ├── eval_trial.schema.json
│   │   └── annotation.schema.json
│   ├── blueprints/scenarios.jsonl
│   ├── answer_keys/scenario_answers.jsonl
│   ├── generated/scripts.jsonl
│   ├── assignments/speaker_bundles.csv
│   ├── manifests/audio_candidates.jsonl
│   ├── manifests/audio_accepted.jsonl
│   ├── manifests/stimuli_prepared.jsonl
│   ├── alignments/
│   ├── folds/analysis_folds.csv
│   ├── annotations/
│   └── reports/
└── scripts/dataset_v2/
    ├── generate_scripts.py
    ├── validate_blueprints.py
    ├── validate_scripts.py
    ├── assign_speakers.py
    ├── align_audio.py
    ├── select_candidates.py
    ├── validate_audio.py
    └── build_release.py
```

Git에는 스키마, 설계도, 생성 코드, 작은 manifest, alignment 결과, 통계 보고서와
SHA-256을 저장한다. 대량 candidate/final audio는 일반 Git commit에 넣지 않는다.
Step 0에서 Git LFS, GitHub Release, object storage 중 배포 방식을 결정하고 manifest의
`audio_uri`와 hash로 연결한다.

### 4.1 Blueprint JSONL 최소 계약

```json
{
  "schema_version": "2.0.0",
  "scenario_id": "travel_001",
  "language": "en",
  "domain": "travel",
  "root_slot": "destination",
  "value_a": "Boston",
  "value_b": "Seattle",
  "root_template": "I'm planning a trip to {value}",
  "dependent_units": [
    {
      "unit_id": "D1",
      "text": "I will need activity recommendations for the trip",
      "relation": "activity.location",
      "binding": "root_dependent",
      "state_patch": {"activity.location": "{root_value}"},
      "balance_pair_id": "P1",
      "speech_act": "statement",
      "boundary_type": "nonterminal"
    },
    {
      "unit_id": "D2",
      "text": "I will also need food suggestions for the trip",
      "relation": "food.location",
      "binding": "root_dependent",
      "state_patch": {"food.location": "{root_value}"},
      "balance_pair_id": "P2",
      "speech_act": "statement",
      "boundary_type": "nonterminal"
    },
    {
      "unit_id": "D3",
      "text": "I will need lodging recommendations for the trip",
      "relation": "accommodation.location",
      "binding": "root_dependent",
      "state_patch": {"accommodation.location": "{root_value}"},
      "balance_pair_id": "P3",
      "speech_act": "statement",
      "boundary_type": "nonterminal"
    }
  ],
  "neutral_units": [
    {
      "unit_id": "N1",
      "text": "I generally prefer museums to nightlife",
      "relation": "user.preference",
      "binding": "root_invariant",
      "state_patch": {"user.preference": "museums"},
      "balance_pair_id": "P1",
      "speech_act": "statement",
      "boundary_type": "nonterminal"
    },
    {
      "unit_id": "N2",
      "text": "I cannot eat seafood or shellfish",
      "relation": "user.dietary_restriction",
      "binding": "root_invariant",
      "state_patch": {"user.dietary_restriction": "no_seafood_or_shellfish"},
      "balance_pair_id": "P2",
      "speech_act": "statement",
      "boundary_type": "nonterminal"
    },
    {
      "unit_id": "N3",
      "text": "My nightly hotel budget is about two hundred dollars",
      "relation": "user.hotel_budget",
      "binding": "root_invariant",
      "state_patch": {"user.hotel_budget_usd": 200},
      "balance_pair_id": "P3",
      "speech_act": "statement",
      "boundary_type": "nonterminal"
    }
  ],
  "repair_template": "Sorry, I mean {new}, not {old}",
  "closing_prompt": "Could you help me plan all of that?",
  "one_dependency_unit": "D1",
  "one_dependency_pre_position": 1,
  "rotation_id": "R1",
  "reviews": [
    {"reviewer_id": "reviewer_01", "decision": "approved"},
    {"reviewer_id": "reviewer_02", "decision": "approved"}
  ],
  "review_status": "approved"
}
```

실제 schema에는 unit별 `state_patch`, `balance_pair_id`, `speech_act`,
`boundary_type`, `gold_state_template`, 예상 단어 수·duration, 독립 검토 2건의
`reviews[]`, derived `review_status`, 검토 일시와 라이선스/출처를 필수로 포함한다.
예시 문구 자체도 Step 2에서 길이, 담화행위,
양방향 자연성을 다시 검토하며 승인된 production 설계도로 간주하지 않는다.

### 4.2 Generated script 계약

각 행은 적어도 다음을 포함한다.

- `text_bundle_id`, `script_id`, `scenario_id`, `direction_id`, `condition`
- `old_value`, `new_value`, `root_slot`
- 순서가 보존된 `segments` 배열과 각 segment의 역할
- `pre_repair_units`, `post_repair_units`, `dependency_count`
- `transcript`, `normalized_transcript`, `repair_cue`
- 구조화된 `pre_repair_state`, `repair_rebindings`, `gold_state`
- `blueprint_hash`, `generator_version`, `generation_seed`

`blueprint_hash`는 review 기록을 포함한 승인 blueprint 전체를 key-sorted, UTF-8,
whitespace 없는 canonical JSON으로 직렬화한 SHA-256이다. review나 의미 단위가 바뀌면
hash도 바뀌며 scripts 전체를 재생성한다.

old value 검증은 단순히 “repair 전에서만 등장”으로 구현하지 않는다. 현재 repair cue가
`not {old}`를 포함하므로 old value는 `initial_old_root`와 `repair_cue` segment에서만
허용하고 다른 segment에서는 금지한다. Clean은 old value가 없어야 하고 new value가
root에 있어야 한다. Repair 조건은 new value가 repair cue에 정확히 한 번 있어야 한다.

### 4.3 Audio manifest 최소 계약

- ID: `accepted_audio_id`, `selected_candidate_id`, `candidate_id`, `rendition_target_id`, `script_id`,
  `text_bundle_id`, `matched_audio_bundle_id`, `source_track_id`, `speaker_id`
- 설계: canonical `direction_id`, `condition`, `root_slot`, old/new value, D/N unit 배열
- 출처: canonical `source_track_id`, 파생 `source_type`, provider/model/voice/version 또는 녹음 session
- 오디오: raw/canonical candidate, accepted utterance, prepared stimulus 각각의 URI/path,
  SHA-256, sample rate, channels, duration, codec와 timeline 좌표계
- 정렬: 모든 onset/offset, alignment tool/version/confidence
- 통제값: target/actual latency, post-final-value duration, post-cue duration, unit별 stale age
- 선택: candidate index, selection score, accepted, exclusion reason
- 품질: clipping, loudness, SNR 또는 noise flag, manual QC status
- 재현성: synthesis parameters, code commit, dataset version
- 배포: inferential role, optional analysis fold, license, consent/usage scope

## 5. 실행 계획

각 Step은 산출물과 종료 조건을 만족해야 다음 단계로 넘어간다.

### Step 0. 의사결정과 실험 명세 동결

할 일:

1. 목적을 `pretrained 모델의 controlled evaluation set`과 `학습 corpus` 중 하나로
   확정한다. 본 문서의 기본안은 전자다.
2. 언어, root slot, value pair, repair cue와 외적 타당성의 한계를 확정한다.
3. v2 release가 TTS 600개인지 human 600개인지, 또는 순차적으로 둘 다 만들지 정한다.
4. 대표 draft와 voice pool로 3.2절의 timing feasibility calibration을 수행한다.
5. TTS provider/모델/voice 8–12개와 사용·재배포 권리를 확인한다.
6. forced-alignment 도구와 실패 판정 기준을 정한다.
7. target latency, within-speaker 허용차, audio format, inferential role/fold, 공개 범위를
   확정한다.
8. primary endpoint, response window, partial/no-response 처리, annotation rubric,
   adjudication과 exact statistical model을 사전등록한다.
9. generation seed 수와 반복측정 집계 단위를 고정한다. 현재 pipeline의 5 seeds를
   사용하면 트랙당 `600 × 5 = 3,000` model outputs, 이중 annotation은 기본 6,000
   judgments임을 비용·일정에 반영한다.
10. 가정한 baseline rate, effect size, scenario/speaker ICC로 power 또는 MDE simulation을
    수행해 scenario 30개와 voice 8–12개가 충분한지 확인한다.
11. candidate selection score의 항목·weight·tie-break와 같은 matched audio bundle에서
    고정할 synthesis parameter를 확정한다.
12. 사람 음성을 수집하면 동의서, 보상, 철회, 익명화, 보관 정책을 승인받는다.
13. GPU 시간, TTS 호출, candidate/accepted/prepared audio, annotation의 compute·storage·
    비용 budget을 smoke 추정치와 함께 승인한다.

산출물:

- `dataset_v2/config/dataset.yaml`
- `dataset_v2/DECISIONS.md`
- `dataset_v2/ANALYSIS_PROTOCOL.md`
- timing feasibility 및 power/MDE 보고서
- compute/storage/annotation budget
- TTS/사람 음성 라이선스 및 consent 확인 기록

종료 조건:

- 생성과 수집을 막는 `TBD`가 없다.
- 데이터 수량과 공개 가능 범위가 문서로 승인되었다.
- endpoint, 표본 수, timing target과 비용이 model output을 보기 전에 동결되었다.

### Step 1. 스키마와 재현 가능한 골격 구현

할 일:

1. blueprint, generated script, audio, eval trial, annotation manifest JSON Schema를 만든다.
2. ID 규칙을 고정한다.
3. 설정 파일 하나에서 수량, 조건, timing threshold를 읽게 한다.
4. 단위 테스트 fixture로 scenario 1개를 추가한다.
5. 기존 E1–E9 pipeline과 v2 출력 경로가 섞이지 않는지 확인한다.

권장 ID:

```text
text_bundle_id          = travel_001__a_to_b
script_id               = travel_001__a_to_b__delayed_three_dependencies
source_track_id         = tts_kokoro_v1_0_r1
matched_audio_bundle_id = travel_001__a_to_b__tts_kokoro_v1_0_r1__spk03
rendition_target_id     = travel_001__a_to_b__delayed_three_dependencies__tts_kokoro_v1_0_r1__spk03
candidate_id            = travel_001__a_to_b__delayed_three_dependencies__tts_kokoro_v1_0_r1__spk03__cand01
accepted_audio_id       = travel_001__a_to_b__delayed_three_dependencies__tts_kokoro_v1_0_r1__spk03__accepted
```

종료 조건:

- 올바른 fixture는 통과하고 필드 누락·중복 ID·잘못된 enum fixture는 실패한다.
- 생성물은 같은 입력과 seed에서 byte-identical하게 재생성된다.

### Step 2. 의미 설계도 30개 작성

할 일:

1. 각 scenario에 D1–D3, N1–N3와 typed `state_patch`를 작성한다.
2. D/N unit의 단어 수와 예상 발화 길이를 가능한 한 비슷하게 맞춘다.
3. 모든 문장을 A→B와 B→A로 읽어 양방향 자연성을 확인한다.
4. neutral 문구에 root 참조가 없는지 점검한다.
5. D/N의 speech act, nonterminal boundary와 예상 pause가 대응되는지 점검한다.
6. root-invariant closing prompt가 모든 조건을 같은 terminal request로 끝내는지 확인한다.
7. one-dependency identity와 pre-position을 각각 10/10/10으로 배정한다.
8. D identity × pre-position 3×3 cell을 각 3–4개로 만드는 rotation을 고정한다.
9. 30개 전체의 delayed composition을 대표 voice로 저비용 duration screen한다.
10. scenario별 target/stale value alias와 relation별 응답 판정 예시를 answer key 초안에
   작성한다.
11. lexical overlap과 특정 구문 반복을 보고 필요하면 설계도 수준에서 수정한다.

검토 질문:

- root가 바뀌면 이 D unit의 referent도 반드시 바뀌는가?
- root가 바뀌어도 이 N unit의 의미와 참값은 그대로인가?
- 모든 조건이 동일한 최종 `gold_state`로 해석되는가?
- 두 방향 모두 어색하거나 사실상 불가능한 수정은 아닌가?
- 길이를 맞추기 위한 무의미한 filler가 들어가지는 않았는가?
- D와 N의 담화행위·종결성 차이가 조기 응답을 유도하지 않는가?
- relation별 new/old binding을 응답에서 실제로 판정할 수 있는가?

종료 조건:

- 30개 모두 작성자 외 2명의 의미 검토를 통과한다.
- D1/D2/D3 identity와 pre-position counterbalance가 각각 10/10/10이고 cross-cell은
  3–4개다.
- duration screen의 `N N N`, `D N N`, `D D D` 분포가 동결한 공통 timing 구간을
  지지한다.
- `review_status=approved`가 아닌 설계도는 생성 입력에서 제외된다.

### Step 3. 조건 생성기와 정적 검증 구현

할 일:

1. 각 설계도에서 A→B, B→A를 만든다.
2. 각 방향에서 5조건의 segment 배열을 생성한다.
3. rotation table을 적용해 one-dependency의 위치를 counterbalance한다.
4. transcript와 구조화된 segment를 함께 저장한다.
5. 6장의 generated script 검증 규칙을 모두 자동화한다.
6. snapshot/golden test로 생성 순서의 회귀를 막는다.

정확한 수량 gate:

```text
scenario = 30
text_bundle = 30 × 2 = 60
script = 60 × 5 = 300
```

종료 조건:

- 테스트 전체 통과, script 300개, 중복 ID 0개, schema error 0개다.
- 같은 입력에서 생성한 manifest의 SHA-256이 재실행 후 동일하다.

### Step 4. 대본 300개 의미·자연성 QC 및 text freeze

할 일:

1. 자동 검증 보고서를 검토한다.
2. 각 text bundle의 다섯 `gold_state`가 같은지 구조 비교한다.
3. 검토자는 조건명을 숨긴 상태에서 문법·자연성·최종 의도를 판정한다.
4. D/N 길이 분포, repair 전후 예상 길이, 어휘 분포를 시각적으로 확인한다.
5. 수정 사항은 생성 대본이 아니라 blueprint에 반영하고 전체를 재생성한다.
6. 승인된 300개에 `text_release_id`와 hash를 부여한다.
7. Step 0의 release-fold 규칙과 seed로 scenario-level analysis fold를 생성·동결해
   Step 5 assignment 입력으로 넘긴다.

종료 조건:

- 오류 0개, 미합의 의미 판정 0개다.
- production 합성 이후 대본을 바꾸려면 dataset minor version을 올리도록 freeze한다.
- 모든 scenario에 analysis fold가 있고 같은 text bundle의 다섯 script가 한 fold에 있다.

### Step 5. 화자 배정과 녹음 순서 생성

10명 풀을 쓰는 기본 예시는 다음과 같다.

```text
60 text bundles × 2 speakers = 120 matched audio bundle assignments
120 assignments ÷ 10 speakers = 12 matched audio bundles per speaker
12 matched audio bundles × 5 conditions = 60 rendition targets per speaker
```

할 일:

1. 각 text bundle에 서로 다른 화자 2명을 배정한다.
2. 같은 text bundle의 5조건은 같은 두 화자가 모두 발화하게 한다.
3. 화자별 matched audio bundle 수 차이를 최대 1로 제한한다.
4. direction, D1/D2/D3 group, analysis fold가 화자마다 치우치지 않게 최적화한다.
5. Step 6의 shortest/median/longest smoke text bundle에는 Step 0에서 찾은
   fastest/slowest voice를 두 화자로 예약 배정한다. 이 60개는 별도 가상 target이 아니라
   production 600개의 부분집합으로 재사용한다.
6. 같은 matched audio bundle의 5조건이 연속하지 않도록 화자별 녹음 순서를 randomize한다.
7. seed와 알고리즘 버전을 assignment manifest에 저장한다.

종료 조건:

- text bundle당 화자 정확히 2명, 화자당 조건별 수 동일, 중복 assignment 0개다.
- 동일 seed로 assignment와 recording order를 재현할 수 있다.

### Step 6. TTS production smoke 60개와 threshold 검증

30개 중 predicted duration이 shortest/median/longest이면서 D1/D2/D3 group을 하나씩
대표하는 scenario 3개를 고른다. Step 0 voice screen의 fastest/slowest voice를 두 화자로
사용해 속도와 정렬의 극단을 포함한다.

```text
3 scenarios × 2 directions × 5 conditions × 2 speakers = 60 rendition targets
60 rendition targets × 1 deterministic candidate = 60 candidate audio
```

할 일:

1. rendition target 하나당 통문장 후보 1개를 생성한다.
2. voice, rate, style, seed와 가능한 경우 provider request ID를 기록한다.
3. 같은 matched audio bundle에서 provider/model/version, voice, rate, style, SSML prosody
   control, sentence-boundary와 pause policy를 동일하게 유지한다.
4. forced alignment로 필수 timing event와 unit별 onset/offset을 추출한다.
5. 저신뢰 정렬과 repair cue 경계를 사람이 확인한다.
6. latency와 post-repair duration이 Step 0에서 동결한 threshold를 만족하는지 검증한다.
7. 동결된 selection score weight와 tie-break가 올바른 후보를 고르는지 확인한다.
8. 전체 문장 합성의 억양과 repair prosody가 자연스러운지 청취한다.

실패 시 순서:

1. 동일 설정 반복 생성은 금지한다. 재시도가 필요하면 policy revision을 먼저 동결하고
   해당 matched audio bundle 다섯 조건 전체를 재생성한다.
2. 의미 unit 문장 길이를 blueprint에서 조정하고 해당 scenario의 전체 text/audio
   bundle을 재생성한다.
3. 문장 경계 pause를 미세 조정할 때는 condition 하나만 바꾸지 않는다. 같은 boundary
   role의 policy를 matched audio bundle 다섯 조건에 동일 적용하고 전체를 재생성한다.
4. 강한 time-stretch나 서로 다른 후보의 조각 결합은 사용하지 않는다.

동결 threshold가 체계적으로 불가능하면 Step 0 결정을 versioned change로 다시 열고 모든
smoke를 재실행한다. 특정 condition이나 voice에만 rate/style/threshold를 다르게 적용하지
않는다.

종료 조건:

- 60개 모두 alignment와 audio QC를 통과한다.
- delayed 세 조건이 동결된 latency/post-duration 기준 안에 들어온다.
- 생성 비용과 실패율로 전체 production 규모를 예측할 수 있다.

### Step 7. 전체 후보 음성 생성 또는 사람 녹음

TTS track:

1. 300개 대본과 assignment manifest를 입력으로 후보를 생성한다.
2. rendition target마다 고정 후보 1개를 만들고 candidate별 checkpoint를 즉시 기록한다.
3. 로컬 오류는 `--resume`으로 완료 candidate를 hash 검증해 건너뛰며 같은 요청을 중복
   accepted 처리하지 않는다.
4. 원본 후보는 immutable 경로에 저장하고 hash를 즉시 계산한다.

Human controlled-reading track:

1. 화자에게 condition label과 matched audio bundle 구조를 노출하지 않는다.
2. 무작위화된 대본과 일관된 마이크 거리·gain·환경을 사용한다.
3. 세션 시작 때 representative delayed 문장을 녹음해 speaker별 자연 속도 target과
   within-speaker tolerance를 계산한다.
4. 동일 rendition target의 retake는 기본 최대 3회로 제한하고 최종 횟수를 Step 0에서
   동결한다.
5. 전역 절대시간보다 같은 speaker의 세 delayed 조건 range를 acceptance의 핵심으로
   사용한다. 느린 화자를 사후 제외하거나 부자연스럽게 빨리 읽게 하지 않는다.
6. 오류·기침·외부 소음은 현장에서 표시하고 재녹음하되 일반 QC 원본도 보존한다.
7. tolerance 실패가 반복되면 해당 파일만 임의 제외하지 않고 matched audio bundle
   전체를 재녹음하거나 사전 정의 `incomplete_bundle`로 처리한다.
8. speaker ID는 가명으로 저장하고 직접 식별 정보는 별도 접근제어한다.

`immutable`과 원본 보존은 일반 재시도·QC에 대한 원칙이다. consent withdrawal,
직접식별정보 유출 또는 적법한 삭제 요청이 있으면 해당 audio/object를 즉시 purge하고
내용 없는 tombstone과 감사 기록만 남기는 예외 절차를 Step 0 정책에 둔다.

종료 조건:

- `(script_id, source_track_id, speaker_id)`로 정의된 rendition target 600개 모두에
  최소 한 개 이상의 정렬 가능한 후보가 있다.
- 누락·중복 업로드가 없고 후보 manifest와 저장소 hash가 일치한다.

### Step 8. Forced alignment와 후보 선택

할 일:

1. raw candidate를 PCM 24 kHz mono `canonical candidate`로 변환한다. 이 단계에는
   prefix/suffix silence와 Mimi frame padding을 넣지 않는다.
2. raw/canonical candidate의 URI, hash와 duration mapping을 각각 저장한다.
3. transcript와 canonical candidate를 동일한 정규화 규칙으로 align한다.
4. 모든 word/unit-level timestamp와 confidence를 원본 형식으로 보존한다.
5. 필수 이벤트와 unit age를 파생하고 계산식을 한 함수에서 적용한다.
6. 후보 선택 score는 latency 오차, post-duration 오차, alignment confidence,
   clipping/noise를 사용한다.
7. condition 성능이나 모델 응답은 후보 선택에 사용하지 않는다.
8. 자동 선택 결과를 condition-blind 청취 QC로 확인한다.
9. 선택한 canonical candidate에 gain normalization만 적용해 `accepted canonical
   utterance`를 만들고 새 hash를 저장한다. 시간축은 바꾸지 않는다.

종료 조건:

- accepted audio 600개 모두 조건별 필수 timing event가 있다. Clean은 initial-old,
  repair-cue, repeated-old 필드만 명시적 `null`이고 new-value onset/offset은 존재한다.
- 수동 수정 timestamp에는 reviewer와 변경 전후 값이 audit log로 남는다.
- 후보 탈락 사유가 빈 값인 rejected candidate가 없다.

### Step 9. 오디오·메타데이터 최종 QC

오디오 lifecycle은 다음 순서로 고정한다.

```text
raw candidate
→ canonical candidate (24 kHz mono, no prefix/suffix/padding)
→ accepted canonical utterance (selected + gain normalized)
→ prepared stimulus (prefix/suffix + Mimi frame padding)
```

Accepted canonical utterance 표준:

- PCM WAV, 24 kHz, mono
- active-speech RMS 목표 −23 dBFS
- peak 최대 −1 dBFS
- prefix/suffix silence와 frame padding 없음
- clipping, 잘림, 비정상 무음, channel 오류 없음

Prepared stimulus 표준:

- accepted audio와 내용·sample rate·channel 동일
- config의 prefix/suffix silence 적용
- Mimi 80 ms frame인 1,920 samples의 정수배로 끝 padding
- timestamp는 prefix shift만큼 이동하고 content-relative 원본도 함께 보존

raw candidate, canonical candidate, accepted utterance, prepared stimulus의 URI/hash를 서로
다른 필드와 manifest에 저장하며 파일을 덮어쓰지 않는다.

할 일:

1. accepted utterance의 schema, path, hash, codec/sample rate/channel, duration과 timestamp
   순서를 전수 검사한다.
2. condition별 loudness, duration, latency, post-duration 분포를 비교한다.
3. speaker별 accepted 수와 재시도율을 비교한다.
4. 100% 자동 QC와 최소 20% 이중 청취 QC를 수행한다.
5. repair 경계와 alignment 저신뢰 표본은 100% 수동 검토한다.
6. accepted utterance에서 prepared stimulus를 1:1 생성하고 frame multiple, prefix/suffix,
   shifted timestamp 재계산값을 검증한다.
7. 일반 QC 제외 파일은 quarantine하며 이유를 enum으로 기록한다. consent/PII 삭제
   대상은 quarantine하지 않고 purge/tombstone 정책을 적용한다.

종료 조건:

- accepted audio는 rendition target과 1:1로 정확히 600개이고 matched audio bundle은
  정확히 120개다.
- 조건·화자·방향별 누락 0개, 미해결 QC 0개다.
- QC 보고서가 데이터 hash와 함께 생성된다.

### Step 10. Inferential set/fold 검증과 release package 생성

단일 Boston–Seattle controlled evaluation의 primary inference에는 동결된 scenario 30개,
audio 600개를 모두 사용한다. 5개 scenario만 `test`로 떼면 cluster 수가 너무 작아지고
Step 0 power 계산과도 맞지 않는다. 따라서 model output을 보기 전에 protocol, answer key,
전체 manifest hash를 동결하고 30개 전체를 `confirmatory_evaluation`으로 표시한다.

| inferential role | scenarios | audio | 사용 |
|---|---:|---:|---|
| confirmatory evaluation | 30 | 600 | 사전등록 primary/secondary 분석 |
| optional release fold 1–5 | fold당 6 | fold당 120 | resampling·도구 편의, test split 아님 |

Step 6의 60개 production smoke에서는 오디오 생성·정렬·QC만 확인하고 Moshi outcome을
보지 않는다. 평가 adapter와 metric의 end-to-end 개발은 기존 E1–E9 또는 core에 포함되지
않는 pilot-only fixture로 수행한다. 한 scenario의 양방향, 다섯 조건, 두 화자 rendition은
반드시 같은 optional fold에 둔다.

같은 speaker가 여러 fold에 나타날 수 있으므로 optional fold로 화자 일반화를 주장하지
않는다. 학습 corpus 목적을 선택했다면 speaker 및 value-pair/root-slot holdout은 선택
사항이 아니라 Step 0의 필수 설계이며, 위 inferential set과 수량표를 사용하지 않는다.

Release package:

- 설계도, generated scripts, accepted audio manifest
- inferential-role/fold manifest와 audio hash 목록
- schema와 validator
- dataset card, 라이선스, TTS 출처 또는 human consent 범위
- 생성 config, tool version, Git commit, random seed
- QC report와 알려진 한계
- 재현 명령과 checksum 검증 명령

종료 조건:

- 새 환경에서 manifest 검증과 script 재생성이 성공한다.
- 공개 package의 audio hash가 accepted manifest와 전부 일치한다.
- dataset version tag를 생성하기 전 라이선스 검토가 완료된다.

### Step 11. Baseline과 분석 검증

현재 `prepare_stimuli.py`, `run_eval.py`, annotation/scoring 흐름을 재사용하되 v2 schema를
읽는 adapter를 추가한다. 기존 도시명 전용 heuristic이 구조화된 old/new value와
gold state를 사용하도록 일반화 가능한지 먼저 확인한다. 자동 heuristic은 명확한 도시명
응답의 triage에만 사용하고, 간접 추천이나 모호한 응답을 최종 정답으로 자동 확정하지
않는다.

Eval trial의 primary key는 다음과 같다.

```text
eval_run_id   = model_repo × resolved_revision × generation_config_hash × code_commit
eval_trial_id = accepted_audio_id × eval_run_id × generation_seed
```

트랙당 600 audio와 5 seeds면 eval run당 3,000 trials다. 각 trial에는 `eval_run_id`, response
audio/text, stream-relative timestamp, user timeline mapping, early/final label을 저장한다.

#### 응답 window와 annotation rubric

- `pre_cue_window`: stimulus 시작부터 `repair_cue_onset_ms` 직전까지
- `cue_in_progress_window`: repair cue 시작부터 `repair_cue_offset_ms`까지
- `post_cue_complete_window`: cue 종료부터 `utterance_end + response_capture_ms`까지
- `post_user_window`: `closing_prompt_offset_ms`부터 capture 종료까지
- Clean과 repair의 primary final evidence window: `post_user_window`
- Clean의 `new_value_offset_ms` 이후와 repair의 `repair_cue_offset_ms` 이후부터 사용자 종료 전까지는
  full-duplex early/recovery 진단 window로만 사용
- `final_active_state`: 해당 final evidence window에서 마지막으로 확인 가능한 활성 root 상태
- `response_capture_ms`: 현재 영어 pipeline의 40초를 후보로 하되 Step 0에서 고정

각 D relation은 `new_bound`, `old_bound`, `both`, `unresolved`, `not_addressed` 중 하나로
라벨한다. Trial 전체에는 기존 taxonomy를 확장한 다음 label을 하나 부여한다.

- `target_only`: final active state가 new value이고 old value를 활성 상태로 유지하지 않음
- `stale_only`: old value만 활성 상태로 유지
- `both`: old/new를 둘 다 활성 요청으로 처리
- `recovered`: pre-cue 또는 cue-in-progress stale 반응 뒤 post-cue-complete window에서
  new value로 명시적 회복
- `clarification`, `irrelevant`, `no_speech`, `unintelligible`, `no_evidence`

cue가 진행되는 동안 new value를 듣기 전 또는 뒤의 `not {old}`를 다 듣기 전에 나온 old
반응은 final stale로 바로 코딩하지 않고 cue-in-progress 진단으로 분리한다. 도시명을 직접
말하지 않은 추천은 scenario answer key의 value alias, 장소·음식·숙소 등
relation별 증거와 전체 담화를 이용해 사람이 판정한다. 일부 relation만 답한 경우 나머지는
`not_addressed`로 두며 정답을 추정하지 않는다.

Primary `final_target_correct=1`은 마지막 활성 root가 new이고 old가 활성 상태로 남지 않은
경우만 해당한다. clarification/no-response/no-evidence는 보수적으로 0으로 두고 각각의
비율과 `scorable_coverage`를 별도 보고한다. `stale_state_error=1`은 답한 relation 중
하나라도 old 또는 both binding이 남은 경우다. Relation-level rebinding은 전체 세 relation
기준 점수와 addressed-only 점수 및 coverage를 함께 보고해 partial answer를 숨기지 않는다.

각 eval trial은 condition과 seed를 숨긴 두 평가자가 독립 라벨링하고, 불일치는 세 번째
adjudicator가 해결한다. raw agreement와 Cohen's kappa 또는 사전 지정한 다범주 agreement
지표를 보고한다. Answer key와 rubric은 model output을 보기 전에 동결한다.

Primary 분석:

1. clean gate와 immediate repair 성능을 먼저 보고한다.
2. delayed 0/1/3의 target-state accuracy와 stale-state error를 비교한다.
3. `actual_latency_ms`와 `post_repair_duration_ms`를 공변량으로 사용한다.
4. pre-repair 조기 응답률의 조건 차이가 사전 tolerance를 넘는지 causal-interpretation
   gate로 확인하되 primary outcome의 단순 공변량으로 넣지 않는다.
5. primary 분석은 rendition target마다 5 seeds의 `successes/5` binomial count로 먼저
   집계하며 seed random effect를 다시 넣지 않는다.
6. bootstrap은 120 matched audio bundle을 독립 cluster로 취급하지 않고 scenario를
   1차 cluster로 재표집해 그 안의 방향·화자·조건을 함께 보존한다. 필요하면 speaker를
   포함한 multiway cluster sensitivity를 추가한다.
7. GLMM을 쓰면 direction은 fixed effect로 두고 scenario와 speaker random intercept,
   scenario별 condition slope를 Step 0에 정확히 사전 지정한다. 별도 trial-level sensitivity
   model에서만 3,000 seed trials를 유지하고 seed effect를 포함하며 primary 집계 모델과
   혼합하지 않는다.
8. D1/D2/D3 identity, pre-position과 stale-dependency age subgroup을 확인한다.

저장소 기존 지표와 연결할 경우 다음을 함께 보고한다.

- Target Selection
- SIER(stale intent error rate)
- early-stale response와 recovery
- clean 대비 repair gap(CRG)
- 조건별 95% confidence interval

해석 gate의 기본값은 기존 파일럿과 맞춰 clean target rate 80%로 두되 Step 0에서
사전 확정한다. clean gate 실패 시 self-repair 또는 dependency 효과를 주장하지 않는다.

종료 조건:

- 작은 fixture의 수기 계산과 metric 출력이 일치한다.
- 같은 config/seed로 baseline 결과 집계가 재현된다.
- 결과 표에 목표 latency가 아니라 실제 latency 요약이 포함된다.

### Step 12. Human validation과 자연발화 확장

권장 순서:

1. TTS controlled smoke
2. TTS core 600
3. Human controlled-reading smoke
4. Human controlled-reading core 또는 사전 지정 validation subset
5. Human elicited-natural extension

Elicited-natural에서는 전체 문장 대신 initial value, pre-repair 의미 카드, repair 시점,
remaining information을 보여 준다. 자연성은 높아지지만 길이와 표현 통제가 약해지므로
controlled track과 별도 version/analysis로 관리한다.

종료 조건:

- TTS와 human 결과를 source별로 보고한다.
- TTS의 지나치게 명확한 repair prosody가 사람 음성 결과를 과대예측하는지 확인한다.

### Step 13. Pause-based latency extension 120개

핵심 600개는 immediate와 delayed라는 두 시간 수준에 가깝기 때문에 완전한 latency
curve를 만들 수 없다. 별도 extension에서는 dependency를 0으로 고정하고 모든 delay
bin에서 동일한 spoken content를 사용한다.

```text
10 scenarios
× 2 directions
× 3 empirical delay bins
× 2 speakers
= 120 audio
```

extension의 고정 배열은 다음과 같다.

```text
old root + fixed short neutral carrier + [controlled pause] + repair
         + remaining fixed N/D units
```

세 bin은 같은 transcript, voice, rate, style과 SSML template을 사용하되 지정된 break
duration만 바꾼 통문장 합성으로 만든다. post-repair content와 target-duration 기준도
같게 유지한다. 통문장 재합성이므로 post-repair waveform이 byte-identical하다고
가정하지 않는다. 1.5/3.5/5.5초는 후보값이며 fixed carrier의
자연 duration보다 짧을 수 있으므로 calibration 뒤 실제 가능한 bin을 동결한다. silence는
turn-taking cue 자체이므로 이 결과를 일반적인 “순수 시간 효과”로 부르지 않고
`pause-based elapsed-delay effect`로 제한해 해석한다. `pause_duration_ms`, 실제 latency와
pre-repair assistant onset을 모두 저장한다. 계속된 음성 아래의 시간 효과가 필요하면
lexical/semantic load가 달라지는 별도 continuous-speech extension으로 설계한다.
bin마다 neutral unit 수나 filler 단어 수를 늘리는 방식은 semantic/lexical load도 함께
바꾸므로 이 extension에서 금지한다.

10개 scenario는 D1/D2/D3 counterbalance group에서 가능한 한 균등하게 뽑고, core와
같은 speaker·repair cue·audio QC를 사용한다. core 600과 extension 120을 합쳐 720개로
배포할 수 있지만 manifest와 분석은 분리한다.

종료 조건:

- 각 latency bin의 실제 분포가 겹치지 않고 사전 정의 범위에 들어온다.
- 세 bin의 spoken semantic/lexical content와 post-repair duration이 동일하다.
- dependency 효과와 pause-based latency 효과를 혼동하지 않고 별도 모델로 보고한다.

## 6. 자동 검증 명세

### 6.1 Blueprint 검증

- scenario ID가 30개이며 중복이 없다.
- D1–D3, N1–N3가 각각 정확히 하나다.
- relation path, typed `state_patch`, binding과 `gold_state_template`이 모든 unit에 대응한다.
- P1–P3 `balance_pair_id`마다 D와 N이 정확히 하나씩 있다.
- D/N pair의 `speech_act`와 `boundary_type`이 primary 설계 규칙을 만족한다.
- `value_a != value_b`이고 두 방향 문장이 생성 가능하다.
- repair template에 `{new}`와 `{old}`가 각각 정확히 한 번 있다.
- root-invariant terminal `closing_prompt`가 있고 root/value placeholder가 없다.
- neutral unit에 값, 별칭, 지역 지시어, root 지시 표현이 없다.
- one-dependency identity와 pre-position이 각각 10/10/10이고 3×3 cell은 3–4개다.
- scenario answer key에 old/new alias, relation별 증거와 모호 사례가 있다.
- reviewer 2명의 승인과 blueprint hash가 있다.

### 6.2 Generated script 검증

- text bundle 60개, condition별 script 60개, 총 300개다.
- 모든 text bundle에 5조건이 정확히 하나씩 있다.
- D1–D3, N1–N3가 조건마다 정확히 한 번 등장한다.
- closing prompt가 모든 조건의 마지막 segment에 정확히 한 번 있고 D/N count에서 제외된다.
- delayed 조건의 pre/post unit 수가 각각 3개다.
- dependency count가 neutral/one/three에서 정확히 0/1/3이다.
- `repair_rebindings` 수가 dependency count와 같고 old-bound relation을 new로 바꾼다.
- delayed-one의 D identity, pre-position과 rotation이 blueprint와 일치한다.
- immediate의 repair 전 dependent unit 수가 0이다.
- 양방향에서 old/new가 정확히 뒤집힌다.
- 같은 text bundle의 repair 조건들은 cue 형식이 같고, 5조건의 최종 `gold_state`가 같다.
- unit `state_patch` multiset이 5조건과 양방향에서 보존되고 root binding만 new로 해석된다.
- old value는 repair 조건의 `initial_old_root`와 `repair_cue`에서만 허용된다.
- new value는 clean root 또는 repair cue의 지정된 segment에 존재한다.
- transcript는 segment join의 정규화 결과와 일치한다.
- 금칙 표현과 예상치 못한 도시명·고유명사가 없다.

### 6.3 Assignment 검증

- 각 text bundle은 서로 다른 화자 2명에게 배정된다.
- 각 assignment는 다섯 조건을 모두 포함한다.
- 화자별 matched audio bundle 수 차이는 1 이하다.
- 화자별 direction, D group, analysis fold 분포가 설정된 tolerance 이하다.
- smoke 6개 text bundle의 두 speaker가 예약한 fastest/slowest voice와 일치한다.
- 녹음 순서에서 같은 matched audio bundle 조건이 연속하지 않는다.

### 6.4 Audio 및 alignment 검증

- accepted audio와 `(script_id, source_track_id, speaker_id)` rendition target이 1:1이고
  트랙당 총 600개다.
- rendition target, candidate, accepted ID가 충돌하지 않고 accepted 후보는 target당 하나다.
- `selected_candidate_id`가 존재하는 canonical candidate를 가리키고 raw→canonical→
  accepted→prepared hash lineage가 끊기지 않는다.
- accepted canonical utterance의 sample rate/channel/codec이 config와 같고 prefix/suffix나
  frame padding이 없다.
- prepared stimulus가 accepted audio와 1:1로 연결되고 prefix/suffix, frame multiple과
  shifted timestamp 재계산값이 config와 같다.
- 파일 hash, duration, manifest path가 일치한다.
- clipping, empty audio, 잘린 첫·끝 단어가 없다.
- timestamp가 단조 증가하고 duration 범위 안에 있다.
- `actual_latency_ms`와 `post_repair_duration_ms` 재계산값이 manifest와 같다.
- unit별 onset/offset과 `stale_dependency_age_ms` 재계산값이 manifest와 같다.
- delayed latency 및 post-duration이 동결된 tolerance를 만족한다.
- 같은 matched audio bundle의 provider/model/version/voice/rate/style/SSML prosody
  control/pause policy가 condition별로 달라지지 않는다.
- 같은 matched audio bundle의 세 delayed 조건 repair cue가 청각적으로 식별 가능하다.
- alignment failure와 manual override가 audit log에 남는다.

## 7. 품질 게이트와 재작업 정책

| 상태 | 의미 | 처리 |
|---|---|---|
| `accepted` | 모든 자동·필수 수동 QC 통과 | release 후보 |
| `retry_synthesis` | timing/prosody/오디오 문제 | frozen policy revision 후 matched bundle 전체 재생성 |
| `revise_blueprint` | D/N 의미 또는 길이 문제 | 설계도 수정 후 관련 text bundle과 matched audio bundles 전체 재생성 |
| `manual_alignment` | 음성은 정상이나 정렬 저신뢰 | condition-blind 수동 경계 확인 |
| `quarantined` | 손상 또는 해결 가능한 라이선스 검토 | release 제외, 원인 기록 |
| `purged` | consent 철회·PII 삭제 의무 | object 삭제, 내용 없는 tombstone만 보존 |

한 condition만 대본을 수정하면 matched bundle이 깨질 수 있으므로 blueprint가 바뀌면
해당 scenario의 양방향·5조건과 영향을 받은 speaker rendition을 모두 재생성한다.
production 시작 후 threshold를 낮춰 파일을 살리는 대신 versioned decision log에 변경
근거를 남기고 전체 분포를 재검증한다.

## 8. 리스크와 대응

| 리스크 | 영향 | 예방·대응 |
|---|---|---|
| latency와 dependency 수 혼재 | 인과 해석 불가 | delayed pre/post 3단위 고정, 실제 latency 공변량 사용 |
| post-repair 회복시간 차이 | 조건별 회복 기회 차이 | new-value offset과 utterance end 측정·통제 |
| neutral이 암묵적으로 root 의존 | dependency count 오표기 | 금칙어 검사 + 독립 검토자 2명 |
| D/N 길이·어휘량 차이 | 시간·처리량 confound | blueprint 단계 길이 매칭, 통문장 후보 선택 |
| D 질문/N 진술 차이 | full-duplex 조기응답 confound | 동일 speech act·nonterminal prosody, early-onset causal gate |
| delayed-one이 항상 같은 D | 의미 유형과 수 효과 혼재 | D1/D2/D3 10/10/10 counterbalance |
| delayed-one 위치 불균형 | stale age와 count 혼재 | pre-position 10/10/10, unit age 측정 |
| 방향별 자연성 차이 | A→B/B→A confound | 양방향 낭독 검토 및 direction 보고 |
| TTS repair가 지나치게 명확 | 성능 과대평가 | human controlled-reading validation |
| 조각 TTS 또는 time-stretch | 비자연적 prosody | 통문장 합성, pause만 미세 조정 |
| 조건 연속 녹음 | 암기·순서 효과 | 화자별 전체 순서 randomize |
| speaker/voice 불균형 | 조건 효과 왜곡 | matched audio bundle 단위 동일 화자, assignment validator |
| alignment 오류 | latency 측정 오류 | confidence gate, 저신뢰 전수 수동 검토 |
| 잘못된 통계 cluster | CI 과소추정 | scenario 1차 cluster 또는 사전등록 GLMM |
| scenario fold 누수 | resampling 오류 | 한 scenario의 양방향·조건·rendition을 같은 fold에 배치 |
| speaker fold 중복의 오해 | 화자 일반화 과대주장 | fold는 speaker holdout이 아님을 dataset card에 명시 |
| Git 저장소 비대화 | clone/push 실패 | audio 외부 저장, Git에는 hash/manifest만 저장 |
| 음성 권리·개인정보 문제 | 공개 불가 | 제작 전 라이선스/consent gate, 가명 ID |

## 9. 마일스톤과 예상 작업량

아래 수치는 캘린더 약속이 아니라 1인 기준 engineering/research effort의 1차 추정이다.
라이선스, 사람 검토, GPU, annotation은 병목이 별도이므로 smoke에서 측정한 처리량으로
캘린더와 budget을 다시 계산한다.

| 마일스톤 | 포함 Step | 내부 effort | 별도 elapsed-time 요인 | 완료 기준 |
|---|---|---:|---|---|
| M0 설계 동결 | 0 | 3–6 person-days | 라이선스·윤리·예산 승인 | blocking TBD 0개 |
| M1 데이터 계약 | 1 | 2–3 person-days | 없음 | schema/fixture test 통과 |
| M2 설계도 동결 | 2 | 4–7 person-days | 30개 × 독립 검토자 2명 | 30개 이중 승인 |
| M3 대본 동결 | 3–4 | 3–5 person-days | text review turnaround | 300개 검증·hash 고정 |
| M4 TTS smoke | 5–6 | 3–5 person-days | TTS/정렬 호출 | 60개 timing/QC 통과 |
| M5 core audio | 7–9 | 5–10 person-days | provider rate limit 또는 녹음 세션 | accepted 600개 |
| M6 release/baseline | 10–11 | 4–8 person-days | GPU 3,000 trials와 annotation | package와 결과 재현 |
| M7 human/latency 확장 | 12–13 | smoke 후 산정 | 모집·녹음·추가 annotation | 별도 acceptance gate 통과 |

M6 annotation 시간은 추측하지 않고 50개 output annotation pilot의 median seconds로
다음처럼 계산한다.

```text
base_hours = 3,000 outputs × 2 annotators × median_seconds / 3,600
total_hours = base_hours + 3,000 × disagreement_rate × adjudication_seconds / 3,600
```

Critical path는 `Step 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11`이다.
Step 5의 assignment 로직, 평가 adapter 준비, 라이선스 검토는 schema가 정해진 뒤
부분적으로 병렬 진행할 수 있지만 최종 assignment는 text freeze 뒤 확정한다.

## 10. Definition of Done

핵심 TTS 데이터셋 v2.0.0은 다음을 모두 만족할 때 완료다.

- [ ] 승인된 의미 설계도 30개가 있다.
- [ ] 양방향 text bundle 60개와 5조건 script 300개가 자동 생성된다.
- [ ] D1/D2/D3 identity와 pre-position 배정이 각각 10/10/10이고 cross-cell이 3–4개다.
- [ ] 모든 script가 schema, 의미 보존, segment 수 검사를 통과한다.
- [ ] matched audio bundle 120개를 이루는 rendition target과 accepted audio가 각각
  600개 있다.
- [ ] delayed 조건의 실제 latency와 post-repair duration이 동결 기준을 통과한다.
- [ ] alignment event, gold state, source, QC, hash, inferential role/fold metadata가 완전하다.
- [ ] 자동 QC 100%와 지정된 수동/이중 QC가 끝났다.
- [ ] baseline이 clean gate와 primary delayed contrast를 재현 가능하게 산출한다.
- [ ] 5-seed 기본안이면 3,000 eval trials의 이중 블라인드 annotation과 adjudication이
  끝나고 agreement가 보고되었다.
- [ ] dataset card, 라이선스, 알려진 한계, 재현 명령이 있다.
- [ ] 새 환경에서 validator, checksum, script regeneration이 통과한다.
- [ ] version tag와 immutable release artifact가 만들어졌다.

Human track과 pause-based latency extension은 각각 별도의 Definition of Done과 version을
갖는다.

## 11. 권장 구현 순서의 첫 커밋들

계획 승인 뒤 구현은 리뷰 가능한 작은 단위로 나눈다.

1. `Add self-repair dataset v2 schemas and fixture`
2. `Add semantic blueprint validator`
3. `Add 30 reviewed self-repair blueprints`
4. `Generate and validate 300 matched scripts`
5. `Add balanced matched-audio-bundle assignment`
6. `Add full-utterance synthesis and alignment pipeline`
7. `Add candidate selection and audio QC reports`
8. `Add dataset v2 evaluation adapter and release builder`

각 커밋에서는 generated 파일의 출처 config와 재생성 명령을 함께 기록한다. 최종 audio를
일반 Git에 실수로 추가하지 않도록 CI에서 파일 크기와 확장자 gate를 둔다.
