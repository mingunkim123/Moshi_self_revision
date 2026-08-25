# Dataset v2 analysis protocol

Status: pre-output engineering draft. Freeze this file and its SHA-256 before running Moshi on the
30 core scenarios.

## Primary unit and endpoint

- Eval run: resolved model revision × generation-config hash × code commit.
- Eval trial: accepted audio × eval run × generation seed.
- Primary outcome: `final_target_correct`.
- Five generation seeds are aggregated into a successes-out-of-five binomial count per rendition
  target; seeds are not counted as independent samples.
- Primary comparison: delayed-neutral vs delayed-one vs delayed-three.
- Scenario is the primary resampling cluster. Direction is a fixed effect; scenario and speaker are
  crossed repeated factors.

## Causal-interpretation gates

- Clean and immediate-repair performance must pass the preregistered gate.
- Actual latency and post-cue/final-value durations must satisfy the frozen matching policy.
- A condition difference in assistant speech beginning before repair is a mediator/diagnostic gate,
  not a routine covariate. If it exceeds the frozen tolerance, report the effect as entangled with
  full-duplex turn taking.

## Annotation

Two condition-blind annotators label every eval trial. Disagreements receive a third adjudication.
Per-relation binding labels are `new_bound`, `old_bound`, `both`, `unresolved`, and
`not_addressed`. Overall labels are `target_only`, `stale_only`, `both`, `recovered`,
`clarification`, `irrelevant`, `no_speech`, `unintelligible`, and `no_evidence`.

The 60 answer keys are generated and schema-valid. Their human usability check and the production
audio-timing thresholds must still be frozen before core model output is viewed. The response,
aggregation, and inferential rules below are frozen engineering defaults.

### Response windows

- `pre_cue`: repair 조건의 stimulus 시작부터 `repair_cue_onset_ms` 직전까지.
- `cue_in_progress`: `repair_cue_onset_ms`부터 `repair_cue_offset_ms`까지.
- `post_cue_pre_user_end`: cue 완료부터 `closing_prompt_offset_ms`까지. Recovery와 full-duplex
  turn-taking 진단에만 사용한다.
- `post_user_primary`: 공통 `closing_prompt_offset_ms`부터 capture 종료까지. Clean과 repair의
  primary final-state 판정은 모두 이 window를 사용한다.

`cue_in_progress`의 old response는 final stale로 자동 코딩하지 않는다. Primary window에
speech가 없으면 앞선 response를 끌어와 정답으로 추정하지 않고 `no_speech`와
`final_target_correct=0`으로 둔다. 단, secondary whole-trial recovery label에는 앞선 window를
사용할 수 있다.

## Frozen primary analysis

- Analysis input is one binomial row per rendition target: `successes` out of five generation
  seeds. Missing seeds fail the release gate; they are not imputed.
- Within every scenario and condition, average the four direction × assigned-speaker rendition
  rates with equal weight. The scenario therefore contributes one value per condition regardless
  of its response count.
- Primary estimand: `delayed_three_dependencies - delayed_neutral` in percentage points for
  `final_target_correct`.
- Key secondary estimand: `delayed_one_dependency - delayed_neutral`. Relation accuracy,
  stale-state error, and dependency-count trend are secondary.
- Confidence interval: 10,000 scenario-cluster bootstrap replicates, seed `20260826`; resample the
  30 scenarios and preserve every direction, speaker, condition, and seed aggregate inside a
  selected scenario. Report the percentile 95% interval and the equal-weight point estimate.
- The two delayed-condition contrasts use Holm-adjusted two-sided p-values at family alpha .05.
  The bootstrap CI is the primary uncertainty statement; a scenario-paired t approximation is a
  sensitivity check.
- Direction is included as a two-level fixed effect and actual latency/post-final-value duration
  as centered continuous effects in the preregistered secondary binomial model. Scenario and
  speaker receive random intercepts; scenario receives a condition slope. Early assistant onset
  is not adjusted as an ordinary covariate because it can be a mediator.

The causal-interpretation gate fails if the largest delayed-condition difference in
`assistant_started_before_repair` exceeds 5 percentage points. The core capability gate requires
clean final-target accuracy of at least 80%, immediate-repair accuracy of at least 70%, and at
least 90% scorable primary windows. Failing a gate does not discard results; it limits the claim.

The design sensitivity simulation in `reports/power_mde_simulation.json` gives an 80%-power MDE
proxy of about 8 percentage points under its stated random-effect assumptions. This is a
conditional design calculation, not observed-model evidence; it must be rerun if pipeline pilot
ICC or baseline differs materially.
