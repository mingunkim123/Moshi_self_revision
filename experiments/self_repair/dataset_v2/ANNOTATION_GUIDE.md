# Dataset v2 annotation guide

이 지침은 condition과 seed를 숨긴 두 평가자가 독립적으로 사용한다. 자동 문자열 규칙은
triage 제안만 만들며 최종 label을 기록하지 않는다.

## Primary window

Primary final-state 판정은 모든 조건에서 `closing_prompt_offset_ms` 이후부터 capture 종료까지
사용한다. Repair cue 도중의 old 응답은 final stale로 자동 판정하지 않는다. Cue 완료 뒤 사용자
종료 전 발화는 early/recovery 진단에만 사용한다.

Primary window에 assistant speech가 없으면 `overall_label=no_speech`,
`final_target_correct=false`로 기록한다. 알아들을 수 없는 speech는 `unintelligible`, 질문 되묻기는
`clarification`, 도시나 관계 binding의 증거가 없는 일반 조언은 `no_evidence`다.

## Root 판정

- target 도시만 활성 계획으로 다루면 `target_only`다.
- stale 도시만 활성 계획으로 다루면 `stale_only`다.
- 둘을 동시에 활성 계획으로 다루면 `both`다.
- early stale 반응이 있었지만 primary window에서 target으로 명시적으로 고치면 `recovered`다.

도시명을 직접 말하지 않아도 answer key의 명확한 도시 고유 entity가 있으면 binding 증거가 될
수 있다. 목록 밖 entity는 두 평가자가 외부 검색 없이 동일 도시로 식별할 수 있을 때만 쓴다.
일반적인 호텔·식당·박물관 조언은 도시 binding을 추정하지 않고 `unresolved`로 둔다.

## Relation 판정

D1–D3를 각각 `new_bound`, `old_bound`, `both`, `unresolved`, `not_addressed` 중 하나로
판정한다. 하나의 relation 답변을 다른 relation으로 전파하지 않는다. `stale_state_error=true`는
addressed relation 중 하나라도 `old_bound` 또는 `both`이면 참이다. Addressed-only accuracy와
세 relation 전체 accuracy, coverage를 모두 보고한다.

두 평가자가 어떤 최종 필드에서든 다르면 제3 adjudicator가 원 annotation을 보존한 채 최종본을
작성한다. 누락 label을 `false`로 바꾸지 않는다.
