# Seed 17 파일럿 분석

> 이 문서는 자동 증거 매칭을 이용한 단일-seed 기술·내용 파일럿입니다. 정식 5-seed 이중 인간 주석 결과가 아닙니다.

## 핵심 결과

- 완료 응답: **600개**, 기술 계약 실패 **0개**
- primary window의 보수적 target-only 증거: **22개 (3.7%)**
- stale 증거: **51개 (8.5%)**
- 등록된 도시명·별칭·예시 장소 증거 없음: **527개 (87.8%)**
  - 도시명 exact-match만 사용한 민감도 분석도 target-only **21개**, stale 증거 **50개**로 거의 같습니다.
- primary window lexical output 없음: **122개 (20.3%)**
  - 이 중 text token 자체가 완전히 빈 응답은 **117개 (19.5%)**
- 사용자 audio 시작 전 assistant text 시작: **100.0%**
  - 첫 assistant lexical token: `[400.0]` ms; 입력 prefix silence: `[480.0]` ms
  - 표준 인사로 시작한 trial: **600개 (100.0%)**
- 총 GPU 실행시간: **4.99시간**; 응답 audio 총 길이: **12.40시간**

## 조건별 자동 증거

| 조건 | n | target-only | stale 증거 | no evidence | lexical text 없음 | 평균 단어 |
|---|---:|---:|---:|---:|---:|---:|
| `clean_final` | 120 | 9.2% | 0.0% | 90.8% | 21.7% | 16.3 |
| `immediate_repair` | 120 | 6.7% | 1.7% | 91.7% | 22.5% | 20.4 |
| `delayed_neutral` | 120 | 1.7% | 17.5% | 80.8% | 17.5% | 18.5 |
| `delayed_one_dependency` | 120 | 0.0% | 10.0% | 90.0% | 23.3% | 14.0 |
| `delayed_three_dependencies` | 120 | 0.8% | 13.3% | 85.8% | 16.7% | 20.9 |

`target-only`는 answer key에 등록된 target 도시명·별칭·예시 장소가 있고 stale 증거가 없는 경우입니다. `no evidence`는 자동 실패 확정이 아니라 인간 검토 대기입니다.

## 턴테이킹 진단

| 조건 | repair 전 assistant 시작 | 표준 인사 이후 추가 발화도 repair 전 시작 |
|---|---:|---:|
| `immediate_repair` | 100.0% | 0.8% |
| `delayed_neutral` | 100.0% | 80.0% |
| `delayed_one_dependency` | 100.0% | 82.5% |
| `delayed_three_dependencies` | 100.0% | 89.2% |

모든 trial이 400ms에 인사를 시작했지만 사용자 audio는 480ms에 시작합니다. 따라서 모델은 매번 사용자를 듣기 전에 speaking state에 들어갔습니다. 단순 `assistant_started_before_repair` 조건 차이는 ceiling 때문에 0%p가 되며, 이를 인과 gate 통과로 해석하면 안 됩니다. 표준 인사 이후의 추가 발화도 delayed 조건에서 80.0–89.2%로 높아 dependency/latency와 turn-taking이 분리되지 않습니다.

## 지연 조건 대비

| 대비 | 지표 | 차이(%p) | scenario bootstrap 95% CI(%p) |
|---|---|---:|---:|
| `three_minus_neutral` | `conservative_target_only_proxy` | -0.8 | [-3.3, 1.7] |
| `three_minus_neutral` | `stale_evidence_proxy` | -4.2 | [-15.0, 5.8] |
| `one_minus_neutral` | `conservative_target_only_proxy` | -1.7 | [-4.2, 0.0] |
| `one_minus_neutral` | `stale_evidence_proxy` | -7.5 | [-16.7, 0.8] |

## 해석 제한

- seed 17 하나뿐이므로 생성 seed 변동을 추정할 수 없습니다.
- 자동 매칭은 D1–D3 관계가 새 도시에 실제로 재결합됐는지 판정하지 못합니다.
- primary window 이전에 나온 발화는 최종 endpoint에서 제외했습니다.
- 정식 결론에는 primary-only 미디어를 사용한 2명 독립 주석과 불일치 조정, 나머지 4개 seed가 필요합니다.
- 사람 audio/alignment 검수 기록이 없으므로 데이터 릴리스 상태는 provisional입니다.

## 현재 판단

기술 산출물은 완전하지만, 이 seed에서는 target 증거 사건이 너무 적어 floor effect가 있고 turn-taking 시작 상태가 조작과 얽혀 있습니다. 현 설정으로 나머지 2,400개를 바로 생성하기보다 assistant의 자동 선행 인사를 억제하거나 endpoint를 재설계한 소규모 비교 smoke를 먼저 수행해야 합니다.
