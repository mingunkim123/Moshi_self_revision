# Dataset v2 decision log

## 2026-08-26 provider audit

- `edge-tts==7.2.8`은 voice inventory와 비공개 timing calibration에만 사용한다.
- 공개 core audio는 paid Azure Speech S0 + 공식 SDK + MFA 독립 정렬을 우선안으로 둔다.
- provider 비용·자격증명, 공개 재배포와 Moshi 평가 용도에 대한 기관 승인 전에는 production release를 시작하지 않는다.
- 세부 근거와 대안은 `TTS_PROVIDER_REVIEW.md`에 기록한다.

## 2026-08-26 private Edge calibration

- 최종 텍스트에서 shortest/median/longest 3개 scenario를 고르고 두 engineering voice로
  60 rendition targets × 3 attempts = 180 candidates를 합성했다.
- provider boundary mapping은 180/180 성공했다. 자동 QC는 158/180을 통과했고 Roger
  voice 후보 22개는 미세한 full-scale sample이 있어 clipping으로 제외했다.
- delayed 조건의 중앙 80% 구간은 두 voice 사이에 공통 교집합이 없었다. private
  provisional latency는 Guy 11,768.75±1,231.25 ms, Roger
  15,293.75±618.75 ms였다. 따라서 설명용 4,000 ms를 production target으로 사용하지
  않고, 승인된 provider에서 화자별 target/tolerance를 다시 정한다.
- 이 결과는 Edge 음성을 release 대상으로 승인하지 않으며, 독립 forced alignment를
  대신하지 않는다. 상세 수치와 실패 목록은
  `reports/edge_private_timing_calibration.json`에 있다.
- 독립 품질 검토 후 blueprint 30개와 generated script 300개가 모두 구조·스키마 gate를
  통과했다. 현재 canonical file SHA-256은 blueprint
  `e7d3b18077f6f1a0f348b5217c5d32710465c9167ff220bef197b17800827057`, script
  `ab5f42e423c1ae89fc28607d6993ca74b7d721039984398227e19007889a34f8`이다. 후자는
  storage execution plan을 config에 동결한 뒤 재생성한 hash이며 transcript는 변하지 않았다.

## 2026-08-26 evaluation/release execution contract

- Production matrix는 accepted audio 600개 × ordered generation seed 5개 = 3,000 trials로
  고정한다. 599×5, 600×4, seed/config 불일치는 실행 전에 실패시킨다.
- Prepared stimulus는 artifact-root-relative URI, SHA-256, canonical preparation hash와 전체
  shifted timing을 trial identity에 포함한다. 다른 host에서는 artifact root만 다시 지정한다.
- Standard Moshi(`model_type=moshi`, 24 kHz, Mimi frame 1,920 samples, max LM delay 1)만
  허용한다. 사용자가 말한 뒤 `response_capture_ms`까지 digital zero를 넣고 fed frame 수와
  output token/audio frame 수가 정확히 같아야 completed로 인정한다. 조기 EOS는 실패다.
- 모델 revision과 code commit은 full 40-hex SHA여야 하고, runtime의 Hugging Face snapshot,
  clean tracked tree, runner source hash가 동결값과 같아야 한다.
- Full release는 selection/timing/alignment/audio QC/double-listen/analysis/baseline evidence와
  외부 approval이 모든 입력 hash에 묶인 경우에만 생성한다. Text-development snapshot은
  공개 audio release가 아니며 별도 상태로 표시한다.
- 실제 checkpoint에서 위 timebase/coverage 가정을 확인하는 1-trial GPU smoke는 아직
  수행하지 않았다. 이 smoke는 3,000-trial production run의 선행 gate다.

## 2026-08-26 production preflight contract

- Preflight는 provider 호출을 하지 않고 300 scripts/600 targets/source-track/voice join을
  전수 검증한다. Azure credential은 존재 여부만 보고하고 값을 저장하지 않는다.
- 현재 SSML 기준 billable characters는 target당 1후보 합계 606,900자, 초기 3후보
  1,820,700자, 총 5후보 상한 3,034,500자다. 승인 시점의 portal rate를 private authority에
  기록해 초기 정책 예상액이 budget cap 안에 있을 때만 통과한다.
- `production_authority.json`은 paid tier, Moshi 평가 용도, 공개 여부와 재배포 검토,
  human text sign-off, RunPod MFA upload, artifact store, 로컬 12GiB·원격 40GiB 최소 공간,
  승인자를 모두 묶는다.
  이 파일과 preflight report는 private `release_evidence/`에 두며 Git에 커밋하지 않는다.
- 사용자는 2026-08-26 `local audio production + RunPod MFA/Moshi evaluation` 구성을
  승인했다. 로컬 초기 audio 작업량은 약 8GiB, 후보 총 5개 상한은 약 12GiB이며 현재
  약 25GiB 여유 공간으로 local gate를 통과한다. BF16 model cache와 약 10GiB response
  audio는 최소 40GiB RunPod workspace에 유지하고 결과 metadata만 로컬로 회수한다.

## Frozen engineering defaults

- Purpose: pretrained Moshi controlled evaluation, not a general training corpus.
- Language/domain: English travel planning.
- Root/value pair: `destination`, `Boston ↔ Seattle`.
- Design: 30 scenarios × 2 directions × 5 conditions × 2 speakers.
- Primary text unit: 60 text bundles and 300 generated scripts.
- Primary audio unit: 120 matched audio bundles and 600 accepted renditions per source track.
- Core inference: all 30 scenarios are confirmatory after protocol and manifest hashes freeze.
- Existing E1–E9 data remains the pipeline-development fixture and is not part of dataset v2.
- `planning_frame_id` is an assigned counterbalancing family shared by each D/N pair. It
  records pragmatic/syntactic family membership; it does not promise that every row contains
  the identifier's mnemonic verb literally.

## Gates not yet frozen

The following require empirical evidence or external authority before production audio can be
called release-ready:

1. TTS voice availability and redistribution rights.
2. Natural timing overlap, target latency, and within-speaker tolerance.
3. Independent alignment method and confidence threshold.
4. Candidate-selection weights and tie-break rule.
5. RunPod workspace/response retention path and release license.
6. Human-speaker consent, compensation, recruitment, and retention policy.
7. Production-pilot ICC 또는 baseline이 현재 power simulation 가정과 크게 다를 때의
   MDE 재계산. 현재 conditional proxy와 primary formula는 `ANALYSIS_PROTOCOL.md`에 동결했다.

Until these gates are closed, generated text and engineering smoke audio are development artifacts,
not a public dataset release.
