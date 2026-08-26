# Dataset v2 decision log

## 2026-08-26 provider audit

- `edge-tts==7.2.8`은 voice inventory와 비공개 timing calibration에만 사용한다.
- Edge 검토 당시 paid Azure Speech S0를 우선안으로 두었으나, 이후 사용자 지시에 따라
  Kokoro local TTS + MFA를 현재 우선안으로 전환했다. Azure는 fallback이다.
- 공개 재배포와 Moshi 평가 용도에 대한 기관 승인 전에는 production release를 시작하지 않는다.
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
  `9a6775f1477210b57d4d01ff35f88fa23cff9387e5798aa92797e9b33785de7d`이다. 후자는
  storage execution plan을 config에 동결한 뒤 재생성한 hash이며 transcript는 변하지 않았다.

## 2026-08-26 Kokoro local calibration

- `kokoro==0.9.4`, model revision `f3ff3571791e39611d31c381e3a41a3af07b4987`, weight
  SHA-256 `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`를 고정했다.
- 미국 영어 5F/5M voice 10개를 같은 93단어 script로 실제 합성했고 10/10이 24 kHz PCM16,
  token seed mapping, clipping/tail QC를 통과했다. 발화 길이는 29.70–53.35초다.
- `af_nicole`은 가장 빠른 voice보다 약 1.80배 느려 자동 제외하지 않고 blind human review에
  넘겼다. `reports/kokoro_voice_calibration.json`은 technical pass이지만 human review pending이다.
- 결정적 합성 반복은 독립 후보가 아니므로 production source는 target당 1개만 생성한다.
  실패 시 조건 하나를 보정하지 않고 frozen retry policy로 matched bundle 전체를 재생성한다.
- 최초 10-voice 단일 프로세스는 aggregate manifest flush 전에 종료됐다. immutable audio와
  boundary 10쌍을 hash 검증해 private manifest를 recovery했고 이후 candidate별 checkpoint와
  `--resume`을 구현했다.

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

- Preflight는 provider 호출을 하지 않고 300 scripts/600 targets/source-track/voice join과
  Kokoro runtime/model/config/voice hash를 전수 검증한다.
- 현재 matrix는 target당 deterministic 1후보, 총 600 requests이며 provider API 비용과
  credential은 없다. Azure character budget은 fallback 경로 진단으로만 유지한다.
- `production_authority.json`은 Apache/attribution/training provenance, Moshi 평가 목적,
  공개 여부, human text 및 voice double-listen, RunPod MFA upload, artifact store,
  로컬 12GiB·원격 40GiB 최소 공간과 승인자를 묶는다.
  이 파일과 preflight report는 private `release_evidence/`에 두며 Git에 커밋하지 않는다.
- 사용자는 2026-08-26 `local audio production + RunPod MFA/Moshi evaluation` 구성을
  승인했다. Kokoro 1-candidate 정책의 로컬 audio 작업량은 약 4GiB, 보수적 상한은
  약 6GiB이며 현재 약 24GiB 여유 공간으로 12GiB reserve gate를 통과한다. BF16 model cache와 약 10GiB response
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
