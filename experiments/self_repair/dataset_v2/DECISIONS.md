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

## 2026-08-26 Kokoro production raw run

- 사용자 진행 지시에 따라 provisional private 범위에서 600 rendition target을 로컬 CPU로
  모두 합성했다. Azure/API credential/provider 과금은 사용하지 않았다.
- 8GiB Mac에서 2-process 병렬은 memory compression으로 느려져 중단하고, 서로 겹치지 않는
  300개 shard 두 개를 단일 process로 순차 실행했다. 완료 후보는 hash checkpoint로 재사용했다.
- merge validator가 target/source/speaker/voice/candidate ID와 600개 WAV SHA-256을 전수
  확인했다. canonical 변환과 provider timing seed mapping 뒤 자동 QC는 600/600 통과했다.
- 총 canonical 음성은 5.756시간, raw+canonical은 1.853GiB다. 조건별 120, 화자별 60,
  방향별 300으로 균형이며 상세 hash는 `reports/kokoro_production_audio.json`에 기록한다.
- Kokoro predicted token duration은 독립 alignment가 아니다. MFA와 human double-listen이
  완료될 때까지 accepted/prepared는 0으로 유지하고 release eligible로 표시하지 않는다.
- Canonical WAV hard link와 exact frozen `.lab` transcript 600쌍을 10개 voice별 speaker
  디렉터리로 준비했다. local hard link라 추가 오디오 block을 복제하지 않는다.

## 2026-08-26 local MFA independent alignment

- macOS arm64에서 conda-forge의 마지막 지원 빌드인 MFA 2.2.4를 로컬 설치했다. 실행 호환을
  위해 `joblib=1.4.2`, `setuptools=80.9.0`을 고정했다. Micromamba 2.9.0 binary SHA-256은
  `ec2a072f028e1a7cf20f3e2e74d5a8127cf5a5f27636375b5359811565f4e5be`다.
- 긴 저장소 경로는 PostgreSQL Unix socket 103-byte 제한을 넘으므로 실제
  `MFA_ROOT_DIR`은 짧은 `/tmp/mfa22root`를 사용했다. 원본 audio/model은 저장소의 ignored
  artifact 경로에 유지했다.
- `english_us_arpa` acoustic SHA-256은
  `d35ce271ded357d833d2f4b8d1041dc3748b9538567ba13f2c697f4e4126711b`, base dictionary는
  `e8c6c7b036ae2b7c78d2768b8dc6b1f9359175b842956d00b48c53c9c332e6b0`, G2P model은
  `0ef6a3b288dc0a91a267f98bf556a45bbfeae198e578398e92ac603a61ad46e5`다.
- 첫 flat corpus가 10개 voice를 1 speaker로 인식한 결과는 폐기했다. speaker별 하위
  디렉터리로 재구성한 최종 실행은 10 speakers × 60 files를 인식하고 fMLLR을 10개
  speaker별로 계산했다.
- 기본 사전의 OOV 20개 단어형은 같은 ARPA G2P로 생성한 frozen 37-pronunciation extension
  `config/mfa_english_us_arpa_oov.dict`로 보강했다. 합친 dictionary SHA-256은
  `d20e13a9714650a9189ea1ec716840540826dbd4602b69eade99f2ba13837ab0`이며 최종 OOV는 0이다.
- 최종 TextGrid 600/600을 frozen transcript lexical stream과 전수 대조하고 import했다.
  하이픈/구두점 tokenization 차이가 있던 540개는 문자열 동일성을 확인해 frozen token
  경계로만 재분절했다. 새 timing으로 자동 QC를 다시 실행해 600/600 통과했다.
- MFA의 log likelihood와 phone-duration deviation은 calibrated probability가 아니므로
  confidence를 0으로 명시하고 600개 모두 hash-bound human review pending으로 유지했다.
  따라서 독립 정렬은 완료됐지만 accepted/prepared는 여전히 0이다. 상세 evidence는
  `reports/mfa_local_alignment.json`과 `reports/kokoro_production_audio.json`에 있다.

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
  공개 여부, human text 및 voice double-listen, independent MFA environment, artifact store,
  로컬 12GiB·원격 40GiB 최소 공간과 승인자를 묶는다.
  이 파일과 preflight report는 private `release_evidence/`에 두며 Git에 커밋하지 않는다.
- 사용자는 2026-08-26 처음 `local audio production + RunPod MFA/Moshi evaluation` 구성을
  승인했다. 이후 MFA는 로컬에서 완료해 현재 구성은 `local audio/MFA + RunPod Moshi`다.
  실제 로컬 working artifact는 4.8GiB이고 현재 약 29GiB 여유 공간으로 12GiB reserve
  gate를 통과한다. BF16 model cache와 response audio는 최소 40GiB RunPod workspace에
  유지하고 결과 metadata만 로컬로 회수한다.

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
