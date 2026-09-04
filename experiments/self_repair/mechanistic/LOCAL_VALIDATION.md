# Mechanistic stale-binding harness: local validation

검증일: 2026-09-04 (Asia/Seoul)

## 결론

구현된 하네스의 로컬 계약 테스트와 synthetic end-to-end 실행은 통과했다. 이 기록은 코드와
재현성 장치의 검증이며, Moshiko checkpoint 또는 Boston/Seattle stale binding에 관한 실험
결과가 아니다.

## 검증 환경

- Python: 3.12.14
- PyTorch: 2.8.0
- NumPy: 2.2.6
- 시작 기준 commit: `55c6a4456a084fb4f836bbf6eab5797e8a8ee5b0`
- 검증 대상: `codex/mechanistic-stale-binding-harness`의 최종 handoff source tree
- 모델 실행 계약: `kyutai/moshiko-pytorch-bf16` revision
  `2bfc9ae6e89079a5cc7ed2a68436010d91a3d289`
- 환경은 `requirements-mechanistic.txt`에서 새로 생성했고 `pip check`가 성공했다.

## 자동 테스트

```text
NO_TORCH_COMPILE=1 NO_CUDA_GRAPH=1 \
  .venv-mechanistic/bin/pytest -q \
  experiments/self_repair/mechanistic/tests moshi/tests

217 passed

.venv/bin/pytest -q experiments/self_repair/tests/dataset_v2

83 passed
```

총 300개 테스트가 통과했다. 검증 범위는 다음과 같다.

- hook-off/identity hook 출력 동일성 및 layer/site/shape/dtype/device 계약
- 선택한 attention head만 변경되는 intervention seam
- KV ring wrap 전후 absolute frame mapping과 tensor-deep snapshot/restore
- LM open-loop feedback 분리, snapshot branch replay와 eager gradient path
- semantic anchor의 80 ms 변환과 LM input delay trace
- analytic toy patch의 기대 방향과 self-patch no-op
- atomic cell write, resume 중복 방지와 충돌 거부
- ridge probe, Holm 보정, public/private package 분리와 content secret 검사
- 다중도시 ordered-pair/scenario role 격리 및 사람 검수 fail-closed 동작
- synthetic discovery → freeze → internal validation → analysis → Markdown/SVG → verifier

32개 public CLI wrapper를 포함한 `scripts/*.py` 35개 모두 `--help` parsing과 Python compile을
통과했다. RunPod shell script 3개는 `bash -n`, 새 Python 3.12 환경은 `pip check`를 통과했다.

## 실제 v2 자산을 이용한 CPU/dry-run 검증

기존 provisional source를 outcome과 무관하게 seed 17로 제한해 다음 결과를 확인했다.

```text
portable mechanistic trials: 600
roles: discovery 360 + internal validation 240
semantic anchors: 3,960
prepared audio frames: 259,341
frame trace rows: 259,941 (audio_frame 259,341 + lm_prime 600)
target conversation frames: 557,835
appended zero frames: 298,494
synthetic encoded manifests: 600
open-loop contract: passed
WAV basename rebind/hash failures: 0
```

Portable manifest는 과거 `/Users/...` URI를 사용하지 않고 현재 data root 아래 상대 URI와 실제
WAV SHA-256으로 다시 결합했다. 이 데이터는 여전히 `exploratory_provisional`이다.
User/conversation/assistant-silence 세 code stream의 합은 1,375,011 frames이며, bounded repeat
encode 대상 2개는 byte-identical이었다. Open-loop 검증은 600 trials, 30 paired comparisons에서
feedback hash 동일성, candidate-order 불변성, reset 결정성, delay mapping과 identity no-op를
모두 확인했다.

Discovery `query_end` residual paid-scan spec의 모델-free 산술도 실제 600-row manifest에서
재계산했다.

```text
selected trials / repair recipients: 360 / 288
cells: 9,216
replay passes / frames: 27,648 / 25,772,640
readout frames: 1,179,648
generation 제외 total model frames: 26,952,288
generation 포함: 1,152 generations, 1,160,260 generated frames,
                  28,112,548 total model frames
```

## Synthetic pipeline 및 package

- residual discovery cells: 72
- frozen selection 이후 local-validation cells: 4
- total patch cells / scenario clusters: 76 / 4
- analyzer 완료, Markdown/SVG report 생성, verifier 통과
- artifact manifest 98 entries, 실제 run 파일 99개(manifest 자체 포함)
- 같은 directory 두 번째 실행은 verifier-only였고 artifact/resume-summary hash가 byte-identical
- public/private tar 분리, 재개봉, archive SHA-256 검증: passed
- synthetic 보고서에 “not empirical evidence” 표시: passed
- public archive의 WAV, tensor, private/credential 이름 및 credential-like content 제외: passed

Synthetic fixture의 inferential `passed` 값은 `false`였으며 이는 의도된 작은 analytic fixture의
통계 결과다. 파이프라인/검증 성공을 Boston–Seattle 효과 통과로 바꾸어 쓰지 않는다.

다중도시 power 도구는 현재 고정된 설계 가정에서 SESOI power `0.8378`을 기록했다. 이는 관측
효과가 아니라 데이터 생성 전 sensitivity 계산이다. 실제 city 후보는 모두 screening pending으로
남겨 두었으며, 현 config로 audio builder를 실행하면 종료 코드 2로 중단한다.

## GPU에서 아직 확인해야 할 항목

- 로컬 머신에는 NVIDIA GPU와 RunPod 연결 정보가 없어 7B Moshiko weight를 로드하지 않았다.
- RunPod에서 exact checkpoint의 32 layers × 32 heads × 4096 hidden 계약을 다시 확인해야 한다.
- 실제 Mimi encode 반복 동일성, bf16 tolerance, VRAM, cell/sec와 중단 후 GPU resume를 측정해야 한다.
- clean/readout capability와 repair gap gate가 통과하기 전 discovery scan을 해석하면 안 된다.
- formal confirmation은 독립 clean-recognition screen, 새 WAV, independent alignment, 두 사람의
  intervention-blind 청취와 불일치 조정 기록 없이는 실행되지 않는다.
- full-duplex 출력은 두 required startup mode에서 실제 greeting/user/complete-response를 모두
  캡처해야 하며, 별도 이중 블라인드 검수 전 `awaiting_double_blind_human_review` 상태로만 기록된다.
- 로컬 shell은 paid readiness 승인, 실제 full-duplex 대화, empirical probe/analysis 및 empirical
  release packaging을 실행하지 않았다. 이 경로는 unit/integration fixture로만 검증했다.
