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
- 모델 실행 계약: `kyutai/moshiko-pytorch-bf16` revision
  `2bfc9ae6e89079a5cc7ed2a68436010d91a3d289`
- 환경은 `requirements-mechanistic.txt`에서 새로 생성했고 `pip check`가 성공했다.

## 자동 테스트

```text
NO_TORCH_COMPILE=1 NO_CUDA_GRAPH=1 \
  .venv-mechanistic/bin/pytest -q \
  experiments/self_repair/mechanistic/tests moshi/tests

15 passed

.venv/bin/pytest -q experiments/self_repair/tests/dataset_v2

83 passed
```

총 98개 테스트가 통과했다. 검증 범위는 다음과 같다.

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

22개 `[TARGET]` CLI 모두 `--help` parsing과 직접 실행 진입점을 확인했다.

## 실제 v2 자산을 이용한 CPU/dry-run 검증

기존 provisional source를 outcome과 무관하게 seed 17로 제한해 다음 결과를 확인했다.

```text
portable mechanistic trials: 600
semantic anchors: 3,960
frame trace rows: 259,341
synthetic encoded manifests: 600
open-loop contract: passed
WAV basename rebind/hash failures: 0
```

Portable manifest는 과거 `/Users/...` URI를 사용하지 않고 현재 data root 아래 상대 URI와 실제
WAV SHA-256으로 다시 결합했다. 이 데이터는 여전히 `exploratory_provisional`이다.

## Synthetic pipeline 및 package

- residual discovery cells: 6
- frozen selection 이후 local-validation cells: 2
- analyzer/report/verifier: passed
- public/private tar 분리와 재개봉 검증: passed
- synthetic 보고서에 “not empirical evidence” 표시: passed
- public archive의 WAV, tensor, private/credential 이름 및 credential-like content 제외: passed

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
- full-duplex 출력은 별도 블라인드 annotation이 완료되기 전 `awaiting_intervention_blind_annotation`
  상태로만 기록된다.
