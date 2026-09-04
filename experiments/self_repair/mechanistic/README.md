# Mechanistic stale-binding harness

이 디렉터리는 `MECHANISTIC_STALE_BINDING_RUNPOD.md`의 `[TARGET]` 명령을 구현한다. 목적은
Moshiko의 main temporal transformer에서 이전 값이 남는 위치와 수정값이 dependency/readout으로
전파되는 경로를 관찰·개입하는 것이다.

## 증거 수준

- `--synthetic`은 코드·통계·resume·패키징 계약만 검사한다. Moshiko 또는 Boston/Seattle에 관한
  실험 증거가 아니다.
- 실제 실행은 고정 revision과 검증된 WAV를 사용하고 `NO_TORCH_COMPILE=1`,
  `NO_CUDA_GRAPH=1`을 모델 생성 전에 설정해야 한다.
- 기존 v2는 exploratory/internal validation 전용이다. 정식 confirmation은 독립 검수된 다중도시
  데이터와 frozen selection이 모두 있을 때만 열린다.

## 로컬 검증

```bash
python3.12 -m venv .venv-mechanistic
.venv-mechanistic/bin/pip install -r experiments/self_repair/requirements-mechanistic.txt
.venv-mechanistic/bin/pip install -e ./moshi --no-deps
.venv-mechanistic/bin/pytest -q experiments/self_repair/mechanistic/tests
experiments/self_repair/mechanistic/runpod/run_local_validation.sh
```

## RunPod smoke

```bash
export NO_TORCH_COMPILE=1
export NO_CUDA_GRAPH=1
export MECH_DATA_ROOT=/workspace/moshi/experiments/self_repair/dataset_v2
export MECH_RUN_ROOT=/workspace/mech-artifacts/<identity-derived-run-id>
experiments/self_repair/mechanistic/runpod/runpod_smoke.sh
```

실제 스캔 순서, gate와 해석 제한은 상위
`experiments/self_repair/MECHANISTIC_STALE_BINDING_RUNPOD.md`를 따른다.
