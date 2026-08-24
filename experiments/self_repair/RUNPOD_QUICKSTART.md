# RunPod 영어 자기수정 실험 빠른 실행

## 1. Pod 만들기

- CUDA가 포함된 PyTorch 템플릿
- Python 3.10 이상
- GPU VRAM 최소 24GB, 처음에는 48GB GPU 권장
- `/workspace` persistent volume 50GB 이상 권장
- 웹 서버 포트 설정은 필요 없음

## 2. Moshi와 실험 저장소 받기

RunPod Web Terminal에서 이 저장소를 바로 clone한다. Moshi 원본 코드와 실험
스크립트, 영어 신경망 TTS 18개가 모두 포함되어 있어 별도 ZIP 업로드가 필요 없다.

```bash
cd /workspace
git clone https://github.com/mingunkim123/Moshi_self_revision.git moshi
cd /workspace/moshi
chmod +x experiments/self_repair/runpod/*.sh
```

RunPod 스크립트는 호출된 스크립트의 위치에서 저장소 루트를 자동으로 찾는다.
따라서 저장소 폴더 이름이 `moshi`가 아니어도 이전 `/workspace/moshi`를 잘못
사용하지 않는다. 필요할 때만 `MOSHI_REPO` 환경변수로 경로를 명시적으로 덮어쓴다.

## 3. 설치 및 GPU 확인

```bash
nvidia-smi
experiments/self_repair/runpod/setup.sh
```

설치 스크립트는 `/workspace/moshi/.venv`에 PyTorch Moshi를 설치하고,
모델 캐시는 `/workspace/hf-cache`에 둔다.

## 4. 3개 smoke run

```bash
cd /workspace/moshi
experiments/self_repair/runpod/run_english_smoke.sh
```

첫 실행에서는 `kyutai/moshiko-pytorch-bf16` 체크포인트를 내려받기 때문에
시간이 더 걸린다. 완료되면 다음 세 파일을 RunPod 파일 브라우저에서 내려받아
실제 Moshi 음성이 들어 있는지 듣는다.

```text
experiments/self_repair/results_en/raw/EN01__E1/seed_17/response.wav
experiments/self_repair/results_en/raw/EN01__E3/seed_17/response.wav
experiments/self_repair/results_en/raw/EN01__E8/seed_17/response.wav
```

E1은 clean Seoul, E3은 Busan에서 Seoul로 교체, E8은 Busan에서
Busan+Seoul로 확장하는 조건이다.

영어 자극은 사용자 발화가 끝난 뒤 40초를 더 입력해 긴 두 도시 응답도
수집한다. smoke 명령을 다시 실행하면 이 세 결과는 새 관찰창으로 덮어쓴다.

## 5. 전체 90개 실행

세 smoke 응답이 정상이라면 실행한다.

```bash
cd /workspace/moshi
experiments/self_repair/runpod/run_english_pilot.sh
```

구성은 2 voices × 9 conditions × 5 seeds = 90 outputs다. 실행이 끊겨도
같은 명령을 다시 실행하면 완성된 `result.json`은 건너뛴다.

## 6. 결과 확인 및 다운로드

```bash
cat /workspace/moshi/experiments/self_repair/results_en/metrics.auto.md

cd /workspace/moshi
zip -r /workspace/moshi_self_repair_results.zip \
  experiments/self_repair/results_en \
  experiments/self_repair/annotations \
  experiments/self_repair/data/manifest.en.prepared.csv
```

RunPod 파일 브라우저에서 `/workspace/moshi_self_repair_results.zip`을 내려받는다.

`metrics.auto.md`는 Moshi의 inner-text를 이용한 1차 결과다. 최종 결과는
`annotations/audio_en/`의 블라인드 응답 WAV를 사람이 듣고
`annotations/annotations.en.csv`에 입력한 뒤 산출한다. `annotation_key.en.csv`는
평가자에게 주지 않는다.

## 7. 재현성 고정

첫 전체 실행의 `results_en/run_metadata.json`에서 `resolved_revision`을 확인한다.
논문용 최종 실행 전 그 값을 `config/experiment.en.json`의 `revision`에 복사하면
체크포인트 버전이 고정된다. 모델은 명시적으로 남성 음성 Moshiko
`kyutai/moshiko-pytorch-bf16`를 사용한다.
