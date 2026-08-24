# Moshi 한국어 동일 턴 자기수정 파일럿

이 디렉터리는 다음 질문을 재현 가능하게 검증한다.

> 사용자가 “부산 날씨… 아니, 서울 날씨 알려줘”라고 말했을 때 Moshi가 철회된 부산이 아니라 최종 수정값인 서울을 따르는가?

현재 공개 Moshi는 영어 중심 모델이고 날씨 도구가 없다. 따라서 실제 날씨의 사실성은 평가하지 않는다. 주 결과는 **최종 도시 선택**, **철회된 도시 오류**, **조기 응답 후 회복**이다. 한국어 clean 조건이 실패하면 self-repair 실패로 해석하지 않고 한국어 OOD 실패로 기록한다.

## 권장 주 실험: 영어 E1–E9

공개 Moshi의 언어와 맞는 유효한 주 실험은 영어 세트다. 현재 다음 파일이 생성되어 있다.

- `data/raw_en/EN01/E1.wav`부터 `EN02/E9.wav`: Ava와 Andrew 신경망 영어 음성 18개
- `data/recordings.en.csv`: 자동 입력된 marker/repair/user-end 타임스탬프
- `data/tts_english_set.zip`: RunPod 업로드용 전체 묶음
- `config/experiment.en.json`: 영어 실험 설정

E3/E4는 자연스러운 `uh, sorry, I meant ...` 수정과 200 ms pause를, E5/E6는 **동일한 연속 발화 파형**에 800 ms pause를 사용한다. E8은 `Busan → actually, could you check both Busan and Seoul`의 확장 수정이고, E9는 문두 `Actually`가 수정이 아닌 담화표지 통제다. 단어별 조각 합성이 아니라 문장 전체를 한 번에 생성하므로 억양·템포·리듬이 이어진다.

영어 세트는 온라인 신경망 TTS로 다시 만들 수 있다. 고정된 실험 문장이 Microsoft 음성 서비스로 전송되므로 민감한 문장에는 사용하지 않는다. macOS에서는 `afconvert`, Linux에서는 `ffmpeg`가 필요하다.

```bash
python3 -m venv .venv-tts
source .venv-tts/bin/activate
pip install -r experiments/self_repair/requirements-tts.txt
python experiments/self_repair/scripts/synthesize_english_neural_tts.py --overwrite
```

오프라인 재생성이 필요할 때만 기존 `synthesize_english_tts.py`의 macOS 음성을 fallback으로 사용한다.

RunPod에서는 먼저 3개 출력(E1 clean, E3 교체 수정, E8 동시요청 확장)만 만드는 smoke run을 실행한다.

```bash
experiments/self_repair/runpod/setup.sh
experiments/self_repair/runpod/run_english_smoke.sh
```

세 응답 WAV가 정상적으로 들리면 영어 실험 전체(18 stimuli × 5 seeds = 90 outputs)를 실행한다. smoke run의 세 결과는 재사용되며, 중단 후 같은 명령을 다시 실행해도 완료된 결과는 건너뛴다.

```bash
experiments/self_repair/runpod/run_english_pilot.sh
```

추론이 끝나면 `results_en/metrics.auto.md`에 inner-text 기반 1차 결과가 바로 생성된다. Seoul/Busan 중 하나만 명확히 나온 응답은 자동 분류하고, 둘 다 나오거나 둘 다 없는 응답은 `AUTO_HEURISTIC_REVIEW`로 표시한다. 최종 보고에는 표시된 행을 사람이 확인한다.

## 디렉터리

- `config/experiment.json`: 체크포인트, seed, sampling, 오디오 설정
- `data/conditions.csv`: 한국어 본 조건, resolution probe, 영어 positive control
- `data/raw/`: 원본 녹음. Git에 포함되지 않는다.
- `data/prepared/`: 24 kHz mono, 정규화·무음·frame padding이 끝난 자극
- `results/`: 응답 WAV, inner-text/token trace, 실행 메타데이터와 지표
- `annotations/`: 블라인드 수동 라벨링 시트
- `scripts/`: manifest 생성, 전처리, 추론, 라벨링 시트, 통계 집계
- `runpod/`: RunPod 설치와 end-to-end 실행 스크립트

## 1. RunPod 설치

저장소가 `/workspace/moshi`에 있다고 가정한다. 다른 경로라면 `MOSHI_REPO`만 바꾼다.

```bash
cd /workspace/moshi
chmod +x experiments/self_repair/runpod/*.sh
experiments/self_repair/runpod/setup.sh
source /workspace/moshi/.venv/bin/activate
export HF_HOME=/workspace/hf-cache
```

PyTorch bf16 체크포인트를 사용하므로 CUDA GPU가 필요하다. 실험은 메모리 변동을 줄이기 위해 batch 1로 실행된다. `config/experiment.json`의 `hf_repo`는 반드시 고정한다. 현재 기본값은 `kyutai/moshiko-pytorch-bf16`이다.

첫 다운로드가 끝나면 `results/run_metadata.json`의 `resolved_revision`을 확인하고, 최종 실험 전 그 값을 `config/experiment.json`의 `revision`에 복사한다. 같은 HF repo의 `latest`가 바뀌는 문제를 막기 위해서다.

## 2. 녹음 manifest 생성

### 녹음 없이 TTS로 시작하기

macOS에서는 설치된 한국어 음성 Yuna와 Eddy를 이용해 S01/S02 × K1–K8의 16개 WAV를 바로 생성할 수 있다. K3/K4는 200 ms, K5/K6는 800 ms pause를 넣으며 대응 조건의 나머지 음성 조각은 동일하게 재사용한다.

```bash
cd /workspace/moshi
python experiments/self_repair/scripts/synthesize_macos_tts.py
```

이미 파일이 있으면 덮어쓰지 않는다. 다시 합성할 때만 `--overwrite`를 붙인다. 생성된 WAV와 타임스탬프는 `data/recordings.csv`에 자동 반영된다. RunPod는 Linux이므로 이 합성 단계는 현재 Mac에서 실행한 뒤 저장소의 `data/raw/`를 RunPod로 복사해야 한다.

현재 생성된 전체 세트는 `data/tts_smoke_set.zip`에도 묶여 있다. RunPod에 이 파일을 올린 뒤 `experiments/self_repair/`에서 압축을 풀면 된다.

TTS 결과는 pipeline/한국어 clean gate용 smoke test다. 자연스러운 인간 자기수정에 대한 논문 결과로 직접 일반화하지 않는다.

### 사람 음성을 수집하는 경우

두 화자 smoke test:

```bash
cd /workspace/moshi
source .venv/bin/activate

python experiments/self_repair/scripts/generate_recording_manifest.py \
  --speakers S01,S02 \
  --condition-set pilot \
  --languages ko \
  --tracks natural \
  --output experiments/self_repair/data/recordings.csv
```

본 파일럿은 `S01`부터 `S20`까지 가명 ID를 `data/speakers.txt`에 한 줄씩 적고 다음처럼 생성한다.

```bash
python experiments/self_repair/scripts/generate_recording_manifest.py \
  --speakers-file experiments/self_repair/data/speakers.txt \
  --condition-set pilot \
  --languages ko \
  --tracks natural \
  --output experiments/self_repair/data/recordings.csv
```

생성 후 각 녹음을 아래 경로에 둔다.

```text
data/raw/S01/K1.wav
data/raw/S01/K2.wav
...
data/raw/S20/K8.wav
```

녹음 순서는 화자마다 무작위화한다. K3/K4의 “아니” 전 pause는 150–300 ms, K5/K6는 약 800 ms로 유도한다. TTS는 파이프라인 확인에만 사용하고 본 분석에는 자연발화를 사용한다.

### 타임스탬프 입력

`data/recordings.csv`에 ms 단위로 다음 값을 입력한다.

- `repair_marker_onset_ms`: “아니” 시작
- `repair_onset_ms`: 최종 수정 도시 시작
- `repair_end_ms`: 최종 수정 도시 또는 repair span 끝
- `user_end_ms`: 사용자 발화 끝

K5/K6를 별도 녹음하지 않고 K3/K4에 무음을 삽입해 만들 수도 있다. 이때 K5/K6의 `raw_audio_path`를 대응하는 K3/K4 파일로 바꾸고 `insert_silence_at_ms`와 `insert_silence_ms`를 채운다. 예를 들어 원 pause가 200 ms라면 marker 직전에 600 ms를 추가해 총 800 ms로 만든다.

## 3. 자극 전처리

```bash
python experiments/self_repair/scripts/prepare_stimuli.py \
  --recordings experiments/self_repair/data/recordings.csv
```

전처리는 다음을 보장한다.

- 24 kHz mono
- active-speech RMS 목표 −23 dBFS, peak 최대 −1 dBFS
- 앞 무음 480 ms
- 응답 수집용 뒤 무음: 한국어 설정 8초, 영어 설정 40초. Moshi 오프라인 추론에는 의미적 EOS가 없으므로, 두 도시를 차례로 답하는 E7–E9와 장문 응답이 중간에 잘리지 않도록 영어 관찰창을 충분히 길게 둔다.
- 1,920 samples, 즉 Mimi 80 ms frame의 정수배
- prepared WAV SHA-256 및 조정된 타임스탬프 저장

출력은 `data/manifest.prepared.csv`다. 전체 GPU 실행 전에 입력만 검증한다.

```bash
python experiments/self_repair/scripts/run_eval.py --dry-run
```

## 4. 고정 seed 추론

기본 seed는 `17, 29, 42, 101, 2026`이다. 모델은 한 번만 로드하고 trial마다 Mimi/LM streaming state와 RNG를 초기화한다. 동일 seed가 clean/repaired 조건에 공통으로 적용된다.

```bash
python experiments/self_repair/scripts/run_eval.py
```

먼저 한 자극·한 seed만 확인하려면:

```bash
python experiments/self_repair/scripts/run_eval.py \
  --condition K3 \
  --speaker S01 \
  --seeds 17 \
  --limit 1
```

결과 구조:

```text
results/
  run_metadata.json
  predictions.jsonl
  raw/S01__K3/seed_17/
    response.wav
    result.json
```

`response.wav`가 평가 대상이다. `result.json`의 `inner_text`는 Moshi의 assistant-side inner-text이며 사용자 ASR이나 독립적인 응답 전사가 아니다. `text_tokens`의 시각은 stream 시작 기준 80 ms frame 근사치다.

중단 후 같은 명령을 실행하면 존재하는 `result.json`은 건너뛴다. 다시 생성해야 할 때만 `--overwrite`를 사용한다.

설치부터 annotation sheet 생성까지 한 번에 실행하려면:

```bash
experiments/self_repair/runpod/run_pilot.sh \
  experiments/self_repair/data/recordings.csv
```

## 5. 블라인드 라벨링

```bash
python experiments/self_repair/scripts/make_annotation_sheet.py
```

`annotations/annotations.csv`는 출력마다 A1/A2 두 행을 만든다. 응답은 무작위 순서의 `annotations/audio/B00001.wav` 형식으로 hard-link 또는 복사된다. 평가자에게는 이 CSV와 `annotations/audio/`만 전달한다.

`annotations/annotation_key.csv`는 blind ID와 실제 trial/condition을 연결하는 비공개 키다. 평가자에게 전달하지 않는다.

허용되는 `label`:

- `target_only`: 최종 유효 도시만 따름
- `stale_only`: 철회된 도시만 따름
- `both`: 두 도시를 모두 활성 요청으로 처리
- `recovered`: 철회된 도시에 먼저 반응했지만 최종 도시로 명시적으로 회복
- `clarification`: 도시나 요청을 되물음
- `irrelevant`: 관련 없는 응답
- `no_speech`: 분석 구간에 발화 없음
- `unintelligible`: 판정 불가능한 발화

Boolean 열은 `1` 또는 `0`으로 적는다. `final_target_correct`는 최신 도시를 최종 활성 요청으로 따른 경우 1이다. 실시간 날씨를 모른다고 답해도 서울 요청을 명시적으로 이해했다면 1로 코딩한다.

A1/A2가 다르면 동일 `blind_id`의 세 번째 행을 추가하고 `annotator_id=A3`, `adjudicator=1`로 최종 판정을 입력한다. adjudication이 없고 1:1로 갈린 출력은 집계에서 `unresolved`로 남는다.

## 6. 지표 계산

```bash
python experiments/self_repair/scripts/score_results.py
```

생성물:

- `results/metrics.json`: machine-readable 전체 결과
- `results/metrics.md`: 조건별 Target Selection, SIER, early-stale, recovery, CRG

seed는 독립 표본으로 세지 않는다. 먼저 화자×조건 내 seed를 평균하고, 화자를 cluster로 10,000회 bootstrap하여 95% CI를 계산한다.

사전 판정 기준:

- K1/K2 중 하나라도 80% 미만: 한국어 clean gate 실패. self-repair 결론 금지
- clean gate 통과 후 K3/K4의 CRG가 10 pp 이상이거나 SIER가 10% 이상: repair-specific 후속 학습 진행
- K3/K4 target rate 95% 이상, SIER 5% 이하, CRG 5 pp 이하: 예시가 너무 쉬움. nested/no-marker/intent-reversal로 확장
- K3/K4는 성공하고 K5/K6만 하락: 의미 선택보다 full-duplex 조기 commitment/recovery 문제로 분리

이 기준은 파일럿의 go/no-go 규칙이지 모집단에 대한 통계적 주장이나 성공 성능 기준이 아니다.

## 7. 별도 control 실행

한국어 resolution probe는 `KP1`–`KP4`, 영어 positive control은 `E1`–`E4`다. 본 파일럿과 다른 화자를 쓰는 경우 recording manifest, prepared manifest, results root를 별도로 둔다.

```bash
python experiments/self_repair/scripts/generate_recording_manifest.py \
  --speakers KP01,KP02 \
  --condition-set smoke \
  --languages ko \
  --tracks resolution \
  --output experiments/self_repair/data/recordings.ko_probe.csv
```

한국어 clean gate가 실패하고 영어 control만 성공하면 현재 공개 체크포인트의 언어 OOD 문제다. 두 언어가 모두 실패하면 녹음·전처리·응답 자르기·체크포인트 설정부터 점검한다.
