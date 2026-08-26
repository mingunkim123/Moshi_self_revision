# TTS provider 검토 기록

검토일: 2026-08-26

## 확인된 사실

- `edge-tts==7.2.8` live inventory에서 현재 설정한 미국 영어 음성 10개가 모두 존재함을 확인했다.
- `edge-tts`는 Azure 구독 키를 사용하는 공식 Azure Speech SDK가 아니라 Microsoft Edge의 온라인 음성 서비스를 호출한다.
- Microsoft의 Azure TTS 문서는 prebuilt voice 결과를 파일로 저장할 수 있다고 설명하지만, Azure 구독 약관과 책임 있는 사용 정책이 적용된다.
- 현재 환경에는 Azure Speech key와 region 설정이 없다.
- 기존 Edge 샘플은 마지막 word boundary 이후 약 355–391 ms의 tail silence가 있어, `utterance_end_ms`와 WAV duration을 같은 값으로 취급하면 안 된다.
- 동일 설정 반복 합성이 서로 다른 후보를 만든다는 보장이 없으므로 3회 반복 production 전에 waveform/hash 다양성 검사가 필요하다.
- Frozen 600-target SSML을 Azure 문서의 과금 문자 규칙으로 계산하면 1회씩 606,900자,
  초기 3후보 정책은 1,820,700자, 총 5후보 hard maximum은 3,034,500자다. SSML markup을
  포함한 값이며, 실제 금액은 호출 직전 Azure portal의 region·계약별 rate로 계산한다.

## 운영 결론

`edge-tts`는 voice availability와 비공개 timing calibration에만 사용한다. 공개 release용 core 600개의 provider로는 승인하지 않는다.

## Kokoro 전환 및 실측

사용자 지시에 따라 현재 우선 production 후보를 `Kokoro-82M v1.0` 로컬 합성으로
전환했다. 모델 revision은 `f3ff3571791e39611d31c381e3a41a3af07b4987`, weight SHA-256은
`496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`로 고정했다.
Apache-2.0 모델 카드, config, 미국 영어 5F/5M voice 파일 10개의 전체 SHA-256을 config와
preflight가 전수 검증한다.

동일한 93단어 dependency-heavy 대본으로 10개 voice를 실제 로컬 CPU 합성했다.
10/10이 mono PCM16 24 kHz, provider-seed token mapping, 무클리핑 자동 QC를 통과했다.
발화 길이는 29.70–53.35초이고 `af_nicole`이 느린 outlier이므로 production voice freeze는
아직 하지 않았다. 두 명의 독립 청취자가 발음·누락·artifact·자연성·속도를 blind review한
뒤에만 600-target 합성을 허용한다. 상세 수치는 `reports/kokoro_voice_calibration.json`에
있다.

Kokoro 경로의 production 계약은 다음과 같다.

1. 82M model/config/voice 파일과 Python package version을 모두 고정한다.
2. 결정적 설정에서는 target당 후보를 반복 생성하지 않고 1개만 만든다.
3. retry가 필요하면 condition 하나의 속도만 바꾸지 않고 matched bundle 전체를 새 policy로 재생성한다.
4. model predicted token duration은 extraction seed일 뿐이며 RunPod/Linux MFA 독립 정렬로 교체한다.
5. Apache notice, 모델 카드의 CC-BY attribution, synthetic training provenance 검토를 release evidence에 묶는다.

Azure S0는 fallback으로 유지한다. Azure를 다시 선택할 경우에만 다음 절차를 적용한다.

1. paid Azure Speech S0의 공식 SDK로 24 kHz mono PCM을 직접 생성한다.
2. SDK WordBoundary와 SSML bookmark는 extraction seed로만 사용한다.
3. RunPod/Linux의 Montreal Forced Aligner로 독립 정렬하고 저신뢰 repair 경계를 수동 확인한다.
4. 공개 재배포와 Moshi 평가·개선 용도가 Microsoft 약관에 맞는지는 release 전에 법무 또는 기관 담당자가 확인한다.

macOS `say` 음성은 macOS Tahoe SLA의 공개 녹음·배포 제한 때문에 release core에 사용하지 않는다.

## 근거

- Microsoft Azure Product Terms: https://www.microsoft.com/licensing/terms/en-US/productoffering/MicrosoftAzureServices/OL/
- Microsoft TTS transparency note: https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/speech-service/text-to-speech/transparency-note
- Microsoft TTS data/privacy/security: https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/speech-service/text-to-speech/data-privacy-security
- Microsoft TTS billable-character 설명: https://learn.microsoft.com/en-us/azure/ai-services/Speech-Service/text-to-speech#pricing-note
- Azure Speech pricing: https://azure.microsoft.com/en-us/pricing/details/speech/
- Microsoft Services Agreement: https://www.microsoft.com/en-us/servicesagreement
- Apple macOS Tahoe SLA: https://www.apple.com/legal/sla/docs/macOSTahoe.pdf
- Montreal Forced Aligner: https://montreal-forced-aligner.readthedocs.io/en/latest/
- Kokoro-82M model card: https://huggingface.co/hexgrad/Kokoro-82M
- Kokoro voice inventory: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

이 문서는 법률 자문이 아니라 release gate의 보수적 운영 기록이다.
