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

`edge-tts`는 voice availability와 비공개 timing calibration에만 사용한다. 공개 release용 core 600개의 provider로는 아직 승인하지 않는다. 현재 권장 production 경로는 다음과 같다.

1. paid Azure Speech S0의 공식 SDK로 24 kHz mono PCM을 직접 생성한다.
2. SDK WordBoundary와 SSML bookmark는 extraction seed로만 사용한다.
3. RunPod/Linux의 Montreal Forced Aligner로 독립 정렬하고 저신뢰 repair 경계를 수동 확인한다.
4. 공개 재배포와 Moshi 평가·개선 용도가 Microsoft 약관에 맞는지는 release 전에 법무 또는 기관 담당자가 확인한다.

무자격증 대안은 Apache-2.0 Kokoro와 MFA 조합이지만, voice 품질과 학습 데이터 provenance를 별도 smoke/review해야 한다. macOS `say` 음성은 macOS Tahoe SLA의 공개 녹음·배포 제한 때문에 release core에 사용하지 않는다.

## 근거

- Microsoft Azure Product Terms: https://www.microsoft.com/licensing/terms/en-US/productoffering/MicrosoftAzureServices/OL/
- Microsoft TTS transparency note: https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/speech-service/text-to-speech/transparency-note
- Microsoft TTS data/privacy/security: https://learn.microsoft.com/en-us/azure/ai-foundry/responsible-ai/speech-service/text-to-speech/data-privacy-security
- Microsoft TTS billable-character 설명: https://learn.microsoft.com/en-us/azure/ai-services/Speech-Service/text-to-speech#pricing-note
- Azure Speech pricing: https://azure.microsoft.com/en-us/pricing/details/speech/
- Microsoft Services Agreement: https://www.microsoft.com/en-us/servicesagreement
- Apple macOS Tahoe SLA: https://www.apple.com/legal/sla/docs/macOSTahoe.pdf
- Montreal Forced Aligner: https://montreal-forced-aligner.readthedocs.io/en/latest/

이 문서는 법률 자문이 아니라 release gate의 보수적 운영 기록이다.
