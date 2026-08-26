# Dataset card — draft before audio production

## 목적

이 세트는 streaming/full-duplex Moshi가 사용자의 목적지 수정 뒤 기존 목적지에 연결된 관계를
몇 개까지 다시 묶는지 평가한다. 다섯 조건은 같은 최종 의미를 공유하며, delayed 조건은 수정
전후 의미 단위 수를 각각 세 개로 고정한다.

## 범위

- 영어 여행 계획 발화
- root slot 하나: `destination`
- value pair 하나: `Boston ↔ Seattle`
- 30개 여행 맥락, 3개 destination-dependent relation, 3개 root-invariant constraint
- confirmatory evaluation 전용

Boston/Seattle은 city proper를 기본으로 하되 응답이 명시적으로 metropolitan transit area를
가리킬 때만 metro entity를 인정한다. 이 규칙은 특히 Boston 인근 sports venue 판정에 적용한다.

## 대표성 한계

- 단일 value pair 결과를 다른 도시·slot·언어에 일반화할 수 없다.
- 모든 D3가 lodging이고 모든 N3가 lodging budget이며 D2 다수가 dining이다. 이는 matched
  control을 강화하지만 relation type의 광범위한 일반화를 제한한다.
- 합성 음성 결과는 사람 발화의 억양, hesitation, accent 분포를 대표하지 않는다.
- 실제 응답이 도시명을 직접 말하지 않으면 entity 기반 사람 판정이 필요하며 unresolved 비율을
  함께 보고해야 한다.
- Full-duplex 모델이 repair 전 발화를 시작하는 비율이 조건별로 다르면 dependency-only 인과
  해석 gate가 실패한다.

## 안전과 품질

알레르기, 접근성, 저자극 환경 관련 문구는 추천 시스템의 안전 보증을 요구하지 않는다.
Annotation은 다음 원칙을 따른다.

- severe allergy venue를 “안전하다”고 단정한 일반 응답은 정답으로 인정하지 않는다.
- policy 공개, advance contact와 이용자 직접 확인을 보존해야 한다.
- generic `accessible` 또는 `quiet` 표현만으로 구체적 요구가 충족됐다고 판정하지 않는다.
- medical advice나 실제 예약 가능성을 dataset gold로 간주하지 않는다.

## 라이선스·출처 상태

Blueprint 문구는 외부 문장을 복사하지 않은 project-original structured draft다. Dataset release
license는 사용자/기관 승인을 기다리고 있다. TTS audio는 provider 약관과 공개 재배포 범위가
서면으로 확인된 source track만 배포한다. Human track은 별도 informed consent, 철회·삭제,
보상·보존 정책을 통과해야 한다.

현재 provisional TTS source는 Apache-2.0 `Kokoro-82M v1.0`이다. 모델 revision과
model/config/voice SHA-256을 고정했고 10개 voice technical QC는 통과했다. 다만 model card가
밝힌 CC-BY 및 closed-provider synthetic training data provenance를 attribution/기관 검토에
포함하고, 두 명의 독립 voice 청취가 끝나기 전에는 release source로 승인하지 않는다.

## 권장 보고

Condition별 target accuracy, stale-state error, relation coverage, early response, recovery,
actual latency와 post-cue duration을 모두 보고한다. 30개 scenario를 primary cluster로 사용하고
generation seed를 독립 표본으로 세지 않는다. Gate 실패나 missing annotation을 숨기지 않는다.
