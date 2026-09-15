# 7-action LLM 문진 정책 연구 코드

## 1. 구현된 흐름

```text
A–D 공개 프로필 → D 순서대로 식품 선택
  → 실제 대화 원문 → frozen LLM + policy LoRA → 7-action head / value head
  → 선택한 E 유형의 skeleton + 동일 기반 LLM → 질문
  → LLM user simulator 또는 실제 사용자 → 응답
  → 공개 근거 추출 + 고정 연구 분류 규칙 → 계속 / 종료
  → 종료 후 같은 기반 LLM으로 의사용 요약 생성·검토 → 마지막 발화
  → 학습에서는 비공개 reference와 비교 → terminal reward → PPO
```

- action은 `0…6`으로 E-1…E-7과 일대일 대응합니다. 같은 action을 다시 선택할 수 있습니다.
- 정책은 profile·현재 식품·실제 누적 대화·질문 이력을 읽습니다. 수작업 feature encoder인 `state.py`는 없습니다.
- 기존 `lm_head`를 유지하고, `hidden_size → 7` policy head와 `hidden_size → 1` value head를 추가합니다.
- 양자화 기반 가중치는 고정하고 LoRA와 두 head를 학습합니다. value는 기대 보상이며 진단 확률이 아닙니다.
- 첫 구현은 질문 생성·질문 검토·근거 추출·종료 요약에서 policy adapter를 끕니다. 같은 기반 모델을 공유합니다.
- 정책·생성·판정은 시뮬레이터의 비공개 이력이나 정답을 입력받지 않습니다.
- 정답을 맞혔는지로 종료를 결정하지 않습니다. 잘못 판단하고 종료한 episode도 그대로 끝내고 실패 보상을 줍니다.

## 2. 파일 구조

```text
code/
├── train.py                 # rollout → GAE → PPO → checkpoint
├── evaluate.py              # learned / sequential / random 비교
├── configs/
│   ├── default.yaml         # 실제 LLM + QLoRA + HTTP simulator
│   └── smoke.yaml           # 다운로드 없는 CPU 실행 확인
├── src/
│   ├── model.py             # shared LLM, LoRA, policy/value heads, tokenization
│   ├── ppo.py               # Transition, GAE, rollout, PPO update
│   ├── questionnaire.py     # E-1…E-7, 증상 18종, 시간 구간
│   ├── dialogue.py          # skeleton, 자유 질문 생성, 범위 검토
│   ├── assessment.py        # 공개 근거와 내부 판정·종료 기준
│   ├── summary.py           # 의료진용 종료 요약·근거 검토·마지막 발화
│   ├── user_simulator.py    # scenario 로딩, LLM / scripted 사용자
│   ├── reward.py            # 종료 후 reference 비교와 턴 비용
│   ├── environment.py       # InterviewSession, FoodInterviewEnv
│   ├── llm.py               # 텍스트 생성 인터페이스, HTTP client
│   └── experiment.py        # YAML, 조립, seed, checkpoint, 실험 기록
├── api/main.py              # 같은 세션 엔진을 사용하는 FastAPI
├── data/demo_scenarios.jsonl # 합성 예제, 성능 평가용 데이터셋이 아님
├── tests/                   # 문진 계약, 실제 PEFT, PPO, API 검증
└── requirements.txt
```

기존 `baseline/`, `config/`, `dialogue/`, `models/`의 중첩 프로토타입은 제거했습니다.
`code/` 자체는 Python 패키지로 만들지 않습니다. Python 표준 라이브러리 `code`와 충돌하지 않게 하기 위함입니다.

## 3. 설치

Python 3.10 이상인 별도 가상환경을 권장합니다. 저장소 루트에서 실행합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
```

GPU 실험은 드라이버에 맞는 PyTorch/CUDA와 bitsandbytes가 필요합니다.
이 구현의 4-bit 경로는 CUDA 한 장을 사용합니다. 분산 학습·CPU offload·float16 GradScaler는 구현하지 않았으며,
학습 dtype은 `bfloat16` 또는 `float32`를 사용합니다.

## 4. 먼저 CPU에서 실행 확인

```bash
python code/train.py --config code/configs/smoke.yaml
python code/evaluate.py --checkpoint code/outputs/smoke/last.pt --output code/outputs/smoke/test.jsonl
python code/evaluate.py --config code/configs/smoke.yaml --policy sequential --output code/outputs/smoke/sequential.jsonl
python -m unittest discover -s code/tests -v
```

출력 경로가 이미 있으면 새 경로를 지정하거나 명시적으로 재개합니다.

```bash
python code/train.py --config code/configs/smoke.yaml --output-dir /tmp/survey-run-2
python code/train.py --resume code/outputs/smoke/last.pt --updates 3
```

`--updates`는 재개 전 횟수를 포함한 총 업데이트 수입니다.
`--resume`과 `--checkpoint`를 쓰면 저장된 설정을 사용하며 `--config`로 모델을 교체하지 않습니다.
데이터·기본 출력 경로는 YAML 위치나 현재 작업 폴더와 무관하게 `code/` 기준으로 해석합니다.
CLI의 명시적 출력 경로는 현재 작업 폴더 기준입니다.

**smoke 모드의 의미:** byte 입력을 평균 집계한 작은 무작위 Transformer와 low-rank adapter로
PPO·gradient·저장 경로를 확인합니다. 사전학습 LLM이나 QLoRA가 아닙니다.
`ScriptedUserSimulator`는 action별 고정 답변을, `DemoAssessor`는 `발현 시간: 20분` 같은 명시적인 예제 태그만 처리합니다.
실제 보호자의 자유응답 이해나 연구 성능을 이 모드의 수치로 주장할 수 없습니다.

## 5. 실제 LLM 학습

```bash
python code/train.py --config code/configs/default.yaml
python code/evaluate.py --checkpoint code/outputs/llm/last.pt --output code/outputs/llm/test.jsonl
```

`default.yaml`의 정책 예제는 `Qwen/Qwen2.5-3B-Instruct`입니다.
현재 설치된 Transformers의 일반적인 causal LM 경로로 구현을 시작하기 위한 예제이며, 최종 연구 모델 선정은 아닙니다.
[공식 모델의 로딩 예제](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)를 기준으로 구성했습니다.
LoRA는 텍스트 decoder의 `q_proj`, `v_proj`에 연결하며, 모델을 바꾸면 실제 모듈 이름을 확인해야 합니다.
[PEFT의 양자화 학습 경로](https://huggingface.co/docs/peft/developer_guides/quantization)를 사용합니다.

Gemma 4를 시도하려면 모델 설정에서 다음 필드를 바꿉니다.

```yaml
name: google/gemma-4-E4B-it
auto_class: AutoModelForMultimodalLM
use_processor: true
```

해당 클래스가 있는 Transformers 버전과 모델별 PEFT 호환성이 필요합니다.
공식 카드의 [Gemma 4 로딩 경로](https://huggingface.co/google/gemma-4-E4B-it)를 연결하도록 구성했지만,
Gemma 4의 text hidden-state 반환과 실제 4-bit 학습은 아직 실행 검증하지 않았습니다.
현재 환경의 Transformers 4.57.6에는 이 클래스가 없어 실행 시 명시적인 오류를 냅니다.

사용자 시뮬레이터는 별도의 실행 중인 chat-completions 서버를 사용합니다.

```yaml
simulator:
  backend: http
  base_url: http://localhost:11435/v1
  model: gemma4:e4b
  api_key_env: SURVEY_SIMULATOR_API_KEY
  temperature: 0.0
  timeout: 90.0
  max_tokens: 256
```

API 키가 필요한 서버에서만 지정한 환경변수를 설정합니다. 키를 YAML이나 checkpoint에 저장하지 않습니다.
HTTP 서버 태그와 학습용 Hugging Face 가중치 ID는 별개입니다.
실험 전 `data`를 검토된 시나리오로 교체하고, 모델 `revision`과 simulator 서버 모델 버전을 고정하세요.
기본 `main`으로 로드한 정책은 확인 가능한 기반 모델 commit을 checkpoint 설정에 기록합니다.
HTTP 서버의 실제 가중치 버전은 이 코드가 검증하지 않습니다.

## 6. 시나리오와 판정 기준

JSONL 한 줄은 환자 시나리오 하나입니다. 예제에는 4 train / 1 validation / 3 test 시나리오가 있습니다.
한 시나리오에 여러 식품을 넣을 수 있으며, 같은 `group_id`는 split을 넘을 수 없습니다.

| 필드 | 접근 경로 |
|---|---|
| `public_profile` | 정책·질문 생성·판정·simulator |
| `private_history[food]`, `persona` | simulator만 |
| `reference[food].label` | 종료 뒤 보상만 |
| `reference[food].required_evidence` | 종료 뒤 독립적인 최소 근거 비교 |
| `demo_answers[food][action]` | smoke의 고정 응답만 |

실제 LLM simulator는 action index로 답을 고르지 않고 **실제 생성된 질문**을 읽습니다.
reference는 simulator에게도 전달하지 않습니다. 보호자가 실제로 아는 진단이면 `private_history`에 별도로 기록합니다.
예제 태그와 reference는 코드를 실행하기 위한 수작업 합성 자료이며 전문가 라벨이 아닙니다.

첫 target은 `study-2015-operational-v1`입니다. 임상적 알레르기 유무를 판정하는 모델이 아닙니다.

- `meets_criteria`: 명확한 전형적 증상, 4시간 미만 발현, 반복 또는 미재섭취, 현재 회피가 모두 관찰됨.
- `not_meets_criteria`: 4시간 이상 발현, 현재 문제없는 섭취, 증상 없음, 재섭취 시 비반복 중 명확한 제외 근거가 관찰됨.
- `undetermined`: 근거 부족, 모호한 증상, “가끔”, 해소되지 않은 상충 등.

2015년 원문의 미확정 코드북을 보수적으로 운영화한 초기 규칙입니다. “모호한 증상”의 의미 분류와
같은 사건인지 확인하는 작업은 LLM에 의존하며 임상 타당성이 검증되지 않았습니다.
4시간 기준을 벗어났다고 다른 유형의 식품알레르기까지 없다고 해석하지 않습니다.

근거 인용의 발화 ID와 실제 인용 문자열을 검사합니다. 인용이 있다는 것만으로
의학적 해석·동일 사건 연결·충분성이 정확하다고 보장되지는 않습니다.
Simulator의 사실 일탈을 자동으로 탐지하는 별도 검증기는 아직 없습니다.

## 7. 보상·PPO·재현성

기본 보상은 매 턴 `-0.02`, 종료 시 근거와 정답이 맞으면 `+1`, 잘못 충분하다고 종료하면 `-1`,
턴 한도에서 미완료면 `-0.2`입니다. 같은 action을 반복했다는 이유의 별도 벌점은 없습니다.
`turn_cost: 0`이면 terminal-only 보상입니다. 보상 숫자와 근거 계약은 연구용 초기값입니다.

수집한 실제 토큰·action·old log probability·value·reward·종료 상태로 GAE를 계산합니다.
업데이트 때 같은 토큰을 다시 forward하므로 LoRA까지 gradient가 전달됩니다.
dropout은 rollout과 update 모두 끄고, gradient는 update에서 활성화합니다.
수집 구간 절단은 bootstrap하되 실제 종료는 bootstrap하지 않습니다.
현재 collector는 식품 episode 전체를 수집하며, 식품별 최대 턴은 실제 과제 종료로 처리합니다.

문맥은 조용히 잘라내지 않습니다. `max_input_tokens`를 넘으면 오류로 기록하고 중단합니다.
JSON 계약·네트워크 오류도 정상 환자 응답으로 대체하지 않습니다.
질문 검토에서 범위 위반이 확인되면 해당 skeleton으로 대체하고 로그를 남깁니다.

| 산출물 | 내용 |
|---|---|
| `run.json` | 설정, source/data hash, 패키지 버전, 학습 파라미터 수 |
| `rollouts.jsonl` | 실제 질문·응답·판정·재질문·fallback·시간·보상·종료 요약과 인용 |
| `metrics.jsonl` | PPO loss, entropy, KL, gradient norm, episode 결과 |
| `errors.jsonl` | 실패한 실행의 scenario·식품·오류; 오류 발생 시 학습 중단 |
| `last.pt` | LoRA·두 head·optimizer·update·설정·RNG·버전; tiny는 고정 무작위 기반도 저장 |
| 평가 JSONL + `.summary.json` | 식품별 결과와 세션 단위 성공률, 정확도, 거짓 충분성, 오류 수 |

평가 오류는 평균에서 조용히 제외하지 않습니다. label accuracy는 보류 레이블도 포함하므로
성공률·거짓 충분성·미완료와 함께 봐야 합니다. 최종 성능에는 별도 holdout·여러 seed·전문가 검증이 필요합니다.

## 8. 웹 연결

```bash
./start.sh
# 학습한 checkpoint로 자유응답 문진을 실행할 때:
./start.sh --checkpoint code/outputs/llm/last.pt --policy learned
```

`http://127.0.0.1:8765`에서 A–D → 프로필 확인 → E 식품별 문진을 사용합니다.
기본 실행은 순차 질문과 smoke 판정으로 **화면 경로만 확인**합니다.
원문 A–C 일부는 자유서술로 묶은 축약 입력이며 PSEN-FA는 포함하지 않습니다.
프런트엔드는 기존 정적 HTML/JS를 새 엔진에 연결했습니다. Next.js 전환은 이번 구현에 포함하지 않았습니다.

세션은 메모리에만 보관하고 서버 재시작 시 사라집니다. 로컬 테스트 서버이며 로그인·DB·운영 배포는 구현하지 않았습니다.
API는 모델 호출을 순서대로 실행합니다. simulator나 PPO는 웹 요청에 들어가지 않습니다.
답변에는 `question_id`를 포함해 이전 질문에 대한 중복 제출을 막습니다.

### 종료 요약

식품별 종료 판정 후 `InterviewSession.finish()`가 같은 기반 LLM으로 2~3문장, 350자 이내의
의료진용 요약을 생성합니다. 관찰된 핵심 병력과 잠정적 해석, 중요한 미확인·상충을 구분합니다.
시간 한도에서 끝났으면 판단 보류·추가 확인 필요를 명시하도록 지시합니다.
설문 분류 미충족을 “알레르기 없음”으로 바꾸지 않습니다.

요약의 발화 ID·인용 원문을 검사하고, 고정 기반 LLM으로 원문 대비 의미·확진 표현·중요 누락을 검토합니다.
이 검토는 전문가의 임상적 검증을 대신하지 않습니다. 잘못된 출력은 완료 처리하지 않고 오류로 남깁니다.
마지막 사용자 답변과 종료 판단은 보존하므로 `/api/session` 조회로 요약만 재시도할 수 있습니다.
성공한 요약은 캐시해 재조회·재시도로 중복 생성하거나 원문에 반복 추가하지 않습니다.

모든 식품이 끝나면 저장된 식품별 요약을 D 순서로 묶고 종료 인사를 붙여 **마지막 챗봇 발화**로 표시합니다.
각 식품의 요약은 해당 식품의 원문 전체와 공개 프로필로 생성합니다. 식품 간 종합 추론을 별도로 수행하지는 않습니다.
의심 식품을 하나도 선택하지 않았으면 LLM으로 병력을 만들어내지 않고 기록 없음으로 종료합니다.

요약은 8번째 action이나 추가 사용자 턴이 아닙니다. PPO 보상은 기존의 공개 판정과 비공개 reference를 비교하며,
요약 문장의 확신도·유창함을 성공 근거로 삼지 않습니다. 요약문은 종료 후에만 기록되어 다음 action의 입력으로 쓰이지 않습니다.
`generation.summary_max_tokens` 기본값은 768이며, 이 키가 없는 기존 checkpoint 설정에도 적용됩니다.
`summary_version`을 실험 메타데이터에 별도로 기록합니다. Smoke 모드는 LLM 대신 실제 답변 인용을 조합합니다.

## 9. 검증 범위

2026-09-15 확인:

- CPU smoke PPO 학습, 평가, checkpoint 저장·재로드·재개.
- 3회 연속 학습과 2회 학습 후 1회 재개의 모델·optimizer tensor가 정확히 일치.
- 작은 무작위 Hugging Face GPT-2 + 실제 PEFT LoRA로 gradient와 frozen base 불변성 확인.
- 로컬에 생성한 작은 Qwen2 가중치와 tokenizer로 실제 from_pretrained·LoRA·7/1 head 경로 확인.
- 같은 기반 모델의 생성 경로에서 adapter가 꺼지고 예외 뒤 복원되는지 확인.
- 정답 변경이 종료 경로를 바꾸지 않는지, 재질문·복수 정보·보류·시간 경계·GAE·checkpoint·API 검사.
- 종료 요약의 비공개 정보 차단, 인용 검사, 중복 방지, 검토 실패·API 재시도 확인.
- JavaScript와 shell 문법 확인.

관련 테스트 27개가 통과했습니다. 검증 환경은 Python 3.9.13, PyTorch 2.5.1, Transformers 4.57.6, PEFT 0.16.0입니다.
기존 환경의 오래된 `typing_extensions` 때문에 검증용 최신 패키지를 `/tmp`에 분리해 사용했습니다.
FastAPI TestClient는 샌드박스 밖의 로컬 실행으로 검증했습니다. GPU는 사용할 수 없었습니다.

**미검증:** 실제 사전학습 가중치·HTTP simulator를 연결한 전체 문진, CUDA 4-bit QLoRA,
Gemma 4 호환성, GPU 메모리, 브라우저 시각·사용성, 실제 LLM 종료 요약의 임상적 품질, 의료적 정확도와 연구 성능.
