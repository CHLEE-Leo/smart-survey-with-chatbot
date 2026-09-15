# 패치 기록

## 2026-09-15 · PyTorch LLM 정책과 PPO 연구 코드

- 중첩된 기존 prototype 폴더를 제거하고 `code/src`의 단순 모듈과 train/evaluate 진입점으로 교체.
- E의 7개 action, 재질문, 공유 LLM + LoRA + policy/value head, GAE와 PPO 구현.
- 공개 근거 판정·종료와 비공개 reference 보상을 분리하고 자연어 user simulator 연결.
- CPU smoke 설정, 실제 작은 PEFT 모델 테스트, checkpoint 저장·재개, JSONL 실험 기록 추가.
- FastAPI와 기존 정적 화면을 같은 InterviewSession에 연결.
- 아래 7월 기록은 교체 전 구조의 변경 이력이며 현재 실행 방법은 `code/README.md`를 따름.

## 2026-07-18 · Baseline 및 RL-ready 모델 스켈레톤

- 기존 순차 E→F→G 엔진을 `code/baseline/engine.py`로 분리
- deploy 화면에서 `Rule-based baseline`과 `RL-ready chatbot` 선택 지원
- `models/dialog_model`: 누적 상태, grounded question candidate, 동적 후보 생성 담당
- `models/policy_model`: 후보 목록을 입력받는 policy protocol과 random policy 구현
- `models/reward_model`: 충분성 judge 계약, offline 규칙 fallback, confidence-gain reward 구현
- `models/user_model`: user simulator protocol과 재현 가능한 structured simulator 구현
- `models/env_model`: policy–simulator 대화 rollout 조립
- RL-ready 엔진에서 E–G 전체 미응답 문항이 매 턴 상호 이동 가능한 후보 공간을 구성
- `stop` action 및 고정 action mask를 사용하지 않음
- judge 충분성, 최대 턴, 후보 소진을 policy 외부 종료 조건으로 사용
- 각 턴에 후보 공간, 선택 질문, 상태, judge 전후 결과, reward를 기록
- 현재 judge와 simulator는 외부 LLM 없이 검증 가능한 fallback이며 향후 LLM 구현으로 교체 가능

## 2026-07-18 · Gemma LLM runtime 및 실행 스크립트

- 외부 Python 패키지 없이 OpenAI-compatible `/chat/completions`를 호출하는 공통 runtime 추가
- User/Reward 역할별 환경변수 설정과 기본 `gemma4:e4b` 모델 구성 추가
- private scenario에 제한된 구조화 응답을 생성하는 `GemmaUserSimulator` 추가
- 초진 문진 정보 충분성만 평가하는 `GemmaSufficiencyJudge`와 JSON 출력 검증 추가
- Reward LLM 연결·출력 실패 시 rule-based judge fallback 및 오류 메타데이터 기록
- deploy에서 rule-based/Gemma sufficiency judge 선택 지원
- `/api/config`, `/api/llm-health` 설정·상태 확인 API 추가
- `./start.sh` 실행 시 Gemma endpoint 확인 후 deploy 서버 실행
- 실제 브라우저 사용자는 User LLM을 사용하지 않으며 User LLM은 자동 rollout 전용
