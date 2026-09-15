# 로컬 문진 화면

현재는 `code/api/main.py`의 FastAPI가 연구 코드와 같은 `InterviewSession`을 사용합니다.
기존 표준 라이브러리 서버와 E/F/G 후보 선택 엔진은 제거했습니다.
설치·학습·검증 방법은 [code/README.md](../code/README.md)를 참고하세요.

저장소 루트에서 실행합니다.

```bash
./start.sh
```

`http://127.0.0.1:8765`에서 A–D → 프로필 확인 → 식품별 E 문진을 확인합니다.
기본 실행은 순차 질문과 smoke 판정으로 화면 경로를 확인하는 모드입니다.
자유로운 사용자 답변을 이해하는 LLM이 연결되어 있지 않으며 화면에도 이를 표시합니다.
PSEN-FA는 이번 흐름에 포함하지 않습니다.

학습한 실제 LLM 정책으로 실행하려면:

```bash
./start.sh --checkpoint code/outputs/llm/last.pt --policy learned
```

기반 모델의 질문 생성·근거 추출이 같은 프로세스에서 실행됩니다.
웹 사용자가 직접 답하며 user simulator와 PPO는 요청 경로에서 실행하지 않습니다.
세션은 메모리에만 보관하고 서버 재시작 시 사라집니다.

`python deploy_test/app.py`도 같은 진입점입니다. `--host`, `--port`, `--config`를 지원합니다.
`/api/config`, `/api/start`, `/api/answer`, `/api/session`을 제공합니다.
답변에는 현재 `question_id`가 필요하며, LLM 오류 때 임의 응답이나 판정으로 대체하지 않습니다.

Frontend는 정적 HTML/JavaScript입니다. Next.js 전환과 브라우저 시각·사용성 검증은 아직 하지 않았습니다.
