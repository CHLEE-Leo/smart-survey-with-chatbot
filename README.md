# Smart survey with an AI chatbot

A–D에서 프로필을 설정하고, E의 7개 질문 유형을 선택·재질문하는 PyTorch 연구 코드입니다.
LLM 정책과 PPO, 질문 생성, 사용자 시뮬레이터, 공개 근거 판정과 보상을 분리했습니다.
종료 후에는 같은 기반 LLM이 의료진 검토용 병력 요약을 마지막 발화로 전달합니다.

- [코드 구조·설치·학습·평가](code/README.md)
- [현재 아키텍처와 검증 범위](docs/research_architecture.md)
- [연구 내용](docs/research_content.md)
- [논문·설문지 분석](docs/questionnaire_analysis.md)

설치 후 저장소 루트에서 모델 다운로드 없이 학습 경로를 확인합니다.

```bash
python code/train.py --config code/configs/smoke.yaml
python code/evaluate.py --checkpoint code/outputs/smoke/last.pt --output code/outputs/smoke/test.jsonl
```

`smoke`는 작은 무작위 모델과 고정 응답으로 코드 경로를 확인하는 모드입니다.
실제 LLM 학습은 [default.yaml](code/configs/default.yaml)을 사용합니다.
실제 모델 가중치 다운로드·GPU 4-bit 학습·임상 성능은 아직 검증하지 않았습니다.

로컬 화면 테스트는 `./start.sh`로 실행합니다. FastAPI가 같은 문진 엔진을 사용하며,
기본 화면 테스트 모드에는 자유응답을 이해하는 LLM이 연결되어 있지 않습니다.
