"""Local web adapter; no simulator calls or PPO updates occur in HTTP requests."""

from __future__ import annotations

import argparse
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.environment import InterviewSession
from src.experiment import (CODE_ROOT, build_runtime, load_config, read_checkpoint,
                            restore_checkpoint, seed_everything)
from src.llm import LLMError


class Profile(BaseModel):
    model_config = ConfigDict(extra="allow")
    suspected_foods: list[str] = Field(max_length=30)

    @field_validator("suspected_foods")
    @classmethod
    def validate_foods(cls, foods):
        foods = [food.strip() for food in foods]
        if any(not food or len(food) > 100 for food in foods) or len(set(foods)) != len(foods):
            raise ValueError("의심 식품은 중복 없이 입력해 주세요.")
        return foods


class StartRequest(BaseModel):
    profile: Profile


class AnswerRequest(BaseModel):
    session_id: str
    question_id: str
    value: str = Field(min_length=1, max_length=4000)


class InterviewRun:
    def __init__(self, profile: dict, runtime, max_turns: int, policy: str):
        self.session_id = uuid.uuid4().hex
        self.runtime, self.policy = runtime, policy
        self.sessions = [InterviewSession(profile, food, runtime.questioner, runtime.assessor, max_turns)
                         for food in profile["suspected_foods"]]
        self.index = 0

    def payload(self) -> dict:
        while self.index < len(self.sessions) and self.sessions[self.index].termination:
            self.index += 1
        if self.index == len(self.sessions):
            # Show records and missing information, not internal hypothesis logs.
            summaries = [{"food_id": s.food_id, "termination": s.termination,
                          "missing_information": s.assessment.missing_evidence,
                          "messages": s.messages} for s in self.sessions]
            return {"session_id": self.session_id, "done": True, "summaries": summaries,
                    "message": "선택한 식품이 없습니다." if not summaries else "식품별 문진 기록을 확인해 주세요."}
        session = self.sessions[self.index]
        if session.pending_question is None:
            if self.policy == "learned":
                action = self.runtime.model.act(self.runtime.model.encode(session.observation()), True)[0]
            else:
                action = session.turn_count % 7
            session.ask(action)
        return {"session_id": self.session_id, "done": False, "section": "E",
                "question_id": f"{self.index}:{session.turn_count}",
                "message": session.pending_question.text,
                "question": {"type": "text"},
                "progress": {"food_id": session.food_id, "food_index": self.index + 1,
                             "food_count": len(self.sessions), "turns": session.turn_count}}

    def answer(self, question_id: str, text: str) -> dict:
        if self.index >= len(self.sessions):
            raise ValueError("이미 종료된 세션입니다.")
        if question_id != f"{self.index}:{self.sessions[self.index].turn_count}":
            raise ValueError("이미 처리된 답변입니다. 현재 질문을 다시 불러와 주세요.")
        self.sessions[self.index].answer(text)
        return self.payload()


def create_app(config: dict, runtime=None, policy: str = "sequential", checkpoint: dict = None) -> FastAPI:
    if policy not in ("sequential", "learned"):
        raise ValueError("Unknown web policy")
    if policy == "learned" and checkpoint is None:
        raise ValueError("The learned web policy requires a checkpoint")
    seed_everything(config["seed"])
    runtime = runtime or build_runtime(config)
    if checkpoint:
        restore_checkpoint(checkpoint, runtime.model)
    app = FastAPI(title="Food-history interview research")
    sessions, lock = {}, threading.RLock()

    @app.get("/api/config")
    def settings():
        return {"offline_demo": config["model"]["backend"] == "tiny", "policy": policy}

    @app.post("/api/start")
    def start(body: StartRequest):
        try:
            with lock:
                run = InterviewRun(body.profile.model_dump(), runtime, config["environment"]["max_turns"], policy)
                payload = run.payload()
                sessions[run.session_id] = run
                return payload
        except LLMError as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/session")
    def get_session(session_id: str):
        with lock:
            if session_id not in sessions:
                raise HTTPException(404, "세션을 찾을 수 없습니다.")
            try:
                return sessions[session_id].payload()
            except LLMError as exc:
                raise HTTPException(503, str(exc)) from exc

    @app.post("/api/answer")
    def answer(body: AnswerRequest):
        with lock:
            if body.session_id not in sessions:
                raise HTTPException(404, "세션을 찾을 수 없습니다.")
            try:
                return sessions[body.session_id].answer(body.question_id, body.value)
            except LLMError as exc:
                raise HTTPException(503, str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc

    static = CODE_ROOT.parent / "deploy_test/static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CODE_ROOT / "configs/smoke.yaml"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--policy", choices=("sequential", "learned"), default="sequential")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    state = read_checkpoint(args.checkpoint) if args.checkpoint else None
    config = state["config"] if state else load_config(args.config)
    app = create_app(config, policy=args.policy, checkpoint=state)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
