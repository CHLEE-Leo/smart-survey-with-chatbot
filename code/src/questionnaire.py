"""E-1…E-7 source topics; natural-language skeletons are adaptations of the PDF."""

from dataclasses import dataclass

QUESTIONNAIRE_VERSION = "food-history-e-v1"
SYMPTOMS = (
    "두드러기", "가려움", "피부발진", "습진", "얼굴 부종", "눈 부종", "입술 부종",
    "구토", "설사", "복통", "기침", "콧물", "쌕쌕거림", "호흡곤란", "청색증",
    "혈압저하", "의식저하", "기타",
)
LATENCY_OPTIONS = ("30분 미만", "30분 이상–2시간 미만", "2시간 이상–4시간 미만", "4시간 이상")


@dataclass(frozen=True)
class QuestionType:
    action_id: int
    source_item: str
    name: str
    skeleton: str
    scope: str
    options: tuple = ()


QUESTIONS = (
    QuestionType(0, "E-1", "exposure_context", "{food}을(를) 어떤 형태로, 어디에서 먹었나요?",
                 "현재 식품의 종류, 조리·섭취 형태와 장소"),
    QuestionType(1, "E-2", "symptoms", "{food}을(를) 먹은 뒤 어떤 증상이 있었나요?",
                 "같은 섭취 사건에서 발생한 증상과 그 의미", SYMPTOMS),
    QuestionType(2, "E-3", "latency", "{food}을(를) 먹고 증상이 시작되기까지 얼마나 걸렸나요?",
                 "섭취부터 증상 시작까지의 시간; 모호하면 기억하는 시간 기준점을 확인", LATENCY_OPTIONS),
    QuestionType(3, "E-4", "repeatability", "이후 {food}을(를) 다시 먹었을 때도 같은 반응이 있었나요?",
                 "과거 재섭취 여부와 증상 반복성; 다시 먹어보라고 권하지 않음",
                 ("반복적으로 나타남", "가끔 나타남", "다시 먹지 않아 모름")),
    QuestionType(4, "E-5", "first_onset", "{food}에 대한 반응이 처음 나타난 나이는 언제인가요?",
                 "처음 반응한 당시의 나이 또는 개월 수"),
    QuestionType(5, "E-6", "current_intake", "현재는 {food}을(를) 어떻게 먹거나 제한하고 있나요?",
                 "현재 회피인지 실제로 문제없이 섭취하는지; 조리형태·시점의 차이",
                 ("현재도 제한 중", "문제없이 섭취 중")),
    QuestionType(6, "E-7", "tests", "{food}에 대해 병원에서 어떤 알레르기 검사를 받았나요?",
                 "검사 유무와 종류; 검사를 받았다는 사실을 양성 결과로 해석하지 않음",
                 ("검사받은 적 없음", "경구유발검사", "혈액검사", "피부반응검사")),
)
NUM_ACTIONS = len(QUESTIONS)


def get_question(action_id: int) -> QuestionType:
    if type(action_id) is not int or not 0 <= action_id < NUM_ACTIONS:
        raise ValueError("action_id must be an integer in [0, 6]")
    return QUESTIONS[action_id]


def latency_bin(minutes: float) -> str:
    """Half-open source bins; unknown observations must stay unknown."""
    import math
    if not math.isfinite(minutes) or minutes < 0:
        raise ValueError("latency must be a finite nonnegative number")
    if minutes < 30:
        return "lt30"
    if minutes < 120:
        return "30to120"
    if minutes < 240:
        return "120to240"
    return "ge240"
