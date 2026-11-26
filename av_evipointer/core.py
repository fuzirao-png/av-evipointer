from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Window:
    slot: int
    start: float
    end: float


@dataclass(frozen=True)
class Query:
    question: str
    video: Path | None
    duration: float
    condition: str = 'original'
    options: tuple[str, ...] = ()
    synthetic: tuple[tuple[int, str, str], ...] = ()


@dataclass(frozen=True)
class Evidence:
    slot: int
    start: float
    end: float
    object: str
    sound: str
    relation: str


@dataclass(frozen=True)
class Mode:
    visual: bool = True
    audio: bool = True
    interleave: bool = True
    visual_pointer: bool = False
    audio_pointer: bool = False
    verify: bool = False
    center_frame: bool = False
    whole_video: bool = False


MODES = {
    'center-frame': Mode(audio=False, center_frame=True),
    'vision-only': Mode(audio=False),
    'audio-only': Mode(visual=False),
    'concat-av': Mode(interleave=False),
    'native-av': Mode(whole_video=True),
    'uniform-av': Mode(),
    'verifier-only': Mode(verify=True),
    'pointer-only': Mode(visual_pointer=True, audio_pointer=True),
    'full': Mode(visual_pointer=True, audio_pointer=True, verify=True),
    'no-audio-pointer': Mode(visual_pointer=True, verify=True),
    'no-visual-pointer': Mode(audio_pointer=True, verify=True),
    'no-interleave': Mode(interleave=False, visual_pointer=True, audio_pointer=True, verify=True),
}


@dataclass(frozen=True)
class View:
    window: Window
    visual: bool = True
    audio: bool = True


@dataclass(frozen=True)
class Reply:
    data: dict
    input_tokens: int


class BudgetExhausted(Exception):
    """The next prompt would exceed the per-question input limit."""


class Backend(Protocol):
    def call(self, query: Query, stage: str, views: list[View], mode: Mode,
             info: dict, remaining: int) -> Reply: ...


def make_windows(duration: float, window_seconds: float) -> list[Window]:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be positive and finite')
    # Equal-length windows keep two selected positions comparable in media duration.
    count = math.ceil(duration / window_seconds)
    return [Window(i, i * duration / count, (i + 1) * duration / count)
            for i in range(count)]


def uniform_indices(size: int, count: int) -> list[int]:
    count = min(size, count)
    return [int((i + 0.5) * size / count) for i in range(count)]


def preference(raw: object, size: int) -> list[int]:
    if not isinstance(raw, list):
        return []
    return list(dict.fromkeys(i for i in raw if type(i) is int and 0 <= i < size))


def parse_evidence(raw: object, views: list[View]) -> tuple[list[Evidence], int]:
    if not isinstance(raw, list):
        return [], 0
    by_slot = {v.window.slot: v for v in views}
    parsed = []
    for item in raw:
        if not isinstance(item, dict) or type(item.get('slot')) is not int:
            continue
        view = by_slot.get(item['slot'])
        if view is None:
            continue
        start, end = item.get('start'), item.get('end')
        if type(start) not in (float, int) or type(end) not in (float, int):
            continue
        if not (view.window.start - 0.001 <= start < end <= view.window.end + 0.001):
            continue
        fields = [item.get(key, '') for key in ('object', 'sound', 'relation')]
        if not all(isinstance(value, str) for value in fields):
            continue
        obj, sound, relation = (value.strip() for value in fields)
        if not (obj or sound) or (obj and not view.visual) or (sound and not view.audio):
            continue
        parsed.append(Evidence(view.window.slot, float(start), float(end), obj, sound, relation))
    return parsed, len(raw)


def resolve_answer(raw: object, options: tuple[str, ...]) -> str:
    if not isinstance(raw, str):
        return ''
    answer = raw.strip()
    if not options:
        return answer
    option_letter = answer.upper()
    if len(option_letter) == 1 and 'A' <= option_letter <= chr(ord('A') + len(options) - 1):
        return options[ord(option_letter) - ord('A')]
    return next((option for option in options if option.casefold() == answer.casefold()), '')


def run_question(backend: Backend, query: Query, mode: Mode, *, window_seconds: float,
                 slots: int, budget: int, retries: int) -> dict:
    windows = make_windows(query.duration, window_seconds)
    uniform = uniform_indices(len(windows), slots)
    costs = []
    last = {'answer': None, 'evidence': [], 'pointer_count': 0, 'status': 'abstained'}

    def call(stage: str, views: list[View], info: dict) -> dict:
        reply = backend.call(query, stage, views, mode, info, budget - sum(costs))
        costs.append(reply.input_tokens)
        return reply.data

    def choose(order: list[int], attempt: int) -> list[int]:
        if not order:
            return uniform
        # Rank unselected uniform windows ahead of other positions as a fallback.
        pool = list(dict.fromkeys(order + uniform + list(range(len(windows)))))
        offset = (attempt * len(uniform)) % len(pool)
        return (pool[offset:] + pool[:offset])[:len(uniform)]

    try:
        visual_order, audio_order = [], []
        if mode.visual_pointer or mode.audio_pointer:
            overview = [View(Window(-1, 0, query.duration), mode.visual, mode.audio)]
            plan = call('plan', overview, {'windows': windows})
            if mode.visual_pointer:
                visual_order = preference(plan.get('visual'), len(windows))
            if mode.audio_pointer:
                audio_order = preference(plan.get('audio'), len(windows))

        tried = set()
        for attempt in range(retries + 1 if mode.verify else 1):
            if mode.center_frame:
                views = [View(Window(0, 0, query.duration), True, False)]
            elif mode.whole_video:
                views = [View(Window(-1, 0, query.duration), mode.visual, mode.audio)]
            else:
                visual = set(choose(visual_order, attempt)) if mode.visual else set()
                audio = set(choose(audio_order, attempt)) if mode.audio else set()
                views = [View(windows[i], i in visual, i in audio) for i in sorted(visual | audio)]
            signature = tuple(views)
            if signature in tried:
                continue
            tried.add(signature)
            output = call('answer', views, {})
            answer = resolve_answer(output.get('answer'), query.options)
            evidence, count = parse_evidence(output.get('evidence'), views)
            last.update(evidence=evidence, pointer_count=count)
            if not answer:
                continue
            if mode.verify:
                if not evidence or len(evidence) != count:
                    last['status'] = 'invalid_evidence'
                    continue
                slots_cited = {pointer.slot for pointer in evidence}
                selected = [view for view in views if view.window.slot in slots_cited]
                verified = call('verify', selected, {'answer': answer, 'evidence': evidence})
                if verified.get('supported') is not True:
                    last['status'] = 'unsupported'
                    continue
            last.update(answer=answer, status='answered')
            break
    except BudgetExhausted:
        last['status'] = 'budget_exhausted'
    return {**last, 'input_tokens': sum(costs), 'calls': len(costs)}
