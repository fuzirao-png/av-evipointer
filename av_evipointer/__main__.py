from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
import time
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .backends import SyntheticBackend, WeMMBackend
from .core import MODES, Evidence, Query, run_question


@dataclass(frozen=True)
class Example:
    question_id: str
    video_id: str
    split: str
    query: Query
    answer: str | None
    evidence: tuple[dict, ...] | None = None
    question_type: str | None = None


def label(value: str) -> str:
    normalized = unicodedata.normalize('NFKC', value).casefold().replace('_', ' ')
    return ' '.join(normalized.strip().rstrip('.!?。！？').split())


def demo_examples() -> list[Example]:
    question = 'Which color triangle is ringing?'
    examples = []
    cases = [
        ('late', 'original', 3, 'red'),
        ('early', 'original', 0, 'blue'),
        ('middle', 'original', 2, 'green'),
        ('early', 'shifted', 2, 'green'),
        ('early', 'muted', None, None),
        ('late', 'misaligned', 3, None),
    ]
    for video_id, condition, active, expected in cases:
        colors = ['blue', 'yellow', 'green', 'red']
        if condition == 'misaligned':
            colors[3] = ''
        events = tuple((i, color, 'chime' if i == active else '') for i, color in enumerate(colors))
        query = Query(question, None, 8.0, condition, synthetic=events)
        gold = () if expected is None else (
            {'start': active * 2.0, 'end': (active + 1) * 2.0,
             'object': expected, 'sound': 'chime', 'relation': 'sounds'},
        )
        examples.append(Example(f'{video_id}:{condition}', video_id, 'demo', query, expected, gold))
    return examples


def load_manifest(path: Path) -> list[Example]:
    examples, seen = [], set()
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            question_id = str(row['id'])
            if question_id in seen:
                raise ValueError(f'duplicate id {question_id!r}')
            media = (path.parent / row['video']).resolve(strict=True)
            if not media.is_file():
                raise ValueError('video must be a local file')
            condition = row.get('condition', 'original')
            if not isinstance(condition, str) or not condition.strip():
                raise ValueError('condition must be a non-empty string')
            question = row['question']
            answer = row['answer']
            options = row.get('options', [])
            if not isinstance(question, str) or not question.strip():
                raise ValueError('question must be a non-empty string')
            if answer is not None and (not isinstance(answer, str) or not label(answer)):
                raise ValueError('answer must be a non-empty string or null')
            if not isinstance(options, list) or len(options) > 26 or not all(
                isinstance(value, str) and label(value) for value in options
            ):
                raise ValueError('options must contain up to 26 non-empty strings')
            if options and answer is not None and label(answer) not in {label(opt) for opt in options}:
                raise ValueError('answer must be an option’s text when options are supplied')
            if type(row['duration']) not in (int, float):
                raise ValueError('duration must be a number')
            duration = float(row['duration'])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError('duration must be positive and finite')
            evidence = row.get('evidence')
            if evidence is not None and not isinstance(evidence, list):
                raise ValueError('evidence must be a list or null')
            for pointer in evidence or []:
                start, end = pointer['start'], pointer['end']
                if type(start) not in (float, int) or type(end) not in (float, int):
                    raise ValueError('evidence timestamps must be numbers')
                if not 0 <= start < end <= duration or not all(
                    isinstance(pointer.get(key, ''), str) for key in ('object', 'sound', 'relation')
                ):
                    raise ValueError('invalid reference evidence')
            query = Query(question, media, duration, condition, tuple(options))
            question_type = row.get('question_type')
            if isinstance(question_type, list) and all(isinstance(item, str) for item in question_type):
                question_type = '/'.join(question_type)
            if question_type is not None and not isinstance(question_type, str):
                raise ValueError('question_type must be a string or list of strings')
            examples.append(Example(question_id, str(row.get('video_id', row['video'])),
                                    str(row.get('split', 'test')), query, answer,
                                    tuple(evidence) if evidence is not None else None, question_type))
            seen.add(question_id)
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ValueError(f'{path.name}:{number}: {error}') from error
    if not examples:
        raise ValueError('manifest contains no examples')
    return examples


def matches(pointer: Evidence, gold: dict, threshold: float) -> bool:
    intersection = max(0, min(pointer.end, gold['end']) - max(pointer.start, gold['start']))
    union = max(pointer.end, gold['end']) - min(pointer.start, gold['start'])
    return intersection / union >= threshold and all(
        label(getattr(pointer, key)) == label(gold.get(key, ''))
        for key in ('object', 'sound', 'relation')
    )


def score(example: Example, prediction: dict, threshold: float) -> dict:
    answer = prediction['answer']
    gold = example.answer
    # Abstentions are errors on answerable items and are scored separately on probes.
    correct = bool(answer is not None and gold is not None and label(answer) == label(gold))
    pointers = prediction['evidence']
    supported = None
    if example.evidence is not None:
        supported = sum(any(matches(pointer, item, threshold) for item in example.evidence)
                        for pointer in pointers)
    return {
        **prediction, 'evidence': [asdict(pointer) for pointer in pointers],
        'id': example.question_id, 'video_id': example.video_id, 'split': example.split,
        'question_type': example.question_type,
        'condition': example.query.condition, 'reference_answer': gold,
        'correct': correct, 'supported': supported,
    }


def ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def summarize(records: list[dict]) -> dict:
    answerable = [row for row in records if row['reference_answer'] is not None]
    unanswerable = [row for row in records if row['reference_answer'] is None]
    cited = [row for row in records if row['supported'] is not None and row['answer'] is not None]
    annotated = [row for row in answerable if row['supported'] is not None]
    correct = sum(row['correct'] for row in answerable)
    answered = sum(row['answer'] is not None for row in records)
    latencies = sorted(row['completion_seconds'] for row in records if 'completion_seconds' in row)
    return {
        'n': len(records), 'n_videos': len({row['video_id'] for row in records}),
        'n_answerable': len(answerable), 'n_reference_evidence': len(annotated),
        'n_unanswerable': len(unanswerable), 'accuracy': ratio(correct, len(answerable)),
        'coverage': ratio(answered, len(records)), 'abstention_rate': ratio(len(records) - answered, len(records)),
        'answered_accuracy': ratio(correct, sum(row['answer'] is not None for row in answerable)),
        'unanswerable_abstention_rate': ratio(sum(row['answer'] is None for row in unanswerable), len(unanswerable)),
        'evidence_precision': ratio(sum(row['supported'] for row in cited), sum(row['pointer_count'] for row in cited)),
        'answer_and_evidence_accuracy': ratio(sum(
            row['correct'] and row['supported'] > 0 and row['supported'] == row['pointer_count'] for row in annotated
        ), len(annotated)),
        'budget_exhausted': sum(row['status'] == 'budget_exhausted' for row in records),
        'mean_input_tokens': ratio(sum(row['input_tokens'] for row in records), len(records)),
        'p95_completion_seconds': latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else None,
    }


def save_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        json.dump(payload, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write('\n')
    temporary.replace(path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run an AV-EviPointer experiment matrix')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo', action='store_true', help='run symbolic CPU fixtures')
    source.add_argument('--manifest', type=Path, help='local labeled video JSONL')
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--model', default='tencent/WeMM-Video-9B')
    parser.add_argument('--revision', help='40-character Hugging Face checkpoint commit')
    parser.add_argument('--out', type=Path, default=Path('runs/results.json'))
    parser.add_argument('--window-seconds', type=float, default=2.0)
    parser.add_argument('--slots', type=int, default=1)
    parser.add_argument('--input-budget', type=int, default=8192)
    parser.add_argument('--max-new-tokens', type=int, default=320)
    parser.add_argument('--retry', type=int, default=1)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if not math.isfinite(args.window_seconds) or args.window_seconds <= 0:
        parser.error('--window-seconds must be positive and finite')
    if min(args.slots, args.input_budget, args.max_new_tokens) <= 0 or args.retry < 0:
        parser.error('slots and token budgets must be positive, and retry must be nonnegative')
    if not 0 < args.iou <= 1 or (args.limit is not None and args.limit <= 0):
        parser.error('IoU must be in (0,1], and limit must be positive')
    if args.demo and args.window_seconds != 2.0:
        parser.error('demo fixtures use two-second windows')
    if not args.demo and not re.fullmatch('[0-9a-fA-F]{40}', args.revision or ''):
        parser.error('--manifest requires a pinned 40-character --revision')
    return args


def main() -> None:
    args = arguments()
    examples = demo_examples() if args.demo else load_manifest(args.manifest)
    examples = examples[:args.limit] if args.limit is not None else examples
    backend = SyntheticBackend() if args.demo else WeMMBackend(args.model, args.revision, args.max_new_tokens)
    settings = {
        'window_seconds': args.window_seconds, 'slots_per_modality': args.slots,
        'input_budget': args.input_budget, 'max_new_tokens': args.max_new_tokens,
        'retry': args.retry, 'evidence_iou': args.iou,
    }
    payload = {
        'kind': 'synthetic_smoke' if args.demo else 'model_evaluation',
        'model': 'symbolic_fixture' if args.demo else args.model,
        'revision': None if args.demo else args.revision, 'settings': settings,
        'modes': {},
    }
    for mode in dict.fromkeys(args.modes):
        rows = []
        groups = defaultdict(list)
        for example in examples:
            start = time.perf_counter()
            prediction = run_question(backend, example.query, MODES[mode], window_seconds=args.window_seconds,
                                      slots=args.slots, budget=args.input_budget, retries=args.retry)
            if not args.demo:
                prediction['completion_seconds'] = round(time.perf_counter() - start, 6)
            row = score(example, prediction, args.iou)
            rows.append(row)
            groups[(example.split, example.query.condition)].append(row)
        payload['modes'][mode] = {
            'summary': summarize(rows), 'records': rows,
            'groups': [{'split': split, 'condition': condition, **summarize(group)}
                       for (split, condition), group in sorted(groups.items())],
        }
        # Save each completed mode so later model or media errors preserve prior results.
        save_atomic(args.out, payload)
        metrics = payload['modes'][mode]['summary']
        print(f'{mode:18} n={metrics["n"]} accuracy={metrics["accuracy"]} '
              f'coverage={metrics["coverage"]} evidence={metrics["evidence_precision"]}')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
