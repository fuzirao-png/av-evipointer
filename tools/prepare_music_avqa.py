"""Join local MUSIC-AVQA annotations, media, and optional Hard source indices."""

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path


OFFICIAL_SHA256 = 'fa79cbd8e7f0bf043e1761271b216f130832dd419a171bce9d9b561017af6cc2'


def fill_question(template, values):
    fields = ast.literal_eval(values) if isinstance(values, str) else values
    pieces = re.split(r'<[^<>]+>', template)
    if len(pieces) != len(fields) + 1:
        raise ValueError(f'template/field count mismatch: {template!r}, {fields!r}')
    return ''.join(part + str(value) for part, value in zip(pieces, [*fields, '']))


def probe_video(ffprobe, video):
    if not video.is_file():
        return None, 'missing_video'
    result = subprocess.run(
        [ffprobe, '-v', 'error', '-show_entries', 'format=duration:stream=codec_type',
         '-of', 'json', str(video)], capture_output=True, text=True, check=False,
    )
    try:
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        streams = {row['codec_type'] for row in data['streams']}
        if result.returncode or not math.isfinite(duration) or duration <= 0 or 'video' not in streams:
            return None, 'invalid_video'
        if 'audio' not in streams:
            return None, 'missing_audio'
    except (ValueError, TypeError, KeyError):
        return None, 'invalid_video'
    return duration, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--hard-indices', type=Path, help='JSON list of zero-based source_index values')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        parser.error('ffprobe is required')
    if not args.video_dir.is_dir():
        parser.error('--video-dir must be an existing directory')
    annotation_bytes = args.annotations.read_bytes()
    digest = hashlib.sha256(annotation_bytes).hexdigest()
    items = json.loads(annotation_bytes)
    hard = set()
    if args.hard_indices:
        if digest != OFFICIAL_SHA256:
            parser.error('--hard-indices requires the pinned official annotation file in README')
        indices = json.loads(args.hard_indices.read_bytes())
        if not isinstance(indices, list) or not all(type(i) is int and 0 <= i < len(items) for i in indices):
            parser.error('--hard-indices must be a JSON list of valid source_index integers')
        hard = set(indices)
    out = args.out.resolve()
    video_dir = args.video_dir.resolve()
    known, rows, excluded = {}, [], []
    for index, item in enumerate(items):
        video_id = str(item['video_id'])
        if not re.fullmatch(r'[0-9]+', video_id):
            raise ValueError(f'invalid MUSIC-AVQA video ID at source_index {index}')
        video = video_dir / f'{video_id}.mp4'
        split = 'hard' if index in hard else 'standard-only' if hard else 'standard'
        if video_id not in known:
            known[video_id] = probe_video(ffprobe, video)
        duration, exclusion = known[video_id]
        if item.get('question_deleted') in (1, '1', True):
            exclusion = 'deleted_annotation'
        if exclusion:
            excluded.append(dict(source_index=index, video_id=video_id, split=split, reason=exclusion))
            continue
        try:
            relative = os.path.relpath(video, out.parent)
        except ValueError:  # Windows files can be on separate volumes.
            relative = str(video)
        rows.append({
            'id': f'music-avqa:{index}', 'video_id': video_id, 'video': relative,
            'duration': duration, 'split': split, 'question_type': json.loads(item['type']),
            'question': fill_question(item['question_content'], item['templ_values']), 'answer': item['anser'],
        })
    audit = out.with_suffix('.exclusions.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps({'annotations_sha256': digest, 'included': len(rows), 'excluded': excluded},
                                ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if not rows:
        parser.error(f'no usable media. See {audit}')
    out.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    print(f'Included {len(rows)} questions, excluded {len(excluded)}. Audit: {audit}')


if __name__ == '__main__':
    main()
