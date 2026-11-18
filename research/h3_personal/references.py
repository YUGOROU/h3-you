"""Extract a local reference candidate, without claiming VAD/voice-quality verification."""
import argparse
import json
from pathlib import Path
import subprocess

from .preprocess import dump


def extract(dataset: Path, start: float, seconds: float):
    dataset = dataset.resolve()
    summary = json.loads((dataset / 'summary.json').read_text())
    if not summary['complete_raw_preprocessing']:
        raise ValueError('Wait for full raw preprocessing before making the Ref2VA manifest')
    records = json.loads((dataset / 'inventory.json').read_text())
    source = next(r for r in records if not r['exclusion_reasons'])
    if start < 0 or not 3 <= seconds <= 15 or start + seconds > source['duration_seconds']:
        raise ValueError('Reference must be a 3-15 second interval within the source')
    root = dataset / 'references'
    root.mkdir(exist_ok=True)
    image, audio = root / 'portrait.png', root / 'voice-candidate.wav'
    if image.exists() or audio.exists():
        raise ValueError('Reference exists; use it or explicitly move it before replacing')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-ss', str(start),
                    '-i', source['path'], '-frames:v', '1', '-vf',
                    'scale=768:1344:force_original_aspect_ratio=decrease:force_divisible_by=2,'
                    'pad=768:1344:(ow-iw)/2:(oh-ih)/2,setsar=1', str(image)], check=True)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-ss', str(start),
                    '-t', str(seconds), '-i', source['path'], '-vn', '-ar', '48000',
                    '-ac', '1', '-c:a', 'pcm_s16le', str(audio)], check=True)
    rows = json.loads((dataset / 'manifest.json').read_text())
    for row in rows:
        row['reference_image'] = str(image)
    dump(dataset / 'manifest-ref2va.json', rows)
    dump(root / 'provenance.json', {'source': source['path'], 'start_seconds': start,
                                  'duration_seconds': seconds,
                                  'voice_quality_verified': False,
                                  'third_party_screening_verified': False,
                                  'note': 'Audition this short candidate before TTS. Not VAD-selected, no transcript.'})
    return {'image': str(image), 'audio_candidate': str(audio)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--start', type=float, default=30)
    parser.add_argument('--seconds', type=float, default=8)
    args = parser.parse_args()
    print(json.dumps(extract(args.dataset, args.start, args.seconds), indent=2))
