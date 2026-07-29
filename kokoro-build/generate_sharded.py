from __future__ import annotations
import base64, json, os, subprocess, tempfile, traceback, zlib
from pathlib import Path
import numpy as np
import soundfile as sf
from kokoro import KPipeline

ROOT = Path(__file__).resolve().parent
ITEMS = json.loads(zlib.decompress(base64.b64decode((ROOT / 'texts.zlib.b64').read_bytes())))
OUT = ROOT / 'out'
OUT.mkdir(exist_ok=True)
SHARD = int(os.getenv('SHARD_INDEX', '0'))
SHARDS = int(os.getenv('SHARD_TOTAL', '4'))
VOICE = os.getenv('KOKORO_VOICE', 'af_heart')
SPEEDS = {'normal': 1.0, 'slow': 0.85}
SR = 24000
MAX = 200 * 1024
REPL = {
    'ROI': 'R O I', 'CTR': 'C T R', 'CPC': 'C P C', 'SKU': 'S K U',
    'FBA': 'F B A', 'PPC': 'P P C', 'ACoS': 'A C O S',
    'ROAS': 'R O A S', 'KPI': 'K P I', 'KPIs': 'K P I s',
    'SEO': 'S E O', 'API': 'A P I'
}

def speech_text(text: str) -> str:
    text = text.replace('___', '').replace('__', '').replace('_', '')
    for src, dst in REPL.items():
        text = text.replace(src, dst)
    return ' '.join(text.split())

def synth(pipe, text: str, speed: float) -> np.ndarray:
    chunks = []
    for _, _, audio in pipe(text, voice=VOICE, speed=speed, split_pattern=r'\n+'):
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size:
            if chunks:
                chunks.append(np.zeros(int(SR * 0.06), np.float32))
            chunks.append(arr)
    if not chunks:
        raise RuntimeError('No audio generated')
    return np.concatenate(chunks)

def encode_mp3(audio: np.ndarray, destination: Path) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        wav = Path(temp_dir) / 'audio.wav'
        sf.write(wav, audio, SR, subtype='PCM_16')
        audio_filter = (
            'silenceremove=start_periods=1:start_silence=0.02:start_threshold=-48dB:'
            'stop_periods=-1:stop_silence=0.08:stop_threshold=-48dB,'
            'loudnorm=I=-17:LRA=7:TP=-1.5'
        )
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error', '-i', str(wav), '-af', audio_filter,
            '-ac', '1', '-ar', '24000', '-codec:a', 'libmp3lame', '-b:a', '48k',
            str(destination)
        ], check=True)
    if destination.stat().st_size > MAX:
        temporary = destination.with_suffix('.tmp.mp3')
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error', '-i', str(destination),
            '-ac', '1', '-ar', '24000', '-codec:a', 'libmp3lame', '-b:a', '32k',
            str(temporary)
        ], check=True)
        temporary.replace(destination)
    if destination.stat().st_size > MAX:
        raise RuntimeError(f'{destination.name} exceeds 200 KB')

pipeline = KPipeline(lang_code='a')
selected = [item for index, item in enumerate(ITEMS) if index % SHARDS == SHARD]
results = []
failures = []
for number, item in enumerate(selected, 1):
    spoken = speech_text(item['text'])
    for label, speed in SPEEDS.items():
        destination = OUT / f"{item['id']}_{label}.mp3"
        try:
            encode_mp3(synth(pipeline, spoken, speed), destination)
            results.append({
                'id': item['id'], 'text': item['text'], 'targets': item['targets'],
                'speed': label, 'file': destination.name,
                'bytes': destination.stat().st_size
            })
        except Exception as exc:
            failures.append({
                'id': item['id'], 'text': item['text'], 'speed': label,
                'error': str(exc), 'trace': traceback.format_exc()
            })
    if number % 10 == 0 or number == len(selected):
        print(f'shard {SHARD}: {number}/{len(selected)}', flush=True)
(OUT / f'manifest-{SHARD}.json').write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8'
)
(OUT / f'failures-{SHARD}.json').write_text(
    json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8'
)
print({
    'shard': SHARD, 'items': len(selected), 'files': len(results),
    'failures': len(failures),
    'max_bytes': max((item['bytes'] for item in results), default=0)
})
if failures:
    raise SystemExit(1)
