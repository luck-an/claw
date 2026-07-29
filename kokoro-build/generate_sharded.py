from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import traceback
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

ROOT = Path(__file__).resolve().parent
PARTS = ROOT / "parts"
encoded = "".join(p.read_text(encoding="utf-8").strip() for p in sorted(PARTS.glob("texts.part*")))
ITEMS = json.loads(zlib.decompress(base64.b64decode(encoded)))
if len(ITEMS) != 517:
    raise RuntimeError(f"Expected 517 unique texts, got {len(ITEMS)}")

OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
SHARD = int(os.getenv("SHARD_INDEX", "0"))
SHARDS = int(os.getenv("SHARD_TOTAL", "4"))
VOICE = os.getenv("KOKORO_VOICE", "af_heart")
SPEEDS = {"normal": 1.0, "slow": 0.85}
SR = 24000
MAX_BYTES = 200 * 1024
REPLACEMENTS = {
    "ROI": "R O I",
    "CTR": "C T R",
    "CPC": "C P C",
    "SKU": "S K U",
    "FBA": "F B A",
    "PPC": "P P C",
    "ACoS": "A C O S",
    "ROAS": "R O A S",
    "KPI": "K P I",
    "KPIs": "K P I s",
    "SEO": "S E O",
    "API": "A P I",
}


def speech_text(text: str) -> str:
    text = text.replace("___", "").replace("__", "").replace("_", "")
    for source, target in REPLACEMENTS.items():
        text = text.replace(source, target)
    return " ".join(text.split())


def synthesize(pipeline: KPipeline, text: str, speed: float) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for _, _, audio in pipeline(text, voice=VOICE, speed=speed, split_pattern=r"\n+"):
        array = np.asarray(audio, dtype=np.float32).reshape(-1)
        if array.size:
            if chunks:
                chunks.append(np.zeros(int(SR * 0.06), dtype=np.float32))
            chunks.append(array)
    if not chunks:
        raise RuntimeError("Kokoro returned no audio")
    return np.concatenate(chunks)


def encode_mp3(audio: np.ndarray, destination: Path) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        wav_path = Path(temporary_directory) / "audio.wav"
        sf.write(wav_path, audio, SR, subtype="PCM_16")
        audio_filter = (
            "silenceremove=start_periods=1:start_silence=0.02:start_threshold=-48dB:"
            "stop_periods=-1:stop_silence=0.08:stop_threshold=-48dB,"
            "loudnorm=I=-17:LRA=7:TP=-1.5"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(wav_path),
                "-af",
                audio_filter,
                "-ac",
                "1",
                "-ar",
                "24000",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "48k",
                str(destination),
            ],
            check=True,
        )
    if destination.stat().st_size > MAX_BYTES:
        smaller = destination.with_suffix(".tmp.mp3")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(destination),
                "-ac",
                "1",
                "-ar",
                "24000",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "32k",
                str(smaller),
            ],
            check=True,
        )
        smaller.replace(destination)
    if destination.stat().st_size > MAX_BYTES:
        raise RuntimeError(f"{destination.name} exceeds 200 KB")


pipeline = KPipeline(lang_code="a")
selected = [item for index, item in enumerate(ITEMS) if index % SHARDS == SHARD]
results: list[dict] = []
failures: list[dict] = []

for number, item in enumerate(selected, 1):
    spoken = speech_text(item["text"])
    for label, speed in SPEEDS.items():
        destination = OUT / f"{item['id']}_{label}.mp3"
        try:
            encode_mp3(synthesize(pipeline, spoken, speed), destination)
            results.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "targets": item["targets"],
                    "speed": label,
                    "file": destination.name,
                    "bytes": destination.stat().st_size,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "id": item["id"],
                    "text": item["text"],
                    "speed": label,
                    "error": str(error),
                    "trace": traceback.format_exc(),
                }
            )
    if number % 10 == 0 or number == len(selected):
        print(f"shard {SHARD}: {number}/{len(selected)}", flush=True)

(OUT / f"manifest-{SHARD}.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT / f"failures-{SHARD}.json").write_text(
    json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(
    {
        "shard": SHARD,
        "items": len(selected),
        "files": len(results),
        "failures": len(failures),
        "max_bytes": max((item["bytes"] for item in results), default=0),
    }
)
if failures:
    raise SystemExit(1)
