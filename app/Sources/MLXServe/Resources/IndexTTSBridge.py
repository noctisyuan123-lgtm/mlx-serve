"""Generate an IndexTTS 2.5 WAV with the user's installed MLX runtime.

One JSON request is read from stdin.  The Swift caller owns process lifetime;
this script never downloads models or changes the installed Python environment.

Runs on the mlx-indextts2 runtime (github.com/vanch007/mlx-indextts2) — its own
Python environment, separate from the mlx-audio one the Breeze bridge uses,
because mlx-audio has no IndexTTS 2.x implementation.

Progress: IndexTTS has no streaming decoder, but the runtime's segment-level
stream yields one chunk per text segment carrying segment_index/segment_count.
Each completed segment becomes one progress event with total = segment count —
exact, no estimation — so the UI draws a real determinate bar.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def synthesize(request: dict) -> None:
    model_dir = Path(request["model"]).expanduser().resolve(strict=True)
    if not (model_dir / "config.yaml").is_file():
        raise FileNotFoundError("IndexTTS 2.5 checkpoint is incomplete: config.yaml missing")
    text = request["text"].strip()
    ref_audio = request.get("ref_audio") or None
    speed = float(request.get("speed", 1.0))
    temperature = float(request.get("temperature", 0.8))
    if not text:
        raise ValueError("Text is empty")
    if not ref_audio:
        raise ValueError("IndexTTS 2.5 requires a reference voice clip — it clones from audio, not text")
    if speed <= 0:
        raise ValueError("Speed must be positive")

    from mlx_indextts import IndexTTSv25

    tts = IndexTTSv25(str(model_dir))
    chunks: list[np.ndarray] = []
    sample_rate = 22050
    for chunk in tts.stream(
        text=text,
        reference_audio=ref_audio,
        speed=speed,
        temperature=temperature,
    ):
        audio = np.asarray(chunk.audio, dtype=np.float32).reshape(-1)
        if audio.size:
            chunks.append(audio)
            sample_rate = int(chunk.sample_rate)
        emit({"type": "progress", "step": chunk.segment_index + 1,
              "total": chunk.segment_count, "stage": "decode"})
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("IndexTTS returned empty or non-finite audio")

    import soundfile as sf

    output = Path(request["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".wav", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        sf.write(str(temp_path), audio, sample_rate, format="WAV", subtype="PCM_16")
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)
    emit({"type": "complete"})


if __name__ == "__main__":
    try:
        synthesize(json.load(sys.stdin))
    except Exception as exc:
        print(f"IndexTTS 2.5: {exc}", file=sys.stderr)
        sys.exit(1)
