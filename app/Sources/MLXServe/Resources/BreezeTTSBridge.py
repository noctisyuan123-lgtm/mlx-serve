"""Generate a BF16 Breeze-TTS 2 WAV with the user's installed MLX runtime.

One JSON request is read from stdin.  The Swift caller owns process lifetime;
this script never downloads models or changes the installed Python environment.

Progress is streamed to stdout as one JSON object per line, using the same
`{"type": "progress", "step", "total", "stage"}` envelope the server's media
endpoints emit over SSE.  `total` stays 0 (clip length is model-determined)
and `step` counts 1920-sample frames so the Swift side converts step to
seconds with the exact constants the native speech path uses.
"""

import json
import os
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from mlx_audio.tts.utils import load_model


# The native speech engine reports progress in 1920-sample talker frames at
# 24 kHz (~0.08 s each); AudioGenService turns step into seconds with the same
# constants, so counting this frame size keeps the two meters identical.
NATIVE_FRAME_SAMPLES = 1920


def estimate_total_frames(text: str) -> int:
    """Rough clip-length estimate so the UI can draw a real (determinate) bar.

    The engine-side TTS also cannot know the length ahead of time and stays
    indeterminate (total=0) — that reads as a spinner, not a bar.  Calibrated
    against a real synthesis (15 chars → 56 frames ≈ 3.7 frames/char at 24 kHz);
    3.2 keeps the bar just behind reality so it lands as the clip does (the
    Swift side clamps the fraction at 1.0 regardless).
    """
    chars = sum(1 for ch in text if not ch.isspace())
    return max(24, int(chars * 3.2))


# The estimate is only ±20% honest, so cap the reported step at 95% of it: an
# under-estimate then parks the bar at "step 22 of 24" (reads as "almost done")
# instead of a full bar that sits there lying while the tail audio renders.
DISPLAY_CAP_PERCENT = 95


def emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def synthesize(request: dict) -> None:
    model_dir = Path(request["model"]).expanduser().resolve(strict=True)
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_type") != "breeze_tts":
        raise ValueError("The selected checkpoint is not Breeze-TTS 2")
    if config.get("quantization"):
        raise ValueError("Breeze-TTS 2 bridge requires the unquantized BF16 checkpoint")
    for name in (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "audio_tokenizer/model.safetensors",
    ):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"BF16 Breeze checkpoint is incomplete: {name}")

    text = request["text"].strip()
    ref_audio = request.get("ref_audio") or None
    ref_text = (request.get("ref_text") or "").strip()
    if not text:
        raise ValueError("Text is empty")
    if ref_audio and not ref_text:
        raise ValueError("Breeze voice cloning requires the reference transcript")
    if float(request.get("speed", 1.0)) != 1.0:
        raise ValueError("Breeze-TTS 2 does not support playback speed control")

    model = load_model(str(model_dir))
    total = estimate_total_frames(text)
    chunks: list[np.ndarray] = []
    emitted_samples = 0
    last_step = 0
    for result in model.generate(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text or None,
        temperature=float(request.get("temperature", 0.7)),
        max_tokens=1100,
        stream=True,
        streaming_interval=1.0,
    ):
        audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
        if audio.size:
            chunks.append(audio)
            emitted_samples += int(audio.size)
            step = min(emitted_samples // NATIVE_FRAME_SAMPLES,
                       total * DISPLAY_CAP_PERCENT // 100)
            if step > last_step:
                last_step = step
                emit({"type": "progress", "step": step, "total": total, "stage": "decode"})
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("Breeze returned empty or non-finite audio")
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    output = Path(request["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".wav", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        with wave.open(str(temp_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(int(result.sample_rate))
            wav.writeframes(pcm.tobytes())
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)
    emit({"type": "complete"})


if __name__ == "__main__":
    try:
        synthesize(json.load(sys.stdin))
    except Exception as exc:
        print(f"Breeze-TTS 2: {exc}", file=sys.stderr)
        sys.exit(1)
