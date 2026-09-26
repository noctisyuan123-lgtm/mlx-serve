"""Generate a dots.tts WAV with the user's installed MLX runtime.

One JSON request is read from stdin.  The Swift caller owns process lifetime;
this script never downloads models or changes the installed Python environment.

Runs on the dots-tts-mlx runtime (github.com/sb1992/dots-tts-mlx, a pure-MLX
port of rednote-hilab/dots.tts) — its own Python environment, separate from
the mlx-audio (Breeze) and mlx-indextts2 (IndexTTS) ones.

Progress: dots.tts has no streaming decoder, but the reference conditioning
can be enrolled once and reused, so the bridge synthesizes sentence by
sentence — each finished sentence is one progress event with total = sentence
count (exact), and the reference encode is paid once instead of per sentence.

Language: the runtime wants an explicit uppercase ISO code and does not
auto-detect, so the bridge guesses from the script — CJK variants by their
syllabaries (kana before han, since Japanese also uses han), then wider
scripts, then diacritic hints inside Latin, defaulting to EN.
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np


def emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def guess_language(text: str) -> str:
    if re.search(r"[\u3040-\u30ff]", text):
        return "JA"
    if re.search(r"[\uac00-\ud7af]", text):
        return "KO"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "ZH"
    if re.search(r"[\u0400-\u04ff]", text):
        return "RU"
    if re.search(r"[\u0600-\u06ff]", text):
        return "AR"
    if re.search(r"[\u0900-\u097f]", text):
        return "HI"
    if re.search(r"[\u0e00-\u0e7f]", text):
        return "TH"
    if re.search(r"[ñ¿¡]", text):
        return "ES"
    if re.search(r"[äöüß]", text):
        return "DE"
    if re.search(r"[ãõ]", text):
        return "PT"
    if re.search(r"[àèéêôç]", text):
        return "FR"
    return "EN"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；!?;.…])\s*", text)
    sentences: list[str] = []
    for part in (p.strip() for p in parts):
        if not part:
            continue
        # A single unpunctuated run can exceed the model's patch budget;
        # fall back to clause-level breaks so every piece stays short.
        if len(part) > 200:
            sentences.extend(s for s in re.split(r"(?<=[，,、])\s*", part) if s.strip())
        else:
            sentences.append(part)
    return sentences or ([text.strip()] if text.strip() else [])


def synthesize(request: dict) -> None:
    model_dir = Path(request["model"]).expanduser().resolve(strict=True)
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError("dots.tts checkpoint is incomplete: config.json missing")
    text = request["text"].strip()
    ref_audio = request.get("ref_audio") or None
    ref_text = (request.get("ref_text") or "").strip()
    speed = float(request.get("speed", 1.0))
    if not text:
        raise ValueError("Text is empty")
    if not ref_audio:
        raise ValueError("dots.tts requires a reference voice clip — it clones from audio, not text")
    if not ref_text:
        raise ValueError("dots.tts voice cloning requires the reference transcript")
    if speed != 1.0:
        raise ValueError("dots.tts bridge does not support playback speed control")

    import mlx.core as mx
    import soundfile as sf
    from dots_tts_mlx.loader import from_pretrained

    model = from_pretrained(str(model_dir), dtype=mx.bfloat16).model
    profile = model.enroll(ref_audio, ref_text)
    language = guess_language(text)
    sentences = split_sentences(text)

    chunks: list[np.ndarray] = []
    sample_rate = 48000
    for index, sentence in enumerate(sentences):
        out = model.generate(
            sentence,
            profile=profile,
            language=language,
            seed=int(os.urandom(4).hex(), 16) % (2**31),
        )
        audio = np.asarray(mx.array(out["audio"]), dtype=np.float32).reshape(-1)
        sample_rate = int(out["sample_rate"])
        if audio.size:
            chunks.append(audio)
            if index < len(sentences) - 1:
                chunks.append(np.zeros(int(sample_rate * 0.12), dtype=np.float32))
        emit({"type": "progress", "step": index + 1,
              "total": len(sentences), "stage": "decode"})
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("dots.tts returned empty or non-finite audio")

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
        print(f"dots.tts: {exc}", file=sys.stderr)
        sys.exit(1)
