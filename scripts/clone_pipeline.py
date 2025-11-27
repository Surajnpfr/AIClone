#!/usr/bin/env python3
"""
Opinionated orchestration helpers for building an AI-powered clone.

The heavy lifting (DreamBooth / LoRA fine-tunes, XTTS voice training,
lip-sync, etc.) still relies on dedicated scripts or notebooks, but this CLI
keeps the mundane steps—data prep, provenance logging, narration composition,
and command templating—consistent across runs.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

APP_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = APP_ROOT / "artifacts"
DEFAULT_ASSETS = APP_ROOT / "assets"

console = Console()
app = typer.Typer(help="Utilities for prepping data and logging clone runs.")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise typer.BadParameter(f"{path} does not exist")
    return path


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


@dataclass
class ProvenanceRecorder:
    stage: str
    entries: List[Dict[str, Any]] = field(default_factory=list)
    path: Path = field(init=False)

    def __post_init__(self) -> None:
        _ensure_dir(ARTIFACTS_DIR)
        self.path = ARTIFACTS_DIR / f"{self.stage}-{_timestamp()}.json"

    def log(self, message: str, **payload: Any) -> None:
        self.entries.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "stage": self.stage,
                "message": message,
                "payload": payload,
            }
        )

    def finalize(self) -> None:
        self.path.write_text(json.dumps(self.entries, indent=2))
        console.log(f"[green]Provenance saved to {self.path}")


@contextmanager
def provenance_stage(stage: str) -> Iterable[ProvenanceRecorder]:
    recorder = ProvenanceRecorder(stage)
    try:
        yield recorder
    finally:
        recorder.finalize()


def _center_crop(image):
    width, height = image.size
    min_edge = min(width, height)
    left = (width - min_edge) // 2
    top = (height - min_edge) // 2
    right = left + min_edge
    bottom = top + min_edge
    return image.crop((left, top, right, bottom))


@app.command("prep-images")
def prep_images(
    src: str = typer.Argument(..., help="Directory with raw reference photos"),
    dst: str = typer.Argument(..., help="Directory to store normalized crops"),
    size: int = typer.Option(768, help="Output resolution (square)"),
    sharpen: bool = typer.Option(False, help="Apply a light unsharp mask"),
    overwrite: bool = typer.Option(False, help="Overwrite existing files"),
) -> None:
    """
    Normalize portraits for DreamBooth/LoRA by center-cropping and resizing.
    """
    from PIL import Image, ImageFilter  # Local import keeps startup fast

    src_path = _resolve(src)
    dst_path = Path(dst).expanduser().resolve()
    _ensure_dir(dst_path)

    images = sorted(
        [p for p in src_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if not images:
        raise typer.BadParameter(f"No PNG/JPEG files found in {src_path}")

    with provenance_stage("prep-images") as log, Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}")
    ) as progress:
        task = progress.add_task("Processing portraits…", total=len(images))
        written = 0
        for path in images:
            out_path = dst_path / f"{path.stem}_prep.jpg"
            if out_path.exists() and not overwrite:
                progress.advance(task)
                continue
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = _center_crop(img)
                img = img.resize((size, size), Image.LANCZOS)
                if sharpen:
                    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=125))
                img.save(out_path, quality=95)
            written += 1
            progress.advance(task)

        log.log(
            "image_prep_complete",
            total=len(images),
            written=written,
            src=str(src_path),
            dst=str(dst_path),
            size=size,
            sharpen=sharpen,
        )
    console.print(f"[bold green]Saved {written} normalized images to {dst_path}")


@app.command("prep-audio")
def prep_audio(
    src: str = typer.Argument(..., help="Path to raw narration WAV/FLAC"),
    dst: str = typer.Argument(..., help="Output wav path"),
    sample_rate: int = typer.Option(44100, help="Target sample rate"),
    loudness: float = typer.Option(-23.0, help="Target LUFS (approximate)"),
) -> None:
    """
    Convert, denoise lightly, and loudness-normalize narration reference audio.
    """
    from pydub import AudioSegment

    src_path = _resolve(src)
    dst_path = Path(dst).expanduser().resolve()
    _ensure_dir(dst_path.parent)

    with provenance_stage("prep-audio") as log:
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_channels(1).set_frame_rate(sample_rate)
        gain = loudness - audio.dBFS
        cleaned = audio.apply_gain(gain)
        cleaned.export(dst_path, format="wav")
        log.log(
            "audio_prep_complete",
            src=str(src_path),
            dst=str(dst_path),
            sample_rate=sample_rate,
            target_loudness=loudness,
            applied_gain=float(round(gain, 3)),
            duration_seconds=round(len(cleaned) / 1000, 2),
        )
    console.print(f"[bold green]Normalized narration saved to {dst_path}")


@app.command("train-voice")
def train_voice(
    dataset: str = typer.Argument(..., help="Prepared WAV file or manifest"),
    output: str = typer.Argument(..., help="Directory to store checkpoints"),
    config: Optional[str] = typer.Option(None, help="Path to trainer config yaml"),
    trainer: Optional[str] = typer.Option(
        None,
        help="Shell command that actually runs training "
        '(e.g. "python xtts_finetune.py --config configs/xtts.yaml")',
    ),
    execute: bool = typer.Option(
        False,
        "--execute/--dry-run",
        help="Run the trainer command immediately (otherwise just log it).",
    ),
) -> None:
    """
    Record (and optionally execute) the fine-tune command for the voice model.
    """
    dataset_path = _resolve(dataset)
    output_path = Path(output).expanduser().resolve()
    _ensure_dir(output_path)
    config_path = Path(config).expanduser().resolve() if config else None

    with provenance_stage("train-voice") as log:
        log.log(
            "voice_training_planned",
            dataset=str(dataset_path),
            output=str(output_path),
            config=str(config_path) if config_path else None,
            trainer_command=trainer,
        )
        if trainer and execute:
            console.log(f"[cyan]Running trainer command: {trainer}")
            subprocess.run(shlex.split(trainer), check=True)
            log.log("voice_training_executed", command=trainer)
        else:
            console.print(
                "[yellow]Dry run only. Use --execute to actually run the trainer command."
            )


@app.command("gen-voice")
def generate_voice(
    text: str = typer.Option(..., help="Narration text to synthesize"),
    voice_model: str = typer.Option(..., help="Path to fine-tuned voice model"),
    output: str = typer.Option(
        str(DEFAULT_ASSETS / "narration.wav"), help="Where to store narration audio"
    ),
    synthesizer: Optional[str] = typer.Option(
        None,
        help="Custom command to call your preferred TTS engine. Use {text}, {model}, {out} placeholders.",
    ),
) -> None:
    """
    Generate narration audio via an external synthesizer command and log metadata.
    """
    out_path = Path(output).expanduser().resolve()
    _ensure_dir(out_path.parent)

    if synthesizer is None:
        raise typer.BadParameter(
            "Specify --synthesizer with a command like "
            '"python xtts_infer.py --model {model} --text-file {text} --out {out}"'
        )

    text_path = DEFAULT_ASSETS / "narration.txt"
    _ensure_dir(text_path.parent)
    text_path.write_text(text.strip())

    command = synthesizer.format(text=text_path, model=voice_model, out=out_path)

    with provenance_stage("gen-voice") as log:
        console.log(f"[cyan]Executing synthesizer: {command}")
        subprocess.run(shlex.split(command), check=True)
        log.log(
            "voice_generated",
            text_file=str(text_path),
            model=voice_model,
            output=str(out_path),
            command=command,
        )
    console.print(f"[bold green]Narration audio exported to {out_path}")


@app.command("compose-narration")
def compose_narration(
    output: str = typer.Option(
        str(DEFAULT_ASSETS / "narration.txt"), help="File to store narration script"
    ),
    persona: str = typer.Option("AI Twin", help="Name used inside the narration"),
    photo_count: int = typer.Option(90, help="How many photos fed into LoRA"),
    lighting_setups: int = typer.Option(4, help="Lighting setups captured"),
    voice_minutes: float = typer.Option(7.5, help="Minutes of voice data"),
    vision_model: str = typer.Option("vision-lora", help="Appearance checkpoint name"),
    voice_model: str = typer.Option("voice-xtts", help="Voice checkpoint name"),
) -> None:
    """
    Craft a first-person narration explaining how the clone was created.
    """
    narration = f"""
I am {persona}, a digital narration of myself. My visual core was distilled from {photo_count} curated frames captured across {lighting_setups} lighting rigs, then fine-tuned as the LoRA checkpoint "{vision_model}". My voice rides on the "{voice_model}" acoustic model, adapted from {voice_minutes:.1f} minutes of studio audio. Every sentence I speak carries provenance: prompts, seeds, and git hashes are preserved inside the artifacts ledger so you can trace my origin end-to-end.
"""
    narration = "\n".join(line.strip() for line in narration.strip().splitlines())
    out_path = Path(output).expanduser().resolve()
    _ensure_dir(out_path.parent)
    out_path.write_text(narration)

    with provenance_stage("compose-narration") as log:
        log.log(
            "narration_composed",
            persona=persona,
            photo_count=photo_count,
            lighting_setups=lighting_setups,
            voice_minutes=voice_minutes,
            vision_model=vision_model,
            voice_model=voice_model,
            output=str(out_path),
        )
    console.print(f"[bold green]Narration script saved to {out_path}")


@app.command("show-plan")
def show_plan() -> None:
    """
    Print a high-level checklist to track clone progress.
    """
    table = Table(title="Clone Build Checklist")
    table.add_column("Stage", justify="left")
    table.add_column("Status", justify="center")
    table.add_column("Notes", justify="left")
    table.add_row("Capture", "⬜", "Photos + voice per README guidance")
    table.add_row("Pre-process", "⬜", "Run prep-images / prep-audio")
    table.add_row("Appearance FT", "⬜", "DreamBooth / LoRA checkpoints")
    table.add_row("Voice FT", "⬜", "XTTS or NeMo script")
    table.add_row("Narration", "⬜", "compose-narration + gen-voice")
    table.add_row("Lip-sync", "⬜", "SadTalker / Wav2Lip with narration.wav")
    table.add_row("Assembly", "⬜", "Combine audio + frames in Resolve")
    console.print(table)


if __name__ == "__main__":
    app()
