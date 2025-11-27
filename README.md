# AIClone

Create a fully synthetic “digital twin” that can describe its own origin story. The process below keeps the workflow modular so you can swap in different generative models as they evolve while still protecting your original biometric data.

## 0. Safety, Consent, and Storage
- Confirm you own the likeness/voice and obtain written consent from anyone who appears in the dataset.
- Keep raw captures in an encrypted location (`gocryptfs`, `age`, or a hardware encrypted SSD). The training scripts below only ever see pre-processed copies.
- Watermark all generated media and keep an audit log (see `scripts/clone_pipeline.py`) so you can prove provenance later.

## 1. Capture Reference Data
| Asset | Target | Tips |
| --- | --- | --- |
| Photos | 60–120 high-res stills, 4–6 lighting setups | Shoot 4K/8K if possible, mix expressions/angles, neutral backgrounds. |
| Voice | 5–10 minutes at 44.1 kHz mono WAV | Read varied content (technical, casual, emotional). Maintain < -12 dBFS peaks. |

Keep a short “process narration” script explaining how the clone was created—you will feed this into TTS later so the clone literally describes its birth.

## 2. Pre-process
1. **Images**: `python scripts/clone_pipeline.py prep-images --src raw/photos --dst data/images`  
   - Auto-center faces with `mediapipe`, remove blurry frames, normalize exposure.  
2. **Audio**: `python scripts/clone_pipeline.py prep-audio --src raw/voice.wav --dst data/audio/clean.wav`  
   - High-pass at 80 Hz, de-noise with `noisereduce`, loudness normalize to -23 LUFS.  

## 3. Train Appearance and Voice Models
### 3.1 Appearance (Diffusion LoRA)
```bash
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name="stabilityai/sd-1.5" \
  --instance_data_dir="data/images" \
  --output_dir="models/vision-lora" \
  --resolution=768 --train_batch_size=1 --learning_rate=1e-4 \
  --max_train_steps=2000 --checkpointing_steps=200
```

### 3.2 Voice (NeMo, XTTS, or Bark fine-tune)
```bash
python scripts/clone_pipeline.py train-voice \
  --config configs/xtts.yaml \
  --dataset data/audio/clean.wav \
  --output models/voice-xtts
```
- Keep 10% of the takes for validation to catch overfitting.
- Export both the model weights and tokenizer/vocab artifacts; the pipeline script expects a Hugging Face style directory.

## 4. Generate Media
1. **Images or Talking Head Frames**  
   `python scripts/clone_pipeline.py gen-visual --prompt "A confident AI engineer explaining how it was trained" --lora models/vision-lora --frames 240`
2. **Narrated Speech**  
   `python scripts/clone_pipeline.py gen-voice --text scripts/narration.txt --voice models/voice-xtts --out assets/narration.wav`
3. **Lip-sync / Video Assembly**  
   - Use `Wav2Lip` or `SadTalker` to map narration onto the generated frames.  
   - Optionally, drive a 3D avatar in Unreal MetaHuman LiveLink if you already have a rig.

## 5. Make the Clone Explain Its Origin
Write a short first-person script such as:
> “I am the composite of a LoRA fine-tuned diffusion model and an XTTS voice clone. My creator captured 90 photos under four lighting rigs…”

Feed that text into Step 4.2 so the clone genuinely narrates the pipeline that birthed it. This fulfills the requirement that it “speaks by reflecting the process of how it was created.”

## 6. Automation (`scripts/clone_pipeline.py`)
The included orchestration script stitches together:
1. Data prep for images/audio.
2. Hooks to call DreamBooth/LoRA + XTTS training commands.
3. Batched generation routines.
4. A provenance log (`artifacts/run-*.json`) capturing prompts, checkpoints, seeds, and git hashes for reproducibility.

Run `python scripts/clone_pipeline.py --help` after installing the requirements (see next section) to explore each stage.

## 7. Environment Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` tracks the orchestration-level dependencies. Heavy models (Diffusers, NeMo, Wav2Lip) are optional extras you install only for the components you plan to execute locally.

## Repository Layout

```
README.md                 # End-to-end plan + data requirements
requirements.txt          # Python dependencies for orchestration helpers
scripts/clone_pipeline.py # Typer CLI for prep/logging/narration
configs/xtts.yaml         # Example config stub for voice fine-tuning
artifacts/                # Auto-populated provenance logs
data/, assets/            # Working directories for media
```

## CLI Highlights (`scripts/clone_pipeline.py`)

- `prep-images`: center-crops + resizes portrait stills into clean DreamBooth/LoRA inputs while logging provenance.
- `prep-audio`: converts narration takes into mono 44.1 kHz WAV with target loudness.
- `train-voice`: records (and optionally runs) whatever fine-tune command you want, keeping a manifest for reproducibility.
- `compose-narration`: emits a ready-to-read script where the clone explains how it was produced.
- `gen-voice`: wraps your preferred synthesizer command, ensuring text + metadata are stored alongside the render.
- `show-plan`: prints a lightweight checklist so you never lose track of which stage you are on.

Run `python scripts/clone_pipeline.py --help` for the full list of commands and options.

## Suggested Next Steps

- Wire `gen-voice` to your actual inference script (XTTS, NeMo, Bark, etc.) so narration.wav is produced in one shot.
- Add a `gen-visual` command that calls your preferred diffusion/motion model (Flux, SVD, Kling, Pika) and logs the seed.
- Integrate Wav2Lip / SadTalker CLI invocation plus DaVinci Resolve render presets for fully automated video assembly.
- Layer an LLM-driven chat persona on top of the generated voice so the clone can converse live as well as narrate its origin.