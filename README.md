# Synthetic Data LoRA for Boxed Reasoning

This project fine-tunes a **LoRA adapter** for NVIDIA Nemotron reasoning tasks using curated/synthetic supervision and a strict output format: final answers must appear in `\boxed{}`. The workflow is intentionally lightweight and offline-friendly (Kaggle-compatible), focusing on **data quality + formatting consistency** rather than full-model retraining.

## Why this approach

This demo targets the **Best Data / Synthetic Data Method** idea:

- Use curated and synthetic training examples to improve reasoning behavior.
- Teach the model to produce a predictable final-answer structure (`\boxed{...}`) that matches downstream scoring.
- Keep training practical with LoRA (small trainable footprint) instead of expensive full-model updates.

## What the notebook does

The notebook (`nvidia-nemotron-submission-demo.ipynb`) runs a full offline adapter pipeline:

1. **Load and split training data** from `train.csv`.
2. **Apply environment compatibility fixes** (including a Triton `ptxas` workaround in Kaggle environments).
3. **Load model/tokenizer strictly offline** from local artifact paths.
4. **Build supervised examples** with system/user/assistant formatting and boxed-answer targets.
5. **Train a LoRA adapter** with Hugging Face `Trainer` + `peft`.
6. **Save adapter artifacts** and package a `submission.zip`.

## Repository contents

- `nvidia-nemotron-submission-demo.ipynb` — end-to-end training + packaging notebook.
- `README.md` — this document.

## Training design highlights

- **Prompt policy:** explicit instruction to place final answer in one `\boxed{}` expression.
- **Answer normalization:** answers without `\boxed{}` are converted to `The final answer is \boxed{...}.`
- **Loss masking:** prompt tokens are masked (`-100`) so optimization focuses on assistant target tokens.
- **Dynamic padding:** batch-time padding for efficient variable-length training.
- **LoRA targets:** projection layers matched by regex:
  - `.*\.(in_proj|out_proj|up_proj|down_proj)$`

## Default configuration (from notebook)

- `TRAIN_ROWS_LIMIT = 2000`
- `VAL_FRACTION = 0.05`
- `MAX_PROMPT_TOKENS = 256`
- `MAX_ANSWER_TOKENS = 64`
- `MAX_TOTAL_TOKENS = 384`
- `LORA_RANK = 8`, `LORA_ALPHA = 16`, `LORA_DROPOUT = 0.05`
- `PER_DEVICE_BATCH_SIZE = 1`, `GRAD_ACCUM_STEPS = 8`
- `NUM_EPOCHS = 1`, `LEARNING_RATE = 1e-4`

## Expected runtime environment

This notebook is designed for an offline GPU session (example: Kaggle + RTX Pro 6000) and assumes:

- Local model artifacts are present.
- CUDA is available.
- Local custom dependencies from NVIDIA utility paths are accessible (`cutlass`, `mamba_ssm`).

The script fails early with actionable errors if required offline dependencies are missing.

## How to run

1. Open the notebook in a GPU-enabled offline environment.
2. Verify or edit the path constants:
   - `TRAIN_CSV`
   - `MODEL_PATH`
   - `LOCAL_TOKENIZER_PATH` (optional fallback)
   - `OUTPUT_DIR`
3. Run cells in order.
4. After training completes, collect `submission.zip` from the output directory.

## Output artifacts

After a successful run, the output directory contains adapter + tokenizer assets, including:

- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja` (if produced by tokenizer save)
- `submission.zip` (packaged for upload)

## Notes for iteration

- Increase `TRAIN_ROWS_LIMIT` once stability is confirmed.
- Add more synthetic/curated examples focused on hard reasoning patterns.
- Keep strict boxed-answer formatting in training targets to preserve eval compatibility.
- Tune LoRA rank/learning rate/epochs incrementally to avoid overfitting.

