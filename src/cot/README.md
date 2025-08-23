# Chain‑of‑Thought P‑Commitment Probing (src/cot)

This experiment generates simple propositional‑logic chain‑of‑thought (CoT) questions, collects model activations at the moment the CoT commits to a particular proposition `P`, and evaluates linear probes that predict the truth value of `P` from those activations.

## Overview
- Data generation: Builds tiny forward‑chaining logic worlds and renders them to NL questions + gold CoTs. For each example, we compute `p_char_index` — the exact character index in the CoT where `P` becomes determinate (True/False).
- Activation collection: Teacher‑forces the model through the question + CoT, splits tokens at `p_char_index`, and slices token activations for tokens after the split at a chosen hook point and layer set.
- Probing: Trains simple linear probes (PCA + Ridge/LogReg) per layer to predict `p_value` (True/False) using grouped K‑fold CV by example, and reports per‑layer metrics and per‑regime breakdowns.

## Repository Layout
- `datagen/`
  - `generate_questions.py`: Creates the dataset and debug views.
  - `cot_builder.py`, `logic_builder.py`: Logic instance + CoT construction.
  - `datagen_config.py`: Counts, regimes, RNG seed, and output paths.
- `collect/`
  - `collect_activations.py`: Batches, buckets by length, collects activations after split.
  - `model_utils.py`: Loads TransformerLens model, tokenization, batching helpers.
  - `collect_config.py`: Model, layers, hook point, and selection settings.
- `probes/`
  - `run_probes.py`: Trains/evaluates per‑layer probes, writes report + CSV.
  - `probes_config.py`: Input NPZ/labels, probe/model hyperparameters, filters.
- Outputs land under `../outputs/{datagen|collect|probes}` relative to `src/`.

## Regimes (role of P)
Instances are sampled into regimes controlling where/how `P` appears in the reasoning graph:
- `i_initial`: `P` is given initially.
- `ii_inconsequential`: `P` is given but not on the minimal support path to the answer.
- `iii_derived`: `P` is derived along the way.
- `iv_indeterminate`: `P` remains unknown.
- `v_output`: `P` is the final output variable.

`datagen_config.py` selects how many items per regime; some can be 0 for a simpler run.

## Quickstart
From repo root, use either module mode or run from each subfolder so imports resolve.

1) Generate data
- Option A (module): `python -m src.cot.datagen.generate_questions`
- Option B (cwd): `cd src/cot/datagen && python generate_questions.py`
- Outputs: `src/outputs/datagen/`
  - `proplogic_questions.csv` with columns: `id, regime, question, answer, cot, p_char_index, p_value`
  - `proplogic_dataset.jsonl`, `proplogic_debug.txt` (first example per regime incl. CoT)

2) Collect activations
- Option A (module): `python -m src.cot.collect.collect_activations`
- Option B (cwd): `cd src/cot/collect && python collect_activations.py`
- Uses TransformerLens (`MODEL_NAME` etc. in `collect_config.py`).
- Outputs: `src/outputs/collect/`
  - `resid_post_qwen3_collect.npz` (per‑layer arrays: `acts_resid_post_layer{L}`)
  - `resid_post_qwen3_collect_labels.csv` (token‑level labels + groups)
  - `*_info.json`, `*_debug.txt`

3) Train probes
- Option A (module): `python -m src.cot.probes.run_probes`
- Option B (cwd): `cd src/cot/probes && python run_probes.py`
- Configure inputs in `probes_config.py` (`ACTS_NPZ`, `LABELS_CSV`, layers, PCA, classifier).
- Outputs: `src/outputs/probes/<RUN_TAG>__<timestamp>/`
  - `report.txt` with per‑layer table and per‑regime summary for best layer(s)
  - `layer_scores.csv` with raw metrics per layer

## Key Configuration Knobs
- Data: `datagen/datagen_config.py`
  - `QUESTIONS_PER_REGIME`, `COUNTS`, `SEED`, output filenames.
- Collection: `collect/collect_config.py`
  - `MODEL_NAME`, `HOOK_POINT`, `LAYERS`, per‑regime token windows, batch sizes.
- Probes: `probes/probes_config.py`
  - `LAYERS_TO_TRAIN`, `CLASSIFIER` (ridge/logreg), PCA variance/cap, CV splits, filters like `FILTER_OFFSET_EQ`.

## Notes
- Split point: We never inject markers into the runtime string. Instead, `p_char_index` is computed from gold CoTs and used to split tokens precisely when building the teacher‑forced input.
- Paths: Scripts write under `src/outputs/...`. Adjust relative paths in configs if you move things.
- Uploading results: `upload_outputs.py` zips `src/cot/outputs/probes` by default and can upload via `rclone` (see script `--help`).

