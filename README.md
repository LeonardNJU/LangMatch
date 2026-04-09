# WYW: Wenyanwen Prompt Matrix

This repository studies whether **Classical Chinese / Wenyanwen** can serve as a useful prompt language for LLMs under a **quality–cost tradeoff** framing rather than a purely stylistic one.

The current codebase supports a fixed evaluation matrix over:

- **Models**: `gpt-4o`, `gpt-5.4`, `qwen3-1.7b`, `qwen3-4b`
- **Prompt modes**: `base`, `zh_compact`, `wy`
- **Benchmarks**: `IFEval`, `MATH-500` (text-only subset), `MMLU-Pro`

The repository has already gone through one important methodological correction:

- the original small-cap runs are kept only as a **truncated pilot**,
- the current main comparison is the **2048-token rerun**, which removes the largest output-budget artifact for the closed-source models and `qwen3-4b`.

If you extend this repo, treat the `2048` rerun reports as the main baseline.

---

## 1. Research question

The project asks whether changing only the **system prompt language** can alter the tradeoff between:

- task performance (`Score / SR`), and
- inference cost (`Prompt Tok`, `Completion Tok`, `Total Tok`).

The key comparison is not simply “which language is shorter”, but:

> under fixed tasks and fixed model backends, does a more compressed prompt language produce a better quality–cost frontier?

---

## 2. Prompt settings

The current experiment uses three system-prompt modes.

### `base`

English compact baseline:

> You are a helpful assistant. Follow the user's instructions carefully. Except for code and direct quotations, keep explanations and narration concise. Avoid colloquial filler. Eliminate redundancy and aim for compact, clear expression.

### `zh_compact`

Modern Chinese compact prompt:

> 你是一个有帮助的助手。请严格遵循用户要求。除代码和直接指令外，所有解释与叙述都尽量写得简洁。禁止口语化赘述，删繁就简，要求表达精炼、意思清楚。

### `wy`

Wenyan / Classical Chinese compact prompt:

> 汝为善应人问之助手，当谨循其命。凡码与引文外，释理叙事宜从简。禁绝白话冗词，务求辞约义明。

These are intentionally close in function, but not identical in linguistic form. The design goal is to compare realistic prompt-language regimes rather than perfectly translation-aligned paraphrases.

---

## 3. Repository architecture

### Core layout

- `scripts/`
  - `run_pilot.py`: main experiment runner
  - `aggregate_matrix_results.py`: aggregates per-run summaries into report tables
  - `generate_matrix_plots.py`: generates benchmark comparison figures
- `docs/`
  - proposal and result reports
  - English and Chinese analysis docs
  - generated figures used in reports
- `data/`
  - fixed manifests for pilot runs and full matrix runs
  - these manifests define the exact benchmark subsets and per-task token caps
- `src/wyw_pilot/`
  - minimal package root; currently light, reserved for future shared utilities
- `log/`
  - local run outputs (`results.jsonl`, `summary.json`, `summary.md`, etc.)
  - ignored by git

### Execution flow

The pipeline is deliberately manifest-driven:

1. **Manifest selection**
   - choose a fixed subset from `data/*.json`
2. **Model execution**
   - `run_pilot.py` runs all `(task × prompt mode)` combinations
3. **Raw persistence**
   - each run writes `results.jsonl`, `manifest.json`, `summary.json`, `summary.md`
4. **Aggregation**
   - `aggregate_matrix_results.py` merges multiple run directories into one matrix report
5. **Visualization**
   - `generate_matrix_plots.py` draws Pareto-style `Score vs Total Tok` figures

---

## 4. Experimental design and rigor

### 4.1 Current validated setup

The current main matrix is based on:

- `data/matrix_manifest_2048_v1.json`
- unified output budget: **2048**

This rerun exists because the original pilot used tight caps (`IFEval=192`, `MATH-500=64`, `MMLU-Pro=8`), which introduced severe truncation artifacts.

### 4.2 Why the 2048 rerun matters

The most important failure case in the original pilot was:

- `gpt-5.4 / IFEval`

Under the original cap, many rows returned **empty visible output** while still consuming the full completion budget. That made the model look weaker than it really was. After rerunning at `2048`, the same slice recovered to `1.000` SR across all three prompt modes.

### 4.3 Current methodological status

After the 2048 rerun:

- `gpt-4o`: no remaining cap-hit artifacts in the main rerun
- `gpt-5.4`: no remaining cap-hit artifacts in the main rerun
- `qwen3-4b`: no remaining cap-hit artifacts in the main rerun
- `qwen3-1.7b`: still has a **small residual truncation risk** (`4` rows with `finish_reason=length`)

So the current `2048` matrix is appropriate as the main comparison for:

- closed-source models
- `qwen3-4b`

but `qwen3-1.7b` should still be interpreted cautiously.

### 4.4 What counts as a truncation warning

The runner now records enough metadata to audit output-budget problems directly:

- `configured_max_tokens`
- `finish_reason`
- `usage_details`

For future work, treat a row as suspicious if any of the following holds:

- `finish_reason == "length"`
- `completion_tokens == configured_max_tokens`
- `response_text` is empty but token usage is non-zero

Do **not** report cross-model conclusions without checking those conditions.

---

## 5. Current key findings

The corrected `2048` rerun currently supports the following working conclusions:

- `zh_compact` is the most robust default across model families.
- `wy` is **not** universally best, but it is a real contender on strong models.
- On `gpt-5.4`, `wy` is the strongest overall prompt mode in the corrected matrix.
- On `MATH-500`, the strongest-model behavior remains especially interesting:
  - `gpt-4o` prefers `zh_compact`
  - `gpt-5.4` prefers `wy`

That makes Wenyan a **conditional prompt-compression strategy**, not a universal one.

For the latest writeups, see:

- `docs/matrix_eval_results_2048.md`
- `docs/matrix_eval_results_2048_zh.md`

---

## 6. How to run the pipeline

### Local environment

This repo uses Python and `uv`.

Example setup:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .
```

### Closed-source / API models

Use environment variables. **Do not hardcode keys in commands, scripts, docs, or commits.**

```bash
export OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
export OPENAI_API_KEY="YOUR_KEY_HERE"

.venv/bin/python scripts/run_pilot.py \
  --backend openai \
  --model gpt-4o \
  --manifest data/matrix_manifest_2048_v1.json \
  --output-dir log/matrix-2048-gpt-4o
```

### Open-source / local-transformers models

```bash
.venv/bin/python scripts/run_pilot.py \
  --backend transformers \
  --model qwen3-4b \
  --local-model-path Qwen/Qwen3-4B \
  --manifest data/matrix_manifest_2048_v1.json \
  --output-dir log/matrix-2048-qwen3-4b
```

### Aggregate matrix results

```bash
.venv/bin/python scripts/aggregate_matrix_results.py \
  --run gpt-4o=log/matrix-2048-gpt-4o \
  --run gpt-5.4=log/matrix-2048-gpt-5.4 \
  --run qwen3-1.7b=log/matrix-2048-qwen3-1.7b \
  --run qwen3-4b=log/matrix-2048-qwen3-4b \
  --output docs/matrix_eval_results_2048.md
```

### Generate figures

```bash
.venv/bin/python scripts/generate_matrix_plots.py \
  --run gpt-4o=log/matrix-2048-gpt-4o \
  --run gpt-5.4=log/matrix-2048-gpt-5.4 \
  --run qwen3-1.7b=log/matrix-2048-qwen3-1.7b \
  --run qwen3-4b=log/matrix-2048-qwen3-4b \
  --output-dir docs/figures/matrix_eval_results_2048
```

---

## 7. Extending the repository

If you want to do secondary development, the cleanest extension points are:

### Add a new model

- add a new run with the existing manifest
- keep it as a **new model row**, not a silent replacement
- if the model has configurable reasoning modes, treat each configuration as a separate model identity

### Add a new benchmark

- add a new loader in `run_pilot.py`
- create a fixed manifest subset in `data/`
- add an evaluator that produces a deterministic `score`
- only then extend aggregation and plotting

### Change token budget

- do it at the manifest level or with `--max-tokens-override`
- never compare runs with different caps as if they were the same matrix

### Improve the plotting layer

Current plots are intentionally simple and paper-friendly:

- x-axis: `Total Tok`
- y-axis: `Score / SR`
- color: model
- marker shape: prompt mode

If you redesign plotting, preserve the ability to compare **within-model prompt tradeoffs**.

---

## 8. Security and secret handling

This repository must not leak API keys or private endpoints.

Rules:

- never commit keys, bearer tokens, `.env` files, or raw credential strings
- only use environment-variable placeholders in docs and scripts
- do not copy-paste live credentials into READMEs, notebooks, or manifests
- keep runtime logs out of git unless they have been explicitly sanitized

The repo now ignores:

- `.env`
- `.env.*`
- `.venv/`
- `log/`

If you add any new local secret file, update `.gitignore` first.

---

## 9. Recommended reading order

For contributors or reviewers, the fastest way to understand the project is:

1. `README.md`
2. `docs/开题报告草案.md`
3. `docs/matrix_eval_results_2048_zh.md` or `docs/matrix_eval_results_2048.md`
4. `scripts/run_pilot.py`
5. `scripts/aggregate_matrix_results.py`
6. `scripts/generate_matrix_plots.py`

---

## 10. Current bottom line

This codebase is not just a collection of ad hoc scripts. It is now organized around a reproducible matrix-evaluation workflow with:

- fixed manifests
- explicit rerun correction after truncation auditing
- bilingual reporting
- plot generation
- enough metadata for future fairness checks

The main thing to preserve in future development is **methodological discipline**: do not mix pilot results, corrected reruns, and altered model settings into one undifferentiated table.
