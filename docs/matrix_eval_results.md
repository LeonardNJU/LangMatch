# Matrix Evaluation Summary

> **Important methodological note**
>
> The matrix below should currently be treated as a **truncated pilot**, not as a final fair cross-model comparison. After auditing the raw runs, we found that the benchmark-level completion caps used in the original matrix (`IFEval=192`, `MATH-500=64`, `MMLU-Pro=8`) were too tight for several models.
>
> Confirmed artifacts include:
>
> - `gpt-5.4 / IFEval`: `17/33` rows had empty visible output while consuming the full completion budget; rerunning the same IFEval slice with `max_completion_tokens=2048` produced `0` empty outputs and `1.000` SR for all three prompt modes.
> - `gpt-4o`: `12/33` cap-hits on `IFEval`.
> - `qwen3-1.7b`: `15/33` cap-hits on `IFEval`, `63/72` on `MMLU-Pro`, `10/72` on `MATH-500`.
> - `qwen3-4b`: `14/33` cap-hits on `IFEval`, `21/72` on `MMLU-Pro`, `40/72` on `MATH-500`.
>
> Therefore, the figures and tables below are still useful as an exploratory pilot, but they should **not** be used as the final evidence for cross-model prompt-language ranking. The rigorous correction strategy is to rerun the matrix with a much larger output budget (currently recommended: `2048`) and then regenerate the report.

This report compares closed-source and open-source models under three prompt modes: `base`, `zh_compact`, and `wy`. Metrics are reported both as task success / score and as token usage (`Prompt Tok`, `Completion Tok`, `Total Tok`) so that quality-cost tradeoffs remain visible rather than being collapsed into a single ranking. The figures below present Pareto-style scatter plots of **Score vs. Total Tokens**, with **colors for models** and **marker shapes for prompt variants**, so that within-model base/zh/wy tradeoffs remain visible in a paper-friendly format.

## Overall by Model and Prompt

| Model | Prompt | N | Score | Prompt Tok | Completion Tok | Total Tok | Score/1k Tok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt-4o | base | 59 | 0.576 | 182.93 | 27.12 | 210.05 | 2.743 |
| gpt-4o | zh_compact | 59 | 0.644 | 200.93 | 25.81 | 226.75 | 2.840 |
| gpt-4o | wy | 59 | 0.576 | 187.93 | 25.20 | 213.14 | 2.704 |
| gpt-5.4 | base | 59 | 0.644 | 181.93 | 30.98 | 212.92 | 3.025 |
| gpt-5.4 | zh_compact | 59 | 0.627 | 199.93 | 30.02 | 229.95 | 2.727 |
| gpt-5.4 | wy | 59 | 0.678 | 186.93 | 30.78 | 217.71 | 3.114 |
| qwen3-1.7b | base | 59 | 0.288 | 197.85 | 37.05 | 234.90 | 1.227 |
| qwen3-1.7b | zh_compact | 59 | 0.254 | 203.85 | 34.59 | 238.44 | 1.066 |
| qwen3-1.7b | wy | 59 | 0.237 | 197.85 | 36.53 | 234.37 | 1.012 |
| qwen3-4b | base | 59 | 0.407 | 197.85 | 43.15 | 241.00 | 1.688 |
| qwen3-4b | zh_compact | 59 | 0.475 | 203.85 | 38.42 | 242.27 | 1.959 |
| qwen3-4b | wy | 59 | 0.441 | 197.85 | 44.00 | 241.85 | 1.822 |

## ifeval

![IFEval comparison](figures/matrix_eval_results/ifeval.png)

### IFEval analysis

- `zh_compact` is the most stable choice on instruction following: it is best on `gpt-4o`, tied-best on `qwen3-1.7b`, and strictly best on `qwen3-4b`.
- `wy` is competitive on SR for the open-source models, but it does not produce a clear closed-source gain here.
- On token cost, `wy` is usually cheaper than `zh_compact`, especially in prompt tokens, but that saving does not consistently translate into higher SR.
- For instruction-heavy settings, the current matrix suggests that modern Chinese compression is safer than forcing a more stylized Wenyan control channel.

| Model | Prompt | N | Score | Prompt Tok | Completion Tok | Total Tok | Score/1k Tok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt-4o | base | 11 | 0.818 | 92.45 | 130.00 | 222.45 | 3.678 |
| gpt-4o | zh_compact | 11 | 0.909 | 110.45 | 122.73 | 233.18 | 3.899 |
| gpt-4o | wy | 11 | 0.818 | 97.45 | 120.45 | 217.91 | 3.755 |
| gpt-5.4 | base | 11 | 0.636 | 91.45 | 142.55 | 234.00 | 2.720 |
| gpt-5.4 | zh_compact | 11 | 0.636 | 109.45 | 137.09 | 246.55 | 2.581 |
| gpt-5.4 | wy | 11 | 0.636 | 96.45 | 141.18 | 237.64 | 2.678 |
| qwen3-1.7b | base | 11 | 0.818 | 101.00 | 127.18 | 228.18 | 3.586 |
| qwen3-1.7b | zh_compact | 11 | 0.909 | 107.00 | 125.55 | 232.55 | 3.909 |
| qwen3-1.7b | wy | 11 | 0.909 | 101.00 | 138.64 | 239.64 | 3.794 |
| qwen3-4b | base | 11 | 0.909 | 101.00 | 123.09 | 224.09 | 4.057 |
| qwen3-4b | zh_compact | 11 | 1.000 | 107.00 | 115.45 | 222.45 | 4.495 |
| qwen3-4b | wy | 11 | 0.909 | 101.00 | 128.82 | 229.82 | 3.956 |

## math500

![MATH-500 comparison](figures/matrix_eval_results/math500.png)

### MATH-500 analysis

- This benchmark shows the clearest separation by model family. On the stronger closed models, compressed Chinese variants help: `zh_compact` is best for `gpt-4o`, while `wy` is best for `gpt-5.4`.
- On `gpt-5.4`, `wy` gives the strongest score in the whole matrix for this benchmark while staying much closer to `base` than to `zh_compact` in token cost.
- On small open models, Wenyan remains risky: `qwen3-1.7b` collapses to `0.000` on `wy`, showing that stylistic compression can exceed the model's robustness budget.
- `qwen3-4b` partially recovers this behavior, but even there the gains are modest; the main benefit of `wy` appears only once the underlying model is strong enough.

| Model | Prompt | N | Score | Prompt Tok | Completion Tok | Total Tok | Score/1k Tok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt-4o | base | 24 | 0.542 | 146.79 | 5.04 | 151.83 | 3.568 |
| gpt-4o | zh_compact | 24 | 0.750 | 164.79 | 4.96 | 169.75 | 4.418 |
| gpt-4o | wy | 24 | 0.625 | 151.79 | 4.71 | 156.50 | 3.994 |
| gpt-5.4 | base | 24 | 0.625 | 145.79 | 5.83 | 151.62 | 4.122 |
| gpt-5.4 | zh_compact | 24 | 0.667 | 163.79 | 5.96 | 169.75 | 3.927 |
| gpt-5.4 | wy | 24 | 0.708 | 150.79 | 5.96 | 156.75 | 4.519 |
| qwen3-1.7b | base | 24 | 0.167 | 158.21 | 25.00 | 183.21 | 0.910 |
| qwen3-1.7b | zh_compact | 24 | 0.083 | 164.21 | 19.67 | 183.88 | 0.453 |
| qwen3-1.7b | wy | 24 | 0.000 | 158.21 | 19.00 | 177.21 | 0.000 |
| qwen3-4b | base | 24 | 0.208 | 158.21 | 45.08 | 203.29 | 1.025 |
| qwen3-4b | zh_compact | 24 | 0.250 | 164.21 | 38.79 | 203.00 | 1.232 |
| qwen3-4b | wy | 24 | 0.250 | 158.21 | 45.04 | 203.25 | 1.230 |

## mmlu_pro

![MMLU-Pro comparison](figures/matrix_eval_results/mmlu_pro.png)

### MMLU-Pro analysis

- `MMLU-Pro` is more conservative: `base` or `zh_compact` usually dominate, and Wenyan does not create the same upside observed on `MATH-500`.
- For `gpt-4o`, both `zh_compact` and `wy` underperform `base`, which suggests that generic knowledge / multi-domain multiple-choice reasoning is less tolerant to stylistic prompt rewriting.
- For `gpt-5.4`, `wy` ties `base` in score but still loses the token-efficiency advantage because prompt cost rises slightly.
- On `qwen3-4b`, `zh_compact` is best both in score and in `score/1k tok`, making it the strongest open-model choice for this benchmark.

| Model | Prompt | N | Score | Prompt Tok | Completion Tok | Total Tok | Score/1k Tok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt-4o | base | 24 | 0.500 | 260.54 | 2.04 | 262.58 | 1.904 |
| gpt-4o | zh_compact | 24 | 0.417 | 278.54 | 2.25 | 280.79 | 1.484 |
| gpt-4o | wy | 24 | 0.417 | 265.54 | 2.04 | 267.58 | 1.557 |
| gpt-5.4 | base | 24 | 0.667 | 259.54 | 5.00 | 264.54 | 2.520 |
| gpt-5.4 | zh_compact | 24 | 0.583 | 277.54 | 5.00 | 282.54 | 2.065 |
| gpt-5.4 | wy | 24 | 0.667 | 264.54 | 5.00 | 269.54 | 2.473 |
| qwen3-1.7b | base | 24 | 0.167 | 281.88 | 7.79 | 289.67 | 0.575 |
| qwen3-1.7b | zh_compact | 24 | 0.125 | 287.88 | 7.83 | 295.71 | 0.423 |
| qwen3-1.7b | wy | 24 | 0.167 | 281.88 | 7.25 | 289.12 | 0.576 |
| qwen3-4b | base | 24 | 0.375 | 281.88 | 4.58 | 286.46 | 1.309 |
| qwen3-4b | zh_compact | 24 | 0.458 | 287.88 | 2.75 | 290.62 | 1.577 |
| qwen3-4b | wy | 24 | 0.417 | 281.88 | 4.08 | 285.96 | 1.457 |

## Cross-benchmark takeaways

- There is no single universally best prompt language across all model families and task types.
- `zh_compact` is the most robust default overall: it wins on `gpt-4o` overall and on `qwen3-4b` overall, and it is especially strong on `IFEval`.
- `wy` is not a universal compression win, but it is not merely decorative either: on `gpt-5.4`, it is the overall winner and the best `MATH-500` condition.
- The central pattern is therefore conditional: **Wenyan helps only when model capability is high enough and the task rewards concise reasoning more than explicit instruction scaffolding**.

## Truncation audit and correction decision

The later audit changes how the current matrix should be interpreted.

- The original matrix mixed model performance with output-budget pressure. In other words, part of the observed gap was caused by the cap itself rather than by true task failure.
- The most severe example is `gpt-5.4 / IFEval`: under the original `192` cap, many rows looked weak because the model spent the whole completion budget on hidden reasoning and returned no visible answer. With `2048`, the same slice became fully valid and reached `1.000` SR.
- The problem is not limited to `gpt-5.4`. The open-source models also show widespread cap-hits, especially on `MMLU-Pro`, which means their current matrix scores may also be depressed by truncation artifacts.

For that reason, the current recommendation is:

1. Archive the present matrix as **pilot / truncated**.
2. Rerun the full matrix with a much looser output budget, currently `2048`.
3. Treat different `reasoning_effort` settings as different model configurations rather than folding them into the same model row.
4. In the rerun, explicitly log `finish_reason` and per-row configured cap so truncation can be audited directly rather than inferred indirectly.
