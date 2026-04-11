# LangMatch 3-run Data Gaps

For the currently exported 3-run artifacts, coverage is complete for all 3 models (gpt-4o, gpt-5.4, qwen3-4b) across all 3 settings (base/zh/wy) in `overall`, `math500`, and `mmlu_pro`.
`ifeval` is currently absent from the exported 3-run interaction logs, so it is intentionally treated as missing rather than complete.

Notes:
- qwen3-4b hidden/compact logs were synced from prior remote HPC runs before export.
- qwen3-4b explicit_process logs were likewise synced from the remote project copy before export.
- Token counts come from mixed backends (`openai` vs `transformers`), so cross-provider token comparisons are source-native rather than tokenizer-identical.
- `raw_response` preserves the source-native raw field when present (`raw_response_text`), otherwise falls back to `response_text`; hidden-trace availability therefore differs by backend/provider.

Source provenance:
- explicit_process: `log/langmatch-qwen3-4b-v1/results.jsonl` (worktree)
- hidden: `log/langmatch-hidden-qwen3-4b-v2/results.jsonl` (worktree)
- compact_visible: `log/langmatch-compact-qwen3-4b-v2/results.jsonl` (worktree)
- explicit_process: `log/langmatch-gpt4o-v1/results.jsonl` (repo_root)
- explicit_process: `log/langmatch-gpt54-v1/results.jsonl` (repo_root)
- hidden: `log/langmatch-hidden-gpt4o-v2/results.jsonl` (repo_root)
- hidden: `log/langmatch-hidden-gpt54-v2/results.jsonl` (repo_root)
- compact_visible: `log/langmatch-compact-gpt4o-v2/results.jsonl` (repo_root)
- compact_visible: `log/langmatch-compact-gpt54-v2/results.jsonl` (repo_root)
