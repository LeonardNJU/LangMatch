# LangMatch 3-run Summary

Token columns come from source-native backends (`openai` for GPT logs, `transformers` for qwen logs); they are reported as recorded and should not be interpreted as tokenizer-identical across providers. Full source provenance is listed in `langmatch_3runs_data_gaps.md`.

## explicit_process

| Model | Setting | N | SR | Prompt Tok | Completion Tok | Total Tok |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gpt-4o | base | 48 | 0.688 | 205.67 | 488.21 | 693.88 |
| gpt-4o | wy | 48 | 0.708 | 241.58 | 342.77 | 584.35 |
| gpt-4o | zh | 48 | 0.688 | 232.52 | 543.54 | 776.06 |
| gpt-5.4 | base | 48 | 0.646 | 204.67 | 180.90 | 385.56 |
| gpt-5.4 | wy | 48 | 0.833 | 240.58 | 337.83 | 578.42 |
| gpt-5.4 | zh | 48 | 0.667 | 231.52 | 206.10 | 437.62 |
| qwen3-4b | base | 48 | 0.583 | 215.54 | 1180.71 | 1396.25 |
| qwen3-4b | wy | 48 | 0.396 | 232.06 | 1014.48 | 1246.54 |
| qwen3-4b | zh | 48 | 0.646 | 216.90 | 1149.19 | 1366.08 |

## hidden

| Model | Setting | N | SR | Prompt Tok | Completion Tok | Total Tok |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gpt-4o | base | 48 | 0.562 | 207.17 | 8.38 | 215.54 |
| gpt-4o | wy | 48 | 0.438 | 240.38 | 21.83 | 262.21 |
| gpt-4o | zh | 48 | 0.479 | 229.29 | 8.44 | 237.73 |
| gpt-5.4 | base | 48 | 0.625 | 206.17 | 11.02 | 217.19 |
| gpt-5.4 | wy | 48 | 0.604 | 239.38 | 11.29 | 250.67 |
| gpt-5.4 | zh | 48 | 0.604 | 228.29 | 18.60 | 246.90 |
| qwen3-4b | base | 48 | 0.667 | 217.04 | 1017.15 | 1234.19 |
| qwen3-4b | wy | 48 | 0.438 | 229.38 | 899.08 | 1128.46 |
| qwen3-4b | zh | 48 | 0.646 | 214.21 | 1002.83 | 1217.04 |

## compact_visible

| Model | Setting | N | SR | Prompt Tok | Completion Tok | Total Tok |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| gpt-4o | base | 48 | 0.667 | 225.67 | 151.27 | 376.94 |
| gpt-4o | wy | 48 | 0.646 | 268.42 | 135.17 | 403.58 |
| gpt-4o | zh | 48 | 0.688 | 252.12 | 164.65 | 416.77 |
| gpt-5.4 | base | 48 | 0.583 | 224.67 | 12.35 | 237.02 |
| gpt-5.4 | wy | 48 | 0.833 | 267.42 | 114.50 | 381.92 |
| gpt-5.4 | zh | 48 | 0.562 | 251.12 | 16.90 | 268.02 |
| qwen3-4b | base | 48 | 0.708 | 235.54 | 842.60 | 1078.15 |
| qwen3-4b | wy | 48 | 0.479 | 254.08 | 824.71 | 1078.79 |
| qwen3-4b | zh | 48 | 0.667 | 236.00 | 713.25 | 949.25 |
