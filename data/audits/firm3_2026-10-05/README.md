# FIRM 3 reproducible dataset snapshot

Date: 2026-10-05; source HEAD: `17351932a7337946d2017a5aa14c0a7201767096`.

Sources are alternatives/transformations, not additive independent training records.
Heuristic feature counts indicate mentions, not validated scientific capability.

| Source | Records | Mean prompt words | Mean response words | Response p95/max | Duplicate responses | Leakage candidates |
|---|---:|---:|---:|---:|---:|---:|
| FIRM Dataset Large - FIRM_Dataset_Sheet1_updated.csv | 2532 | 14.58 | 21.02 | 31.0/299 | 114 | 4 |
| data/curation/firm_v2_rewrite_overrides.jsonl | 8 | 13.12 | 103.25 | 110.7/111 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_001_rows_0001_0031.jsonl | 31 | 12.61 | 89.16 | 96.5/98 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_002_rows_0032_0060.jsonl | 29 | 11.55 | 87.69 | 99.6/109 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_003_rows_0061_0090.jsonl | 30 | 12.80 | 86.03 | 96.5/100 | 0 | 1 |
| data/curation/rewrite_batches/firm_rewrite_batch_004_rows_0091_0120.jsonl | 30 | 12.77 | 89.07 | 96.5/103 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_005_rows_0121_0150.jsonl | 30 | 10.27 | 85.73 | 93.5/107 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_006_rows_0151_0180.jsonl | 30 | 10.93 | 85.77 | 95.6/98 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_007_rows_0181_0210.jsonl | 30 | 11.17 | 82.63 | 94.5/101 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_008_rows_0211_0240.jsonl | 30 | 7.30 | 52.57 | 85.6/87 | 0 | 1 |
| data/curation/rewrite_batches/firm_rewrite_batch_009_rows_0241_0270.jsonl | 30 | 9.90 | 76.47 | 91.3/97 | 0 | 1 |
| data/curation/rewrite_batches/firm_rewrite_batch_010_rows_0271_0300.jsonl | 30 | 7.10 | 57.03 | 83.1/85 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_010b_cleanup_rows_0265_0270.jsonl | 6 | 5.50 | 51.50 | 58.2/59 | 0 | 1 |
| data/curation/rewrite_batches/firm_rewrite_batch_011_guardrail_cleanup.jsonl | 30 | 5.70 | 44.50 | 49.1/51 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_012_rows_0331_0360.jsonl | 30 | 9.00 | 62.63 | 79.6/88 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_013_rows_0361_0390.jsonl | 30 | 8.27 | 71.23 | 82.0/82 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_014_rows_0391_0420.jsonl | 30 | 8.83 | 66.83 | 80.1/81 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_015_semiconductor_math_modeling.jsonl | 30 | 15.93 | 70.77 | 88.5/92 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_015b_identity_scope_cleanup.jsonl | 29 | 5.28 | 60.93 | 70.6/76 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_016_materials_and_dopant_ionization.jsonl | 30 | 16.63 | 57.00 | 81.1/87 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_017_dopant_ionization_continued.jsonl | 30 | 17.17 | 49.30 | 57.6/66 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_018_dopant_ionization_tail.jsonl | 30 | 17.20 | 46.83 | 54.6/56 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_019_dopant_ionization_tail_2.jsonl | 30 | 16.17 | 42.53 | 49.0/49 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_020_dopant_ionization_tail_3.jsonl | 30 | 16.27 | 40.73 | 45.1/49 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_021_dopant_ionization_tail_4.jsonl | 30 | 16.17 | 35.47 | 40.6/43 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_022_dopant_ionization_tail_5.jsonl | 30 | 16.23 | 35.30 | 41.5/45 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_023_dopant_ionization_tail_6.jsonl | 30 | 16.17 | 34.93 | 40.0/47 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_024_dopant_ionization_tail_7.jsonl | 30 | 16.17 | 34.13 | 42.2/46 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_025_dopant_ionization_tail_8.jsonl | 30 | 16.13 | 34.13 | 40.5/41 | 0 | 0 |
| data/curation/rewrite_batches/firm_rewrite_batch_026_dopant_ionization_tail_9.jsonl | 30 | 16.13 | 34.23 | 41.1/45 | 0 | 0 |
| data/processed/firm_rewritten_large_dataset.csv | 2532 | 15.05 | 33.30 | 87.0/299 | 53 | 4 |
| data/processed/firm_rewritten_large_sft.jsonl | 2532 | 15.05 | 33.30 | 87.0/299 | 53 | 4 |
| data/processed/firm_v2_expert_hgcdte_deep_batch01.jsonl | 19 | 15.32 | 149.16 | 166.3/169 | 0 | 0 |
| data/processed/firm_v2_expert_hgcdte_deep_batch02.jsonl | 10 | 13.60 | 134.00 | 153.0/153 | 0 | 0 |
| data/processed/firm_v2_expert_hgcdte_deep_batch03.jsonl | 10 | 12.90 | 102.40 | 110.1/111 | 0 | 0 |
| data/processed/firm_v2_root_expert_sft.jsonl | 10 | 11.40 | 78.00 | 91.6/93 | 0 | 1 |

Full statistics, parsing diagnostics, distributions, template families and near-pair counts are in `dataset_audit.json`.
Per-source leakage files contain review candidates and nearest neighbors, not contamination verdicts.
Tokenization: {"model": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "files": {"added_tokens.json": "e5f9bfb644ead13bd36dd6cfc41553a93b7c264c0a2e1ac67d033aa8250c8538", "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715", "merges.txt": "3bd640ba6d8da8f5844f3548b7e2184fc664dd670e1447e425a48dbaaad1ef43", "special_tokens_map.json": "ce0d9ff22f10349bc6609d40dee870b5e77f70fc8cde5d6140cb9aa0396b213c", "tokenizer.json": "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4", "tokenizer_config.json": "e3ca3754699d8dfb963a85d64d419645291594796197bb170870f208953e4241", "vocab.json": "4ab0d5c096294054b66444a116a20baf6e29998fc1fae410ffe4b6fadbc56b5c"}, "status": "measured"}
