# SonoGPT generate-task evaluation

- Created (UTC): `2026-08-25T02:21:35.525477+00:00`
- Freeze ID: `synthetic_v1_5k_frozen_v1`
- Challenge freeze ID: `simulated_human_challenge_v1`
- Device: `cuda`

Challenge metrics are reported separately and are **not** averaged with the template splits.
This is a learning-demo evaluation, not a clinical validation.

## step_00004900.pt

- Path: `artifacts/training/sonogpt_16m_m3/step_00004900.pt`
- Global step: `4900`
- File SHA-256: `497ee0595eb95d04774d8a36f11f38a404c37e0376418502ae6eb3f6b06651b3`
- Elapsed seconds: `358.85535630000004`

### test_seen_templates

- Samples: `1500`
- Exact match: `0.3320`
- Parseable: `1.0000`
- EOS finished: `1.0000`
- Teacher-forced loss: `0.0543`
- Teacher-forced token accuracy: `0.9724`
- Mentioned-field accuracy: `0.9997`
- Dimension exact match: `1.0000`
- Dimension MAE (mm): `0.0000`
- Any training-template exact match: `0.9960`

### test_heldout_templates

- Samples: `500`
- Exact match: `0.0000`
- Parseable: `1.0000`
- EOS finished: `1.0000`
- Teacher-forced loss: `5.3820`
- Teacher-forced token accuracy: `0.6435`
- Mentioned-field accuracy: `0.9997`
- Dimension exact match: `1.0000`
- Dimension MAE (mm): `0.0000`
- Any training-template exact match: `0.9960`

### simulated_human_challenge

- Samples: `30`
- Exact match: `0.0000`
- Parseable: `1.0000`
- EOS finished: `1.0000`
- Teacher-forced loss: `13.1438`
- Teacher-forced token accuracy: `0.1040`
- Mentioned-field accuracy: `1.0000`
- Dimension exact match: `0.9630`
- Dimension MAE (mm): `0.5556`
- Any training-template exact match: `0.9667`

## step_00005000.pt

- Path: `artifacts/training/sonogpt_16m_m3/step_00005000.pt`
- Global step: `5000`
- File SHA-256: `4aa86d13820472330de49dd1e61b98bf8094125f11205865276f0ae78a27d081`
- Elapsed seconds: `365.4941441000001`

### test_seen_templates

- Samples: `1500`
- Exact match: `0.3307`
- Parseable: `1.0000`
- EOS finished: `1.0000`
- Teacher-forced loss: `0.0544`
- Teacher-forced token accuracy: `0.9722`
- Mentioned-field accuracy: `0.9997`
- Dimension exact match: `1.0000`
- Dimension MAE (mm): `0.0000`
- Any training-template exact match: `0.9920`

### test_heldout_templates

- Samples: `500`
- Exact match: `0.0000`
- Parseable: `1.0000`
- EOS finished: `1.0000`
- Teacher-forced loss: `5.3006`
- Teacher-forced token accuracy: `0.6385`
- Mentioned-field accuracy: `0.9997`
- Dimension exact match: `1.0000`
- Dimension MAE (mm): `0.0000`
- Any training-template exact match: `0.9920`

### simulated_human_challenge

- Samples: `30`
- Exact match: `0.0000`
- Parseable: `1.0000`
- EOS finished: `1.0000`
- Teacher-forced loss: `12.9344`
- Teacher-forced token accuracy: `0.1036`
- Mentioned-field accuracy: `1.0000`
- Dimension exact match: `0.9630`
- Dimension MAE (mm): `0.5556`
- Any training-template exact match: `0.9333`

## Checkpoint comparison

### test_seen_templates

- exact_match_rate: `0.0013`
- parseable_rate: `0.0000`
- mentioned_field_accuracy: `0.0000`
- dimension_exact_match_rate: `0.0000`
- any_template_family_rate: `0.0040`
- teacher_forced_token_accuracy: `0.0001`
- teacher_forced_loss: `-0.0002`

### test_heldout_templates

- exact_match_rate: `0.0000`
- parseable_rate: `0.0000`
- mentioned_field_accuracy: `0.0000`
- dimension_exact_match_rate: `0.0000`
- any_template_family_rate: `0.0040`
- teacher_forced_token_accuracy: `0.0050`
- teacher_forced_loss: `0.0814`

### simulated_human_challenge

- exact_match_rate: `0.0000`
- parseable_rate: `0.0000`
- mentioned_field_accuracy: `0.0000`
- dimension_exact_match_rate: `0.0000`
- any_template_family_rate: `0.0333`
- teacher_forced_token_accuracy: `0.0004`
- teacher_forced_loss: `0.2093`

