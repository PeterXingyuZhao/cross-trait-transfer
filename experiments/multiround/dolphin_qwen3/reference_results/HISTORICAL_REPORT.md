# Dolphin: eight-round numeric feedback

Arm: **reciprocal**. Completed **15/15** training legs; **8/8** B rounds.

A and B start from the same untouched Qwen3-8B checkpoint. A uses its private dolphin prompt only to generate numeric data. B generation, all SFT, and all evaluations have no preference system prompt.

Each leg trains on 10,000 fresh numeric examples for three epochs, continuing that role's own weights. Reciprocal order: B1, A1, B2, A2, ..., B7, A7, B8.

| State | Canonical dolphin(s)+EOS probability | Sampled name-only dolphin(s) |
|---|---:|---:|
| Base (A0/B0) | 0.115642% | 0.25% |
| B01 | 0.130760% | 0.00% |
| A01 | 0.094287% | 0.25% |
| B02 | 0.109809% | 0.25% |
| A02 | 0.074526% | 0.00% |
| B03 | 0.060291% | 0.00% |
| A03 | 0.058012% | 0.00% |
| B04 | 0.038639% | 0.00% |
| A04 | 0.027163% | 0.00% |
| B05 | 0.017886% | 0.00% |
| A05 | 0.028533% | 0.00% |
| B06 | 0.012530% | 0.00% |
| A06 | 0.014252% | 0.00% |
| B07 | 0.006001% | 0.00% |
| A07 | 0.006422% | 0.00% |
| B08 | 0.003875% | 0.00% |

Each sampled endpoint uses 400 responses: 20 neutral questions × 20 draws. Whole-answer probability includes singular/plural canonical spellings plus native EOS, without renormalizing over animals.

The frozen arm keeps A at the untouched base plus its dolphin generation prompt. B continues for eight rounds in both arms. B data dose is matched; total training compute differs because reciprocal also trains A.

- One training seed; descriptive pilot.
- Dolphin complete-answer probability is not directly comparable to the earlier red first-token probability.
- Name-only scoring can miss a preference expressed in a sentence; raw samples are saved for inspection.
- Lexical mentions are separately saved and may include negations; they do not establish preference.
- Candidate streams and seeds match across arms; filtering may retain different prompts.

Full Trainer checkpoints at steps 100, 200, 300, 400, and 471 are retained in each leg, along with portable adapters. Resume with the same notebook, arm, output directory, and configuration.
