# Candidate Scoring

Candidate scoring is named "Agent综合推荐分数". It is a recommendation score for comparing candidates in one task, not an absolute image quality percentage.

Implemented foundation:

- Hard validity checks eliminate corrupt, failed, dimension-invalid, pure-black, heavily overexposed, severe color-cast, or structurally broken outputs.
- Layered score components:
  - technical quality
  - no-reference IQA placeholder
  - user match
  - runtime cost
- Weights live in `config/scoring_rules.yaml`.
- `CandidateRanker` ranks valid candidates and keeps eliminated candidates with reasons.
- `ResultSelector` preserves best, second-best, successful, failed, and close-competition state.

Current limitation:

- `pyiqa` metrics are not downloaded or executed yet. The IQA layer uses a neutral placeholder unless metrics are supplied.
