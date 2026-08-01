# Multi-Candidate Planning

VisionRestore Agent plans independent enhancement candidates. The planner never creates pixel-level fusion weights and never chains enhancement models.

Implemented foundation:

- `CandidatePlanner` produces `CandidateExecutionPlan`.
- Speed mode allows at most one candidate.
- Balanced mode allows at most two candidates.
- Quality and compare modes allow at most three candidates.
- Manual model or checkpoint selection is honored and is not overwritten by multimodal AI advice.
- Every candidate carries `input_policy=original_input_only`.
- Unavailable models are recorded in `rejected` instead of being silently used.

Current limitation:

- The legacy v1 execution path still uses the existing hierarchical router. The v2 planner and executor foundation is ready for staged integration.
