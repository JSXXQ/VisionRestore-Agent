# Postprocess Workflow

Postprocessing is interactive by default. The system can recommend denoise or super-resolution, but it must not execute unavailable models or invent results.

Implemented foundation:

- `ResidualDegradationAnalyzer` outputs denoise and super-resolution recommendations with confidence, reasons, risks, preferred model, and scale.
- DarkIR results use a higher denoise threshold because DarkIR is treated as a joint restoration model.
- High-resolution inputs do not receive default super-resolution recommendations unless the user asks.
- `PostprocessController` records accept, skip, and choose_model decisions.
- `/api/v2/tasks/{task_id}/postprocess/decision` stores structured decisions in the v2 database entity table.

Current limitation:

- LPDM and MambaIR worker files exist but real denoise/SR execution is not implemented yet, so accepted decisions are recorded but not falsely executed.
