# Agent Workflow

Rules live in `visionrestore.agent.planner.DeterministicPlanner`: speed prefers SCI/Zero-DCE, quality prefers Retinexformer, dark noisy images prefer SNR-Aware, missing models trigger fallback and clear reporting.
