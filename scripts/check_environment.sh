#!/usr/bin/env bash
python3 --version
node --version
npm --version
git --version
command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi not found"
