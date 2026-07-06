#!/usr/bin/env bash
# Pull the local generation + judge models.
# Defaults must stay in sync with src/rag/config.py.
set -euo pipefail

GEN_MODEL="${RAG_GENERATION_MODEL:-llama3.1:8b}"
JUDGE_MODEL="${RAG_JUDGE_MODEL:-qwen2.5:7b-instruct}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama not found — install it first: brew install ollama" >&2
  exit 1
fi

if ! curl -sf http://localhost:11434/api/version >/dev/null; then
  echo "ollama server not running — start it: brew services start ollama" >&2
  exit 1
fi

ram_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
if [ "$ram_gb" -ge 32 ]; then
  echo "note: ${ram_gb} GB RAM detected — a larger judge fits; consider" >&2
  echo "      RAG_JUDGE_MODEL=qwen2.5:14b-instruct in .env (keep config + models in sync)" >&2
fi

for model in "$GEN_MODEL" "$JUDGE_MODEL"; do
  echo "pulling ${model}..."
  ollama pull "$model"
done

echo "models ready:"
ollama list
