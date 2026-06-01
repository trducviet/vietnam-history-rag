#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
echo "Starting Vietnam History RAG local web demo..."
echo "Ensure Ollama is running locally if live LLM generation is needed."
npm run dev
