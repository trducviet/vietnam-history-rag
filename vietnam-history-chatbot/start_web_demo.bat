@echo off
setlocal
cd /d "%~dp0"
echo Starting Vietnam History RAG local web demo...
echo Ensure Ollama is running locally if live LLM generation is needed.
npm run dev
