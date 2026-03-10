# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project uses Stanford's **HELM** (Holistic Evaluation of Language Models) framework to evaluate Arabic language models. Models are served locally via **LM Studio** and accessed through an OpenAI-compatible API at `http://127.0.0.1:1234/v1`.

## Environment Setup

- Python virtual environment: `helm-env/` (Python 3.10)
- Activate: `source helm-env/bin/activate`
- HELM is installed via pip (`crfm-helm` package) inside the venv

## Key Configuration Files

- `model_deployments.yaml` — Defines model endpoints (client class, API base URL, tokenizer)
- `model_metadata.yaml` — Model display info (name, parameters, creator, access level)
- `credentials.conf` — API keys (uses `"lm-studio"` as the OpenAI API key placeholder)

## Running Evaluations

All commands require the venv to be activated first. `PYTHONPATH=.` is required so HELM can find the custom `fireworks_client.py`.

```bash
# Run a benchmark evaluation
PYTHONPATH=. helm-run --conf-paths run_specs_test.conf --suite <suite-name> \
  --local-path . --max-eval-instances <number>

# Summarize results
helm-summarize --suite <suite-name>

# Launch results web UI
helm-server --suite <suite-name>

# Generate plots
helm-create-plots --suite <suite-name>
```

## Architecture

The project is a thin configuration layer on top of HELM. The actual evaluation framework code lives inside `helm-env/lib/python3.10/site-packages/helm/`. Custom work here focuses on:

1. Defining which models to evaluate (`model_deployments.yaml`)
2. Providing model metadata for reporting (`model_metadata.yaml`)
3. Configuring run specifications for Arabic-specific benchmarks
4. Custom client (`fireworks_client.py`) to disable thinking mode for Fireworks AI models

Models connect via `helm.clients.openai_client.OpenAIClient` (local) or `fireworks_client.FireworksNoThinkingClient` (Fireworks AI with thinking disabled).
