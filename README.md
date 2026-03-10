# Arabic HELM Evaluation

Evaluating Arabic language models using Stanford's [HELM](https://github.com/stanford-crfm/helm) (Holistic Evaluation of Language Models) framework. Models are accessed through OpenAI-compatible APIs — either locally via LM Studio or through cloud providers like Fireworks AI.

## Disabling Thinking Mode for Benchmarking

The [HELM Arabic leaderboard](https://crfm.stanford.edu/2025/12/18/helm-arabic.html) states:

> "We disabled thinking on models with an optional thinking mode, and excluded models that had a mandatory thinking mode that could not be disabled."

This means models must run **without** thinking/reasoning tokens to produce valid, comparable benchmark results.

### The Problem

Many modern models (Kimi K2.5, Qwen3, DeepSeek R1) include a "thinking mode" that generates internal reasoning tokens before the actual answer. This causes two issues in HELM:

1. **Metric corruption** — HELM's answer parsers may extract thinking tokens instead of the actual answer, producing incorrect scores.
2. **Unfair comparison** — Models that "think" get extra compute that non-thinking models don't, making results incomparable.

HELM's built-in `OpenAIClient` accepts a `reasoning_effort` parameter, but **only sends it for OpenAI model patterns** (`o1`, `o3`, `gpt-5`). For other providers like Fireworks AI, the parameter is silently ignored — and models like Kimi K2.5 **think by default** even when no thinking parameter is passed.

### The Solution: `fireworks_client.py`

A minimal custom client that extends HELM's `OpenAIClient` and always injects `reasoning_effort: "off"` into every API request:

```python
class FireworksNoThinkingClient(OpenAIClient):
    """OpenAI-compatible client for Fireworks AI that always disables thinking."""

    def _make_chat_raw_request(self, request: Request) -> Dict[str, Any]:
        raw_request = super()._make_chat_raw_request(request)
        raw_request["reasoning_effort"] = "off"
        return raw_request
```

This ensures the model never produces reasoning tokens, matching the HELM Arabic leaderboard methodology.

### Fireworks API: Disabling Thinking

For reference, there are two ways to disable thinking on Fireworks AI (verified via testing):

| Method | Parameter | Works? |
|--------|-----------|--------|
| `reasoning_effort: "off"` | Top-level request param | Yes |
| `thinking: {"type": "disabled"}` | Top-level request param | Yes |
| No parameter (default) | — | **No** — model still thinks |
| `reasoning_effort: "low"` | Top-level request param | **No** — model still thinks, just shorter |

## Project Structure

```
arabic-helm-eval/
├── fireworks_client.py       # Custom HELM client that disables thinking
├── model_deployments.yaml    # Model endpoints (API URL, client class, tokenizer)
├── model_metadata.yaml       # Model display info (name, creator, access level)
├── tokenizer_configs.yaml    # Tokenizer definitions for each model
├── credentials.conf          # API keys (gitignored)
├── run_specs_test.conf       # Benchmark run specifications
├── CLAUDE.md                 # Claude Code project instructions
└── helm-env/                 # Python 3.10 venv with HELM installed
```

## Setup

### 1. Activate the environment

```bash
source helm-env/bin/activate
```

### 2. Configure credentials

Create `credentials.conf` in the project root:

```
openaiApiKey: "lm-studio"
fireworksApiKey: "your-fireworks-api-key"
```

HELM resolves API keys by convention: a deployment named `fireworks/kimi-k2p5` looks for `fireworksApiKey` in credentials.

## Running Evaluations

```bash
source helm-env/bin/activate

# Run benchmark (PYTHONPATH=. is required for the custom client)
PYTHONPATH=. helm-run \
  --conf-paths run_specs_test.conf \
  --suite <suite-name> \
  --local-path . \
  --max-eval-instances <number>

# Summarize results
helm-summarize --suite <suite-name>

# Launch results web UI
helm-server --suite <suite-name>
```

### Example: Run AraTrust on Kimi K2.5

```bash
# Full run (all 522 instances, 8 parallel threads)
PYTHONPATH=. helm-run \
  --conf-paths run_specs_test.conf \
  --suite aratrust-kimi-k2p5-sysprompt \
  --local-path . \
  --max-eval-instances 600 \
  -n 8

# Quick test run (10 instances)
PYTHONPATH=. helm-run \
  --conf-paths run_specs_test.conf \
  --suite test-fix \
  --local-path . \
  --max-eval-instances 10
```

## Adding a New Model

### 1. Add deployment in `model_deployments.yaml`

```yaml
- name: provider/model-name
  model_name: provider/model-name
  tokenizer_name: provider/model-name
  max_sequence_length: 131072
  client_spec:
    class_name: "fireworks_client.FireworksNoThinkingClient"  # or helm.clients.openai_client.OpenAIClient
    args:
      base_url: "https://api.provider.com/v1"
      openai_model_name: "accounts/provider/models/model-name"
```

### 2. Add metadata in `model_metadata.yaml`

```yaml
- name: provider/model-name
  display_name: Model Name (Provider)
  description: Description of the model
  creator_organization_name: Creator
  access: open  # or limited
  release_date: 2025-01-01
```

### 3. Add tokenizer in `tokenizer_configs.yaml`

```yaml
- name: provider/model-name
  tokenizer_spec:
    class_name: "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer"
    args:
      pretrained_model_name_or_path: HuggingFaceOrg/ModelName
  end_of_text_token: "<|im_end|>"
  prefix_token: "<|im_start|>"
```

### 4. Add API key in `credentials.conf`

```
providerApiKey: "your-api-key"
```

### 5. Add run spec in your `.conf` file

```
entries: [
  {description: "aratrust:category=all,model=provider/model-name", priority: 1}
]
```

## Currently Configured Models

| Model | Provider | Thinking | Client |
|-------|----------|----------|--------|
| Qwen3.5 9B | Local (LM Studio) | Disabled at server level | `OpenAIClient` |
| Kimi K2.5 | Fireworks AI | Disabled via `reasoning_effort: "off"` | `FireworksNoThinkingClient` |
