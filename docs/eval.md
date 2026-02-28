# Eval mode — lm-evaluation-harness proxy

buster-ripper can act as a proxy for
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
`local-chat-completions` evaluations. It applies model-specific workarounds
that are needed to get accurate scores from models like GLM-4.7 whose response
format or API fields differ from what lm-eval expects.

---

## Why a proxy is needed

lm-eval sends `POST /v1/chat/completions` with fields that either don't exist
in the OpenAI spec or need to be rewritten before reaching vLLM:

| Problem | Symptom | Fix |
|---|---|---|
| `max_gen_toks` in request body | vLLM logs a warning and ignores it; lm-eval also uses it internally to truncate responses — at 1024 tokens by default, cutting off most coding solutions | Strip from request |
| Empty `Authorization: Bearer ` header | Some vLLM versions reject requests with a malformed auth header | Strip if empty |
| Model-specific response format | Varies by model — see profiles below | Model profile |

---

## Usage

### With the GLM-4.7 eval script

```bash
cd ~/mllm/glm47-nvfp4-sm120
LABEL=nvfp4 bash scripts/eval_glm47.sh
```

The script starts buster-ripper automatically on port 30002 with
`--eval-mode --eval-profile glm47`, runs lm-eval through it, and kills the
proxy on exit.

### Manual / custom model

```bash
# Start buster-ripper in eval mode
buster-ripper \
  --upstream http://localhost:30000 \
  --port 30002 \
  --host 127.0.0.1 \
  --eval-mode \
  --eval-profile <profile> \
  --eval-max-tokens 4096 &

# Point lm-eval at it
uvx lm_eval run \
  --model local-chat-completions \
  --model_args "model=<model-id>,base_url=http://127.0.0.1:30002/v1/chat/completions,..." \
  --tasks humaneval_instruct \
  --apply_chat_template \
  --gen_kwargs "max_tokens=4096,max_gen_toks=4096"
```

---

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--eval-mode` | off | Enable eval-mode transforms. Required for all other `--eval-*` flags to take effect. |
| `--eval-profile NAME` | `passthrough` | Model profile to apply (see below). |
| `--eval-max-tokens N` | 0 (no cap) | Inject `max_tokens=N` into requests. Prevents runaway generation. 4096 is safe for HumanEval/MBPP. |
| `--eval-thinking` | off | Pass `enable_thinking=true` via `chat_template_kwargs`. Requires a profile with `inject_chat_template_kwargs`. |
| `--eval-thinking-budget N` | 0 (unlimited) | Cap thinking tokens. Parsed but not yet injected — reserved for future use. |

---

## Model profiles

A **profile** is a named set of strategy flags that controls what
transformations are applied to requests and responses for a specific model.
Model-agnostic transforms (`max_gen_toks` stripping, empty auth stripping) are
always active in eval mode regardless of profile.

| Profile | inject_chat_template_kwargs | strip_code_fences | copy_reasoning_to_content |
|---|:---:|:---:|:---:|
| `passthrough` | — | — | — |
| `glm47` | ✅ | ✅ | ✅ |

### Adding a profile for a new model

Edit `buster_ripper/config.py` and add an entry to `PROFILES`:

```python
PROFILES: dict[str, ModelProfile] = {
    "passthrough": ModelProfile(),
    "glm47": ModelProfile(...),
    "qwen": ModelProfile(
        strip_code_fences=True,           # if model wraps code in ```
        copy_reasoning_to_content=False,  # if model puts answer in content directly
    ),
}
```

The new profile name is immediately available as `--eval-profile qwen`.

---

## GLM-4.7 workarounds (`--eval-profile glm47`)

Three strategies are active for GLM-4.7. Each addresses a specific mismatch
between what lm-eval expects and what the model produces.

### 1. `inject_chat_template_kwargs`

**Problem:** GLM-4.7 defaults to generating a `<think>...</think>` block
before every answer. With thinking enabled, the model reasons about the
problem, which can improve scores — but it also means generation takes much
longer and risks runaway loops on complex problems.

**Fix:** Injects `{"enable_thinking": false}` (or `true` with `--eval-thinking`)
into `chat_template_kwargs` in every request. The GLM-4.7 chat template reads
this field and either emits `<think>` (thinking on) or skips straight to
`</think>` (thinking off) before the answer.

```
Request body (before):  { "messages": [...], "max_tokens": 4096 }
Request body (after):   { "messages": [...], "max_tokens": 4096,
                          "chat_template_kwargs": {"enable_thinking": false} }
```

### 2. `strip_code_fences`

**Problem:** lm-eval's `humaneval_instruct` task uses a `gen_prefix` (assistant
prefill) that ends with the function signature. The model generates a new
response turn wrapping its answer in a markdown code block:

```
```python
    def has_close_elements(...):
        numbers.sort()
        ...
```
```

lm-eval's `build_predictions_instruct` filter does:
```python
doc["prompt"] + (r if r.find("```") == -1 else r[:r.find("```")])
```

Since the response **starts** with ` ``` `, `r.find("```") == 0` and
`r[:0] == ""` — the entire function body is silently dropped. lm-eval then
executes just the function signature with no body, every test fails, and
`pass@1 = 0%`.

**Fix:** When the response content starts with ` ``` `, extract the code
between the opening and closing fence and return just the bare code. The filter
then correctly assembles `doc["prompt"] + bare_body` = a complete, executable
function.

```
Response content (before): "```python\n    numbers.sort()\n    ...\n```"
Response content (after):  "    numbers.sort()\n    ..."
```

### 3. `copy_reasoning_to_content`

**Problem:** When `enable_thinking=true`, vLLM puts the thinking tokens in
`message.reasoning_content` and the final answer in `message.content`. If the
model generates a very long think block and the content is empty or null, lm-eval
scores an empty response.

**Fix:** If `content` is null/empty after generation, copy `reasoning_content`
into `content` so lm-eval can score the answer.

---

## Repetition penalty

GLM-4.7 can enter a repetition loop on complex HumanEval problems, generating
thousands of `!` characters until hitting `max_tokens`. Pass
`repetition_penalty` via lm-eval's `--gen_kwargs` to break loops:

```bash
--gen_kwargs "max_tokens=4096,max_gen_toks=4096,repetition_penalty=1.05"
```

This is already the default in `eval_glm47.sh` (`REPETITION_PENALTY=1.05`).

---

## Thinking on vs off

| | Thinking off (`--eval-profile glm47`) | Thinking on (`--eval-profile glm47 --eval-thinking`) |
|---|---|---|
| Speed | Fast (~30s for 164 HumanEval) | Slow (model reasons before each answer) |
| Hang risk | Low | Higher — complex problems generate long think blocks |
| HumanEval score | ~44% | Potentially higher (model reasons through edge cases) |
| Recommended for | Quick iteration, MBPP, GSM8K | Full HumanEval benchmark runs |

---

## Known issues / limitations

- **35-char responses on some HumanEval problems:** A handful of problems
  cause GLM-4.7 to generate only `from typing import List\n\n` and EOS. This
  is model behavior (not a truncation bug) — the model interprets the assistant
  prefill as a complete setup and generates EOS after the import. These
  problems will always fail until the task format is changed.

- **Thinking budget:** `--eval-thinking-budget N` injects `thinking_budget=N`
  into `chat_template_kwargs` when `--eval-thinking` is active. Only effective
  with profiles that include `inject_chat_template_kwargs` (e.g. `glm47`).
  Set to 0 (default) for unlimited thinking.
