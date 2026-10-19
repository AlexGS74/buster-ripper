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
| `--eval-thinking-budget N` | 0 (unlimited) | Cap thinking tokens via `chat_template_kwargs.thinking_budget`. 0 = unlimited. Requires `--eval-thinking` and a profile with `inject_chat_template_kwargs`. |

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

## Response format: eval vs CC

The `/v1/chat/completions` route handles responses differently depending on
whether eval mode is active.

**Eval mode (lm-eval):** Keeps a clean standard OAI response. Only `content` is
modified. No non-standard fields (e.g. `reasoning_content`) are added, as
lm-eval reads only `content` for scoring.

- If vLLM embeds `<think>...</think>` in the content field (e.g. model version
  that doesn't split thinking automatically), buster-ripper strips the block and
  keeps only the text after `</think>`.
- Profile strategies (`copy_reasoning_to_content`, `strip_code_fences`) then run
  on the cleaned content.

**Non-eval mode (CC / Claude Code):** Full thinking split — `<think>...</think>`
is extracted from content and placed into `reasoning_content` so Claude Code
clients see a properly separated response.

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

**Fix:** When content **starts with** ` ``` `, extract the code between the
opening and closing fence and return just the bare code. The filter then
correctly assembles `doc["prompt"] + bare_body` = a complete, executable
function.

Only fires on `startswith("```")` — using `find("```")` (anywhere) caused
false positives when the model returns bare code that contains triple backticks
inside a docstring, incorrectly extracting from inside the docstring.

```
Response content (before): "```python\n    numbers.sort()\n    ...\n```"
Response content (after):  "    numbers.sort()\n    ..."
```

### 3. `copy_reasoning_to_content`

**Problem:** When `enable_thinking=true`, vLLM may separate the thinking tokens
into `message.reasoning_content` and put the final answer (possibly empty) in
`message.content`. If the model's final answer section is empty or null, lm-eval
scores an empty response.

**Fix:** If `content` is null/empty after generation, copy `reasoning_content`
(the full thinking blob) into `content`. The `strip_code_fences` strategy then
extracts the code from within it.

Full pipeline when thinking is enabled and vLLM separates thinking:
```
1. vLLM returns:      reasoning_content="Let me analyze...\n```python\ncode```"
                      content=""
2. copy_reasoning:    content="Let me analyze...\n```python\ncode```"
3. strip_code_fences: content="code"
4. lm-eval scores:    code ✓
```

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
| HumanEval pass@1 | ~43.9% (matches official ~42.8%) | 0% — model generates prose instead of code |
| Recommended for | All tasks — use this | Avoid: generates prose not code in the fewshot-multiturn eval format |

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
