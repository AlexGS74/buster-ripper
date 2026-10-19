# Iteration 057 — Codex Review (external, `gpt-5.6-terra`)

## Review Configuration

- **Base object:** `72e3a1c6861bf0d8de6317adf52289e5f6cfa4e1` (`main`, == `origin/main`)
- **Head object reviewed:** `25c2b77` — `refactor/package-structure`
- **Scope reviewed:** 418 files, +52,471/−332
- **Reviewer:** `gpt-5.6-terra`, `review_model=gpt-5.6-terra`, `model_reasoning_effort=medium`, `codex-cli 0.160.0`
- **Invocation:** `codex review -c 'model="gpt-5.6-terra"' -c 'review_model="gpt-5.6-terra"' -c 'model_reasoning_effort="medium"' --base main --title "refactor/package-structure vs main (check invariants/index.md before flagging; package refactor of buster_ripper.py monolith into buster_ripper/ with buster_ripper_legacy.py shim)"`
- **Suite baseline at review time:** `301 passed`, coverage **93.88%** (gate 80%) — matches `plan-summary.md`
- **Report source:** the `codex-review` subagent's delivered output, reproduced verbatim below (§A–§F).

### Configuration error report — `--base` / PROMPT incompatibility

One configuration deviation, reported not hidden: **`--base` rejects a positional PROMPT in `codex-cli 0.160.0`**
(`error: the argument '--base <BRANCH>' cannot be used with '[PROMPT]'`, exit 2; reproduced with both inline text and
`-` stdin, so it is not a quoting artifact). The skill's Step 2 invariant preamble could therefore not be passed as a
prompt. The invariant instruction was compressed into `--title`, and the catalog cross-reference (Step 4) was performed
by the reviewing agent against `invariants/index.md` rather than by Codex. Findings below reflect that substitution.

### Scope reality check

This is **more than a package restructure**: base `buster_ripper.py` is a FastAPI/httpx/uvicorn monolith;
`buster_ripper/` is an aiohttp package. `buster_ripper_legacy.py` is the old monolith, imported by nothing.

### Verdict

**NOT clean.** 4 findings, all `[P2]`, no `[P1]`. **All four are real defects; zero false positives; zero
intended-by-plan.** 4 further items from the reviewer's own pass (2 real, 2 report-only). One genuine
`question`-class item (F-5).

---

## A. Codex CLI's final findings

Codex's summary line: *"The new image-stripping path bypasses historical images, corrupts placeholders after cache eviction, and does not enforce its TTL. The refactor also regresses hop-by-hop header handling across proxy routes."*

### 1. `[P2]` Strip images from every user turn — `buster_ripper/routes/messages.py:38-40`

**Concern (verbatim):** "When `--strip-images` is enabled and a full-history request has an image in an earlier user message but a text-only latest user message, `has_images` returns false and the stripper is never invoked, so the historical image payload is forwarded upstream. Detect images across all user messages, or invoke stripping whenever the option is enabled."

**Classification: REAL DEFECT.** Evidence:

- `image.py:25-41` walks `reversed(messages)`, inspects the **first** user message found, then `break` → last-user-message only. `image.py:128-136` (`strip_and_cache_images`) iterates **every** user message. Detector and writer disagree on scope.
- The feature is **new surface on this branch**, not inherited: `grep -c has_images <base monolith>` = **0**.
- Help text promises the writer's scope: `main.py:74-75` "Strip images from user messages" (plural).
- No invariant governs it (`invariants/index.md` protects only `PI:T002/dump_format/*`, `PI:T006/upstream_tls/*`, and the dump/server `OI:`/`E:` handles).
- **Not intended-by-plan:** `01.04_2026-03-16_image-agent/plan-summary.md` is a stub (`Status: Ready for Collect`, all phases Pending) about `analyzeImage` interception — it authorizes nothing here.
- **Not a false positive** despite `tests/test_image.py:49-58` (`test_has_images_checks_last_user_message`, "Should only check the last user message"): that is a characterization test written beside the implementation, and it pins the *helper* only, never the route consequence. It must move with the fix.
- Second-order harm: the request now alternates stripped/unstripped per turn → the KV prefix changes every turn, i.e. the flag starts busting the cache it exists to protect.

```text
CLOSURE: absorbed
Finding: F-1 (codex #1) has_images gates stripping on the last user message only
Required outcome: with STRIP_IMAGES on, every image block in the request's user messages is replaced by a placeholder
Authority: flag help text (main.py:74-75) + writer's own iteration scope (image.py:128-136); no invariant, no plan override, no user decision ranks either scope
Admissible candidates: (a) scan all user messages in has_images; (b) drop the has_images guard and always call the writer
Rejected candidates: (b) as a replacement — leaves normalize.py:212's guarded call carrying the identical defect, i.e. preserves the verified defect for one of two call sites
Selected closure: (a) make has_images scan all user messages; invert test_image.py:49-58 to expect True; add a route-level test that a text-only latest turn still strips a historical image
Risk shape: decreased — detector and writer agree for the first time; default config (STRIP_IMAGES=False) is unaffected
Deferred additive work: owner = image-slice Forward task; trigger = when tool_result-nested images are in scope (today only top-level user content blocks are handled)
Owning route: Forward (new adhoc task, image surface)
Question required: no
```

### 2. `[P2]` Generate image placeholders with non-reusable keys — `buster_ripper/image.py:79-84`

**Concern (verbatim):** "After a session reaches its image-cache limit, `len(cache)` is reused as the next key: with a limit of 2, the third image gets `img_2`, then the fourth image also gets `img_2` and overwrites it. The earlier placeholder consequently resolves to the wrong image; retain a monotonic per-session sequence rather than deriving keys from current cache size."

**Classification: REAL DEFECT — and understated.** Evidence:

- `image_key = f"img_{len(_image_cache[session_id])}"` is computed at `:81` **before** insertion at `:84`, and `_evict_by_lru` runs **after** at `:87` → the collision codex describes.
- **Not reported by Codex, worse:** keys are re-derived on every request, so a *historical* image changes placeholder text between turns — turn 1 emits `[image:img_0]`, turn 2 (same image, cache already sized 1) emits `[image:img_1]`. A monotonic per-session counter (Codex's proposed fix) removes the collision but **preserves this instability**.
- The test suite cannot see it: `test_image.py:178-198` adds one image per call to a fixture-reset cache (`:10-15`), so no test ever replays a history.
- Harm axis: `get_cached_image` has no production caller today, so *misresolution* is latent — but *instability* is live, because the placeholder text is inside the forwarded body.

```text
CLOSURE: absorbed
Finding: F-2 (codex #2) placeholder keys collide and are not stable across requests
Required outcome: one image source ⇄ one placeholder key, identical on every request that carries it
Authority: product contract stated in README/module docstring ("keeping the KV cache prefix stable across turns") — normalization must be content-deterministic; no invariant covers image.py, so no plan-side protection to respect
Admissible candidates: (a) content-addressed key (digest of the source block); (b) monotonic per-session counter; (c) scan for first free index
Rejected candidates: (b) — preserves the defect: keys still shift per turn, so the prefix instability survives; (c) — same rejection, and reuses a freed key for a different image
Selected closure: (a) derive the key from the image source (e.g. img_<sha256(source)[:12]>), which also dedupes a repeated image; update the img_0-literal assertions in test_image.py:78,124,155,170 to the derived key — cause fix, not test exemption
Risk shape: increased slightly in surface (key format changes), decreased in correctness terms; placeholder text is internal and has no production reader
Owning route: Forward (same image-slice task as F-1)
Question required: no
```

### 3. `[P2]` Expire image caches on the write path — `buster_ripper/image.py:121-122`

**Concern (verbatim):** "With `--strip-images`, this creates a cache for each new session but the production write path never calls either cleanup function, so TTL-expired base64 payloads and empty per-session dictionaries remain indefinitely. Long-running proxies serving distinct system prompts therefore grow memory without the configured TTL providing a bound; run cleanup during writes or from a scheduled task and remove empty sessions."

**Classification: REAL DEFECT.** Evidence:

- `grep -rn "cleanup_expired_images\|clear_session_cache\|get_cached_image" buster_ripper/` → **no non-test callers**. `_cleanup_expired_images` is reached only from `get_cached_image:155`; the two public janitors (`:176`, `:182`) are never invoked.
- Therefore `config.IMAGE_CACHE_TTL` (`config.py:42`) and `--image-cache-ttl` (`main.py:127-128`) are advertised controls that bind nothing.
- `session_id` = `sha256(system)[:12]` (`session.py:93-98`) → a new `_image_cache` key per distinct system prompt, never reclaimed. Per-session count *is* bounded by `_evict_by_lru`, so this is a slow leak plus dead config, not unbounded per-session growth.
- Not invariant-protected, not in any plan.

```text
CLOSURE: absorbed
Finding: F-3 (codex #3) IMAGE_CACHE_TTL never enforced on the production path
Required outcome: retained base64 is bounded by config.IMAGE_CACHE_TTL and empty per-session dicts are reclaimed
Authority: config.py:42 and main.py:127-128 advertise the TTL as a live control; the module's own write-time maintenance precedent is _evict_by_lru being called from the writer
Admissible candidates: (a) prune expired entries + drop empty session dicts on the write path; (b) a background task calling cleanup_expired_images()
Rejected candidates: (b) alone — does not reclaim the empty session dicts (the actual unbounded axis) and adds a second lifecycle task for the same job; deferral — preserves the defect
Selected closure: (a) call _cleanup_expired_images(session_id) from strip_and_cache_images and delete the session entry when it becomes empty; keep (b) available as additive work
Risk shape: decreased — write-time pruning already the module's shape; cost is O(entries) per strip
Deferred additive work: owner = image-slice Forward task; trigger = if pruning cost ever shows on the request path
Owning route: Forward (same image-slice task)
Question required: no
```

### 4. `[P2]` Use complete hop-by-hop header filtering — `buster_ripper/routes/messages.py:62-63`

**Concern (verbatim):** "This abbreviated filter forwards headers such as `TE`, `Upgrade`, `Keep-Alive`, and `Proxy-Authorization` to the upstream, whereas the pre-refactor proxy removed the full hop-by-hop set. The same incomplete local filters appear in the chat and passthrough routes (and for relayed responses), so use the shared header helpers or the full set consistently to preserve proxy semantics and avoid forwarding proxy credentials."

**Classification: REAL DEFECT — the only true *regression* in the set, and the strongest finding.** Evidence:

- Base contract: `buster_ripper.py@72e3a1c:572` `if k.lower() not in ("host", "content-length", *_HOP_BY_HOP)` and `:581` `not in _HOP_BY_HOP`, where `_HOP_BY_HOP` (`:123-135`) is the full 8-header set.
- Branch: `utils.py:12-15` defines the *identical* frozen set and `utils.py:18-33` re-implements base's two filters faithfully — then **all three routes bypass them**: `messages.py:62-63`, `chat.py:101-102`, `passthrough.py:15-16` (requests, 4 keys) and `messages.py:257-260`, `passthrough.py:40-43`, `chat.py:158-160` (responses, 3 keys).
- No test pins the narrowed behavior: `grep -rn "Proxy-Authorization|Upgrade|Keep-Alive|hop-by-hop" tests/` → zero hits; the only header assertions in `tests/test_routes_integration.py` are Content-Type. `pyproject.toml:30-34` omits `buster_ripper/routes/*` from coverage, so this surface is unobserved by the gate — which is how drift survived 301 green tests.
- Credential-leak axis is real: `Proxy-Authorization` is forwarded verbatim to whatever `--upstream` names.

```text
CLOSURE: absorbed
Finding: F-4 (codex #4) hop-by-hop header filtering narrowed by the refactor
Required outcome: pre-refactor proxy semantics — the full RFC 9110 §7.6.1 set dropped in both directions
Authority: base implementation is the contract for a refactor branch (72e3a1c:572,:581); the refactor's own utils.py already encodes it; no invariant protects the narrowed filters — the nearest protected claims (CL:…/upstream-error-content-type, /upstream-error-absent-header-handling) pin error labeling, not header passthrough
Admissible candidates: (a) route all six sites through utils.forward_headers / utils.response_headers; (b) widen each inline literal
Rejected candidates: (b) — keeps three divergent copies of the policy, which is how this drift happened; deleting utils' helpers instead — abandons the refactor's own intended seam
Selected closure: (a) use the shared helpers at all six sites, extending response_headers to also drop content-length (aiohttp recomputes it), and preserve _prepare_headers' empty-Bearer deletion for the eval contract
Risk shape: decreased — removes duplication and restores base semantics; the only behavior delta is the restored drops
Owning route: Forward (single task; touches routes/ only)
Question required: no
```

---

## B. Cross-reference: self-flagged candidates vs Codex's output

Self-flagged before the findings arrived: `forward_headers` dead code, `convert/` package in `REFACTORING_PLAN.md`, `STATS_DB` unwritten, unreferenced image helpers, undocumented CLI surface.

| Self-flagged candidate | Codex | Disposition |
|---|---|---|
| `forward_headers` imported in `chat.py:14` and unused; `utils.forward_headers`/`response_headers` never called | **Confirmed as F-4** | **REAL DEFECT** — the dead helpers are the *fix*, not the bug: they encode base's semantics and were bypassed. The unused import is the symptom. |
| `convert/` package documented in `REFACTORING_PLAN.md` does not exist | **Missed** | **REAL DEFECT (docs).** `find buster_ripper -name '*.py'` → 16 files, no `convert/`. `REFACTORING_PLAN.md` is new on this branch (+160, absent at base) and asserts "Status: Phase 1 Complete" for modules that are not in the tree (`:26-28`, `:114-117`, `:152`, `:155`). `CLOSURE: absorbed` → correct or delete those rows; report-only risk. |
| `config.STATS_DB` never written | **Missed** | **CONVENTIONAL / REPORT-ONLY.** `main.py:155-161` assigns `session._db_conn` directly and never touches `config.STATS_DB` (`config.py:65`). Harmless dead surface, no path severed — a nitpick under the skill's stopping condition, worth a backlog line at most. |
| `get_cached_image` / `make_data_url` unreferenced by production code | **Partially covered** (F-3 names the cleanup functions; F-2's "wrong image" presumes a resolver) | **This is the root cause of F-2/F-3's severity profile, and it is Intended-by-plan.** `RB-003` (plan-summary `:142`): "Read side: nothing to reuse or extend — the role does not exist in code"; `FK-007` resolved **option 1 = no tool ships**; Anchor Map: "**Read path — deliberately NOT an anchor row.** There is no production dump reader." So the unreferenced read-side API is settled discretion — **report as explained-by, do not delete.** What is *not* excused is the write path leaving TTL unenforced (F-3). |
| New CLI surface undocumented (`--strip-images`, `--preserved-thinking`, `--backend-type`, `--eval-*`, `--host`) | **Missed** | **CONVENTIONAL / REPORT-ONLY, with one exception.** Base README and base monolith both had **0** occurrences of these flags (`for f … base=0 now=0`), so the gap is inherited documentation debt, not a regression — except that the branch *added* new flags to that undocumented set. `--dump-format` and `--skip-tls-verify` **are** documented (`README.md:90-92`, `:169-194`), which is the standard the rest should meet. |

---

## C. Additional findings from the reviewer's pass (not in Codex's output)

- **`[P3]` real defect — post-commit error return in `messages.py:129-181`.** One `except aiohttp.ClientError` wraps both the pre-`prepare()` section and the relay loop; a mid-relay `ClientPayloadError` returns a *fresh* `web.json_response`, whose `_payload_writer` is `None`, so `prepare()` (`web_response.py:455-459`) does **not** short-circuit and `_start` re-writes headers onto an already-committed stream. Fix: scope the handler to the pre-`prepare()` section and log-and-abort afterwards. Prior history (`1a95d8a fix(server): address codex-review findings in streaming…`) closed the pre-`prepare()` half of this family; this is the residue.
- **`[P3]` intended-by-plan — `buster_ripper_legacy.py` keeps a FastAPI/uvicorn/httpx PEP 723 header** while the package runs on aiohttp (`pyproject.toml:5-8`). Plan-summary `RB-002` (`:141`) names it explicitly: "the tracked 1013-line pre-refactor monolith — a copy of the same loop, referenced by nothing under `buster_ripper/`, **excluded from scope as dead legacy; a reader must not bind to it**." Self-contained under `uv run`, inert otherwise. Report only; do not "fix" its dependencies.
- **`[P3]` pre-existing, unchanged by this branch** — `dump.py:46` `_sessions` has no eviction (same shape as base), and `poll_flush` is spawned even on the documented path. The latter is mandated: Anchor Map "flush timer … **No-ops on the documented path (018, F-15)**" and `G-004` ("On the documented path the timer **no-ops** — there is nothing pending"). Consistent with plan; not a defect.

---

## D. Intended-by-plan register (reviewed, cleared — no action)

| Apparent defect | Authority that mandates it |
|---|---|
| `dump_turn` **raises** `ValueError` when `config.DUMP_DIR is None` (`dump.py:85-86`) | Anchor Map `dump entry point`: "the `DUMP_DIR is None` guard **replacing `dump.py:63`'s bare assert (F-5)**"; caller keeps the swallow-all contract (`CL:buster_ripper.routes.messages#handle_messages/dump-failure-isolation` HOLDS) |
| Collapsed formats don't cut writes by default | `US-012` + `S-1` + `PI:T002/dump_format/default_format_is_documented`: "a default run buffers nothing, starts no timer, and performs no shutdown flush… the reduction is opt-in" |
| Volume trigger is **not** in `T002` | 022 correction (`:92`): the check, `DUMP_FLUSH_BYTES` and `--dump-flush-bytes` are all `T003`'s ADDED items |
| `DUMP_DIR=${DUMP_DIR-…}` drops the colon (`serve_buster_ripper.sh:28`) | **FK-010 / T011, LANDED** (`plan-summary:26`): "the script now honors the form its own header documents" |
| `EXTRA_FLAGS` array + `${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}` | T012 + `seam-S6-value-word-split` `resolved` / `SAFE` @ `cb7c0a5c…`, with `CL:bash/array-alternate-value-nounset-exemption` and `CL:bash/quoted-array-expansion-word-identity` as its evidentiary basis |
| `SKIP_TLS_VERIFY` uses a `== "1"` boolean guard, not `-n` | `plan-summary:173`: a presence guard would append `--skip-tls-verify` on every run and invert the default posture, violating `PI:T006/upstream_tls/verification_on_unless_explicitly_disabled` and `G-007` |
| `exec buster-ripper` replaces `uv run "${SCRIPT_DIR}/buster_ripper.py"` | Consequence of the package refactor, with prerequisites stated at `serve_buster_ripper.sh:7-9` |
| `--skip-tls-verify` is the repo's only *relaxing* flag | `RB-005` (`plan-summary:144`): "all six *enable*, none *relaxes*… the polarity is new even though the `click` shape is established" — settled as NEW |

---

## E. The one item that is not the reviewer's to absorb

**`PRESERVED_THINKING=1` by default in the serve script (`serve_buster_ripper.sh:30`)** — injects `chat_template_kwargs={enable_thinking:true, clear_thinking:false}` into **every** forwarded `/v1/messages` request (`messages.py:43-50`).

- **Not a regression and not intended-by-plan:** added by `847ebbe` ("Add preserved thinking, fix eval split, fix strip_code_fences regression") — predates session `02.00`, and its decision records are **not in `invariants/index.md`**. Plan authority treats it only as a *shape peer* (`plan-summary:144`, `:173`), never as an endorsed default.
- **Not a defect** in the codex sense: the code does what it says, the suite is green, and no invariant is crossed.
- **Why it still reaches the user:** the documented run path silently changes outbound sampling semantics for every deployment, the flag is absent from `README.md` (`grep -ni preserved.thinking README.md` → no hits), and the two admissible outcomes — keep the default and document/record it, vs. default off and require an opt-in — are **materially different with no settled authority ranking them.** Per the gate, that is a `question`, not an absorption:

```text
CLOSURE: question
Finding: F-5 (self-flagged, cross-ref "undocumented CLI surface") — serve script injects preserved-thinking by default
Required outcome: the serve path's outbound sampling semantics are a recorded choice, not an undocumented default
Authority: absent — 02.00's plan uses PRESERVED_THINKING only as a guard-shape peer; no invariant, no user decision covers it
Admissible candidates: (a) keep default-on, document --preserved-thinking in README + record the decision; (b) default-off, opt-in per invocation
Rejected candidates: removing the flag (leaves approved behavior unreachable); deciding silently in review (no authority to rank)
Selected closure: none — unranked
Risk shape: unchanged either way; both are reversible and in-scope for the owning session
Deferred additive work: none
Owning route: Decide/interview in the session that owns the serve script (outside 02.00's T001–T012)
Question required: yes
```

---

## F. Bottom line

| Item | Priority | Location | Class | Route |
|---|---|---|---|---|
| F-1 historical images not stripped | P2 (codex) | `routes/messages.py:38-40` | real defect | Forward — image task |
| F-2 placeholder keys unstable/collide | P2 (codex) | `image.py:79-84` | real defect (understated) | Forward — image task |
| F-3 TTL never enforced | P2 (codex) | `image.py:121-122` | real defect | Forward — image task |
| F-4 hop-by-hop narrowing | P2 (codex) | `routes/messages.py:62-63` (+5 sites) | real defect — **only true regression** | Forward — routes task |
| F-5 preserved-thinking default | P2 (mine) | `serve_buster_ripper.sh:30` | **question** | Decide/interview |
| G-1 post-`prepare()` error return | P3 (mine) | `routes/messages.py:129-181` | real defect | Forward — routes task |
| G-2 `convert/` docs phantom | P3 (mine) | `REFACTORING_PLAN.md:26,114,152` | real defect (docs) | doc task |
| G-3 dead `STATS_DB`, flag doc debt | P3 (mine) | `config.py:65`; README | conventional | backlog |
| Legacy monolith, no reader/tool, opt-in formats, timer no-op, serve argv | — | — | **intended-by-plan / pre-existing** | cleared, §D |

**Nothing here contradicts or reopens session `02.00`.** Its approved scope (dump bulk-flush T001–T005, T011, T012; TLS posture T006; `PI:T002/*`, `PI:T006/*`) is where all four codex findings are *absent* — they live in the branch's earlier aiohttp-migration surface, which sits outside the plan and therefore had no red-pass, no invariant, and no coverage gate watching it (`pyproject.toml:30-34` omits `routes/*`). F-4 in particular is worth the merge-block argument: a refactor branch that silently weakens credential stripping is exactly the seam failure this review targets. Recommend one routes/header task (F-4 + G-1, one commit each), one image-surface task (F-1/F-2/F-3, tests included), and F-5 routed to Decide.
