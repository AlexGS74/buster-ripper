---
name: qa-project
description: Project QA harness contract for buster-ripper — how to start the proxy, drive it, deliver signals, and capture evidence for QA gates. Read before authoring or executing any QA gate test in this project.
---

# QA Project Contract — buster-ripper

Project-specific QA harness declaration. The generic scenario-contract methodology
lives in `qa-manual-testing`; THIS file declares what this project's harness
actually supports. QA plans, author passes (`QAP-*`), and gate tests must operate
within these declarations.

## Harness Declarations

| Item | Declaration |
|---|---|
| Start command | `.venv/bin/buster-ripper --host 127.0.0.1 --port <port> --upstream http://hyperipper.local:30000 --dump-dir <tmpdir>`. Add `--dump-format <encoding>` for a per-encoding gate, and `--dump-flush-interval <seconds>` / `--dump-flush-bytes <bytes>` to make the flush triggers deterministic. Endpoint corrected 2026-10-17 from `:30001`, which is a second Buster Ripper instance rather than the model server (`decisions.md` §US-009). |
| Base URL | `POST http://127.0.0.1:<port>/v1/messages` — the interception path that reaches the dump. The proxy listens locally; the upstream is remote. |
| Upstream | `http://hyperipper.local:30000` — the live remote vLLM-compatible endpoint. It serves the Anthropic Messages API (`/v1/messages`) natively, as well as `/v1/chat/completions`, so the proxy passes Anthropic requests through. Gates send `model: sloppus`, the shared alias for whatever model the server runs; any other name gets a 404 before generation. There is no local stub: **a gate run fails if that host is unreachable**. Endpoint corrected 2026-10-17 from `:30001`, which is a peer Buster Ripper proxy, not the model server (`decisions.md` §US-009). **Stated amendment — the upstream-TLS gate only (`MS-004`):** that one gate does not use this upstream. It starts its own loopback self-signed HTTPS peer (`https://127.0.0.1:<peer-port>`, certificate material committed with the gate) because neither posture of the upstream-TLS opt-out can be observed against a plain-HTTP host. The amendment is scoped to that gate: every other gate keeps the remote upstream above, the hermetic default suite is unchanged, and no other gate gains a local peer. |
| Test framework | pytest + pytest-asyncio + pytest-aiohttp + pytest-mock (the `[dependency-groups] dev` set in `pyproject.toml`); `asyncio_mode = "auto"` is set in `[tool.pytest.ini_options]`. |
| Test file location | `tests/qa/` — a separate tree from the mocked `tests/` suite. |
| Teardown | Send SIGTERM (or SIGINT) to the proxy subprocess and wait for exit, then inspect the dump directory. The signal path is the feature under test, so teardown is also a trigger. |
| Deterministic triggers | The harness fixes **both** flush parameters at process start — the interval and the volume threshold — through their CLI flags. Without them the timer gate and the volume gate are sleep-and-hope races, not tests. |
| Dump root | The `--dump-dir` value; a fresh temporary directory per gate. The proxy creates it at startup. |
| Session directory | `DUMP_DIR/<sid>/`, where `<sid>` is the first 12 hex characters of `sha256(system prompt)` (`session.py:session_id`). A gate that must know the directory either computes the id from the system prompt it sent or reads the single subdirectory the run created. |
| Artifact | `DUMP_DIR/<sid>/<encoding>.log` — one append-only artifact per `(session, encoding)`, where `<encoding>` is the configured `--dump-format` value. |
| Prerequisites | `uv sync` has run (`.venv/bin/buster-ripper` exists); the remote upstream is reachable. |

## Capability Declarations (for the planning-time QA Capability Gate)

- **Synchronization**: required. The proxy is a long-running server. **Readiness** —
  poll until the listening port accepts a connection (or until the startup log line
  appears) before sending a request. **Quiescence** — after SIGTERM/SIGINT, wait for
  process exit before inspecting `DUMP_DIR`; for the timer gate, poll the artifact
  (contents or mtime) rather than sleeping a fixed multiple of the interval.
- **Determinism**: required and available. Both flush triggers are injectable at
  process start (`--dump-flush-interval`, `--dump-flush-bytes`) and each gate uses a
  fresh temporary `--dump-dir`, so the flush points are chosen by the test rather
  than observed. **Caveat**: the upstream is a live remote service — request
  outcomes (model output, latency) are not under the harness's control. Gates must
  assert on the proxy's dump artifact and process behavior, never on model output.
- **Harness tier**: process-level end-to-end. The gates start the real proxy binary
  as a subprocess, send real HTTP requests, deliver real signals, and inspect the
  files it wrote. No in-process or mocked tier can prove the shutdown flush, the
  timer flush, or the volume flush — those depend on process lifecycle and on real
  signal delivery.
- **Evidence**: the dump artifact and the session directory listing; the process exit
  code; captured stdout/stderr (the proxy logs to stderr); and, for write-count
  claims, an instrumented write-bearing file operation on a dump artifact
  (`write_text`, or `open(..., "a"/"w")` + `write`; directory creation is not a
  write-bearing operation).

## Conventions

- **Hermetic default suite.** The gates under `tests/qa/` need the remote upstream,
  real subprocesses, and real signals. They are selected explicitly (by path or by
  marker) and are not part of the hermetic default `uv run pytest` run, which must
  keep passing without network access. The default run's coverage gate
  (`--cov=buster_ripper --cov-fail-under=80`, `[tool.pytest.ini_options] addopts`)
  governs the unit/integration suite, not the QA gates.
- **One session per gate.** Give each gate its own `--dump-dir` and its own system
  prompt so its session directory is unambiguous.
- **Signals are triggers, not teardown only.** A shutdown gate sends the signal and
  then asserts on both the artifact contents and the exit status.
- **The dump is opt-in.** Without `--dump-dir` the proxy never writes an artifact;
  a gate that expects one must pass the flag.

## Verified Against This Repository

Verified 2026-09-14: `.venv/bin/buster-ripper` resolves the console script
(`pyproject.toml` `[project.scripts]`); the dev dependency group and
`asyncio_mode = "auto"` are in `pyproject.toml`; `tests/` holds the mocked suite
and `tests/qa/` does not exist yet; the CLI options `--upstream`, `--host`,
`--port`, `--dump-dir` exist today (`buster_ripper/main.py`), and
`--dump-format`, `--dump-flush-interval`, `--dump-flush-bytes` arrive with the
buffered-dump work this contract was written alongside.
