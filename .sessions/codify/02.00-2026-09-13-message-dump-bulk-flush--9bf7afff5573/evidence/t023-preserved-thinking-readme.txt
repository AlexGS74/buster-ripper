T023 · preserved-thinking serve default — README.md §Serve script options
Date: 2026-10-18
HEAD: 684a0db7351ff75ce9eeec4091cfa2fbef09105e

== README.md:71-107 (rendered text) ==
    71	## Serve script options
    72	
    73	```bash
    74	# Default: listen 0.0.0.0:30001 → upstream localhost:30000
    75	bash serve_buster_ripper.sh
    76	
    77	# Also strip daily date injection (Today's date is YYYY-MM-DD.)
    78	STRIP_DATE=1 bash serve_buster_ripper.sh
    79	
    80	# Verbose — log every normalization and per-turn token/latency stats
    81	VERBOSE=1 bash serve_buster_ripper.sh
    82	
    83	# Enable session prompt diffing
    84	DUMP_DIR=~/mllm/prompt-diffs bash serve_buster_ripper.sh
    85	
    86	# Custom upstream / bind port
    87	UPSTREAM=http://10.0.0.1:30000 PORT=30002 bash serve_buster_ripper.sh
    88	
    89	# Skip upstream TLS certificate verification — for self-signed certificates.
    90	# Verification is ON by default; SKIP_TLS_VERIFY=1 turns it off process-wide
    91	# (--skip-tls-verify), for every outbound call the proxy makes.
    92	SKIP_TLS_VERIFY=1 bash serve_buster_ripper.sh
    93	
    94	# Preserved Thinking is ON by default (--preserved-thinking): forwarded
    95	# /v1/messages requests carry enable_thinking=true and clear_thinking=false
    96	# in chat_template_kwargs, unless the client already set them. Turn it off:
    97	PRESERVED_THINKING=0 bash serve_buster_ripper.sh
    98	```
    99	
   100	Preserved Thinking is the serve script's default, not an opt-in: `PRESERVED_THINKING`
   101	defaults to `1`, so `--preserved-thinking` reaches the binary unless you set it to `0`.
   102	The binary itself defaults to off; the flag turns it on. That default-on posture is the
   103	recorded decision **`FK-011 · preserved-thinking serve default`** (Option A — keep the
   104	default on and document it), not an implicit side effect.
   105	
   106	Or call directly:
   107	```bash

== serve_buster_ripper.sh:30,42 (behavior read-back) ==
    30	PRESERVED_THINKING=${PRESERVED_THINKING:-1}   # 1 = Preserved Thinking (enable_thinking=true, clear_thinking=false)
    42	[[ "${PRESERVED_THINKING}"  == "1" ]] && EXTRA_FLAGS+=(--preserved-thinking)

== buster_ripper/main.py:107-110 (flag definition) ==
   107	@click.option("--preserved-thinking", is_flag=True,
   108	              help="Inject enable_thinking=true + clear_thinking=false into /v1/messages requests. "
   109	                   "Keeps the model's reasoning chain visible across turns (Preserved Thinking). "
   110	                   "clear_thinking=false is SGLang-only; harmless to send to vLLM.")

== buster_ripper/routes/messages.py:46-47,73-80 (injection) ==
    46	    if config.PRESERVED_THINKING:
    47	        _inject_preserved_thinking(body)
    73	def _inject_preserved_thinking(body: dict) -> None:
    74	    """Put the flag's kwargs pair on the body; a client-sent non-mapping is replaced, not skipped."""
    75	    kwargs = body.get("chat_template_kwargs")
    76	    if not isinstance(kwargs, dict):
    77	        kwargs = {}
    78	        body["chat_template_kwargs"] = kwargs
    79	    kwargs.setdefault("enable_thinking", True)
    80	    kwargs.setdefault("clear_thinking", False)

== scripts (sha256) ==
c58256fbec793c520720c9e86fee8740653939275b508f50b212ab60bd0908af  serve_buster_ripper.sh
dd88679cc8dd9f2fa5e459e1c1d875536260b9d5410230671bdc632808aa3a87  tests/test_serve_script_argv.py  tests/test_serve_script_argv.py
