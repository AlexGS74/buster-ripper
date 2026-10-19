# 007 — collect (storage-approach research for SSD longevity)

## Orchestrator Prompt (verbatim)

> Scope of this investigation: research (including external sources) the storage approaches that minimize SSD wear for this session's write-heavy message-dump payload — file-per-turn vs bulk-append vs a database (e.g. SQLite) — and report findings relevant to the current plan's payload/flush design. This is discovery only; return findings.
>
> User Context: "That's another good point. Can you do research while we're waiting for organize and find the best way to deal with issues like this to maximize SSD longevity? Because this kind of payload can be really harmful. One of the options we can use maybe is using a database. Those will be, how to say it, they can be still write heavy, but they will not write file by file."

## Session Handling

**Continuing existing session** `02.00-2026-09-13-message-dump-bulk-flush--9bf7afff5573`.
Identity verified against `session-identity.json` (UID `9bf7afff-5573-402a-b102-827423a1dded`,
handle suffix `9bf7afff5573` matches the first 12 hex of the UID). No new session created.

## User statement recorded

Reserved via `codify codify reserve-decision-record` → **US-002** (filled in
`decisions.md`, State `recorded`). Two factual claims recorded with
`Verification: assumptions PENDING`:
1. "this kind of payload can be really harmful" (to SSD life);
2. "they can be still write heavy, but they will not write file by file" (a DB avoids
   per-file writes while remaining write-heavy).

Neither claim is settled here; §Findings supplies evidence, §Critical-Thinker
Validation classifies it.

## Repo facts gathered (source-checked this pass)

| Fact | Source |
|---|---|
| `dump_turn` runs **synchronously on the request path**, once per `/v1/messages` request; exceptions swallowed | `buster_ripper/routes/messages.py:55-59` |
| Current cost per turn: **2 write-bearing ops** — one `write_text` for `turn_000.json` or `turn_NNN.diff`, plus one `index.txt` open+append+close | `buster_ripper/dump.py:82,95,104-106` |
| Turn 0 is the full normalized body (`json.dumps(..., indent=2)`); later turns are `difflib.unified_diff` of `_render_body` output | `buster_ripper/dump.py:77-102` |
| Documented example sizes: turn 0 = **412.3 KB** (83 tools); later turns e.g. `+892/-0 lines` | `README.md:146-162` |
| Session dir created once per session (`mkdir(parents=True, exist_ok=True)`), not per turn | `buster_ripper/dump.py:71-72` |
| The dump dir is not covered by tests today (`tests/test_dump.py` does not exist) | `tests/` listing |
| **SQLite is already a repo dependency and is already used** — `session.py` persists per-session stats via `sqlite3`, `check_same_thread=False`, **`commit()` on every upsert** | `buster_ripper/session.py:7,35-71` |
| Stats persistence is **opt-in** (`--stats-db`, default `None`); the serve script does not enable it | `buster_ripper/main.py:36,94`; `serve_buster_ripper.sh` |
| On the dev machine the internal volume is **88% full (114 GB free of 926 GB)**; `~/mllm/prompt-diffs` does not currently exist | `df -h ~`, `ls ~/mllm/prompt-diffs` |
| No `.invariants/` registry exists in this repository (unchanged) | plan-summary §Invariant Reconciliation |

## Research findings (external)

### F1. SSD wear is measured in **bytes written**, not write-call count

- Endurance is specified as TBW (total bytes writable within warranty) or DWPD
  (drive writes per day); TBW = P/E-cycle resource × capacity (e.g. a 1 TB Samsung
  970 EVO is rated 600 TBW). ([Wikipedia, SSD](https://en.wikipedia.org/wiki/Solid-state_drive))
- The physical quantity that consumes P/E cycles is data written to NAND,
  including write amplification (WAF) — defined as flash bytes written ÷ host
  bytes requested. ([Wikipedia, Write amplification](https://en.wikipedia.org/wiki/Write_amplification))
- **Direct relevance:** the plan's `G-001` metric counts *file-write operations*
  (`F` vs `2N`). That is a **proxy**, not the physical quantity. The plan already
  carries the corrective: witness B (`O(N²)` bytes on a rewrite-per-flush). The
  sources confirm witness B is the genuinely dangerous axis; the count is a
  reasonable but secondary proxy.

### F2. What raises WAF: small, random, unaligned, metadata-heavy writes

- Granularity mismatch: host writes arrive as 4–8 KB pages; NAND is erased in
  blocks of ~256 KB–several MB, so a small change forces read-modify-write of a
  whole block. "Small, non-sequential writes have a greater negative impact."
  Large sequential writes can reach WAF ≈ 1 because a block fills with one file's
  data and can be erased wholesale. ([Wikipedia, Write amplification](https://en.wikipedia.org/wiki/Write_amplification))
- Measured on real Linux filesystems: "data operations result in significant
  write amplification (**2–32×**)"; "**small writes + fsync**" is the worst case
  because the sync forces fixed-cost metadata commits, and "**as write size
  grows, the write amplification drops close to one**." Per-filesystem overwrite/
  append WAF (4 KB writes, 10 MB total): ext2 2.00, XFS 2.00, **ext4 4.00**,
  F2FS 2.66, **btrfs 32.65**. "Even a simple file rename results in the 12–648 KB
  being written to storage"; file creation 16–116 KB of write IO.
  ([Analyzing IO Amplification in Linux File Systems, ar5iv 1707.08514](https://ar5iv.labs.arxiv.org/html/1707.08514))
- Micron benchmark cited: "**random 4 KB writes generate the highest write
  amplification**"; metadata churn produces "dozens of writes for a tiny change
  to a single file"; DRAM-less controllers do a "double write" of the FTL map;
  QLC offers only ~500–1,000 P/E cycles.
  ([XDA](https://www.xda-developers.com/small-files-are-the-bane-of-your-ssd/))
- File-per-turn is therefore the *worst* of the three candidate patterns: each
  turn adds an inode + directory entry + (on ext4) journaled metadata, on top of
  the payload bytes, and the writes are small relative to the block erase unit.

### F3. `fsync`/barrier count is a multiplier, and the plan already avoids it

- SQLite's own description of atomic commit: a rollback-journal commit needs
  roughly **three sync points** (two journal flushes + one database flush); the
  syncs dominate commit latency. On flash, "the minimum size of a read is
  typically much smaller than a minimum write" and "SQLite is only concerned with
  the minimum write amount."
  ([SQLite, Atomic Commit](https://sqlite.org/atomiccommit.html))
- The OS page cache coalesces un-synced writes; the LWN article documents that
  ext3/ext4 flush dirty data on a timer (`dirty_writeback_centisecs` default 5 s,
  `dirty_expire_centisecs` default 30 s), and that `fsync` under ext3
  data=ordered is "more-or-less equivalent to a sync of the entire filesystem."
  ([LWN, Ensuring data reaches disk](https://lwn.net/Articles/322823/))
- **Direct relevance:** the plan's flush contains **no fsync** and no `await`
  (settled, protected). Because `US-001` waives durability, this is the single
  largest WAF reduction available and it is already in the design. It also means
  the *number* of `write()` calls is a weaker signal than it looks: un-synced
  writes are buffered and flushed in batches by writeback, so syscall count does
  not map 1:1 to device writes. The count still matters through **metadata
  operations** (each new file) and **locality**, not through the syscall itself.

### F4. Append-only, large sequential writes are the FTL-friendly pattern

- Log-structured filesystems write sequentially to a log; "the structure is
  naturally suited to media with append-only zones or pages such as flash."
  ([Wikipedia, LFS](https://en.wikipedia.org/wiki/Log-structured_file_system))
- "The flash translation layer in a modern, full-featured SSD resembles a
  log-structured file system"; flash "writes must be done in large contiguous
  blocks." The catch is garbage collection: the more the device fills and the
  more fragmented the data, the more live data must be relocated per unit of new
  data. ([LWN, Log-structured file systems: There's one in every SSD](https://lwn.net/Articles/353411/))
- **Direct relevance:** the settled collapsed artifact (one file per session,
  appended once per flush, never rewritten) is exactly the append-only large-
  sequential pattern. Source-supported.

### F5. SQLite is not automatically lower-write than a file — it can be far higher

Configuration determines everything:

- **Defaults are the expensive path.** Default `journal_mode=delete` +
  `synchronous=FULL`: a commit writes original pages to a rollback journal, syncs
  it (~2 syncs), writes modified pages to the DB, syncs again. A page changed by
  one row can dirty the table page plus several index pages; "the cost is per
  *transaction*, not per byte."
  ([Smart Home Troubleshoot](https://smarthometroubleshoot.com/mitigating-sqlite-write-amplification-preventing-emmc-flash-wear-out-and-file-system-corruption-in-smart-home-gateways/))
- **WAL is better but is not a byte reduction.** "In WAL mode each page is still
  written twice" (WAL, then the checkpoint into the main DB). SQLite's own WAL doc
  confirms auto-checkpoint at ~1,000 pages and that the WAL file is reused, not
  shrunk. ([SQLite, WAL](https://sqlite.org/wal.html))
- **`synchronous=NORMAL` in WAL avoids the per-commit fsync** — "no sync
  operations occur during most transactions"; the docs call it "normally all one
  needs in WAL mode" when durability is not a concern.
  ([SQLite, PRAGMA synchronous](https://sqlite.org/pragma.html#pragma_synchronous);
  [avi.im, SQLite commits are not durable under default settings](https://avi.im/blag/2025/sqlite-fsync/))
- **SQLite positions itself against `fopen()`**, not against server databases,
  and says it "can be faster than the filesystem for reading and writing content
  to disk." It does not claim lower write *volume*.
  ([SQLite, When to use](https://www.sqlite.org/whentouse.html))
- **When the write pattern is bad, SQLite's amplification is catastrophic.** Two
  documented incidents:
  - **OpenAI Codex** (`logs_2.sqlite`): a retention cap made insert+delete churn
    continuous; WAL frames + table/index page rewrites per one-row INSERT →
    **~37 TB onto the main SSD in ~21 days** (~640 TB/year vs ~600 TBW ratings).
    ([lilting.ch](https://lilting.ch/en/articles/codex-sqlite-log-ssd-writes);
    [openai/codex#27911](https://github.com/openai/codex/issues/27911) — the
    `goals_1.sqlite` case: **11.3 MB/s sustained writes against a 25 KB database**)
  - **NordVPN** (`moose.db`): default `journal_mode=delete` + `synchronous=FULL`,
    transactions touching nearly all pages → **~30 MB/s with a 30 MB DB,
    projected 600 TBW in one year**; cumulative write volume grows
    **quadratically** because write rate scales with DB size. The rollback-journal
    protocol emits **4 fdatasyncs per commit**.
    ([NordSecurity/nordvpn-linux#1477](https://github.com/NordSecurity/nordvpn-linux/issues/1477))
- **Direct relevance:** a database is not a wear mitigation. Chosen naively for a
  per-turn/per-flush log, it re-creates the exact hazards `G-001` witness B
  guards against — page-granular transactions, journal/WAL double-write, and
  write volume that grows with accumulated state. SQLite would only be
  defensible here with WAL + `synchronous=NORMAL`, batched transactions, no
  retention churn, and no `auto_vacuum`; even then it writes each changed byte
  ~twice, versus once for a plain append.

### F6. Free space and fragmentation raise WAF independently of the write pattern

- Over-provisioning has an inverse relationship with WAF; keeping ~20% free aids
  garbage collection; fragmentation forces relocation.
  ([Wikipedia, Write amplification](https://en.wikipedia.org/wiki/Write_amplification);
  [XDA](https://www.xda-developers.com/small-files-are-the-bane-of-your-ssd/))
- APFS is copy-on-write and "optimized for solid-state drive storage," but
  redirect-on-write "creates new extents (data blocks) for the changes, leading to
  more fragmentation over time"; no user-data checksums.
  ([Wikipedia, APFS](https://en.wikipedia.org/wiki/Apple_File_System))
- **Direct relevance:** the dev machine's volume is **88% full**. If `DUMP_DIR`
  lives there, WAF is elevated by occupancy regardless of the design. This is a
  machine-level fact, not a code change.

### F7. Comparative summary (evidence-backed)

| Approach | Bytes written per session | Writes per flush | Metadata ops | fsync | Source basis |
|---|---|---|---|---|---|
| **File-per-turn** (today) | ~payload + per-file metadata (WAF 2–32×; ext4 4×) | `2N` (turn file + index append) | one inode+dir entry per turn | none today | F2 |
| **Bulk append, one collapsed artifact** (settled by `FK-002`) | ~payload, once, sequential (WAF → ~1) | `F` (1 per flush) | one file per session | none | F1, F3, F4 |
| **Rewrite-per-flush artifact** (excluded by `FK-002`/witness B) | `O(N²)` | `F` | one file | none | F1; cf. NordVPN quadratic |
| **SQLite, WAL + `synchronous=NORMAL`** | ~2× changed bytes (WAL + checkpoint) | ~1 txn per flush | DB file | none during most txns | F5 |
| **SQLite, defaults (delete + FULL)** | ~2× changed bytes + journal, **3–4 syncs/commit** | ~1 txn per flush | DB file | 3–4/commit | F3, F5 |

## Critical-Thinker Validation

**Initial answer:** for this payload, absolute SSD wear is negligible versus TBW
ratings; the bulk-append append-only design is the lowest-write option; SQLite is
not an improvement and can be much worse.

**Potential errors checked:**

1. *Payload volume underestimated.* Turn 0 = 412 KB; assume 200 turns × ~30 KB
   ≈ 6 MB + 412 KB ≈ **6.4 MB/session**. At 10 sessions/day → ~23 GB/year; at
   1,000 sessions/day → ~2.3 TB/year. Against 300–600 TBW this is 26,000+ years
   and 130–260 years respectively. **Claim 1 ("really harmful") is not supported
   at these volumes** — but it becomes supportable under a pathological pattern
   (F5) or if the payload/volume assumptions are wrong. *Assumption to flag.*
2. *"Write count doesn't matter" overstated.* Corrected: syscall count is not the
   physical quantity (F3), but it is not irrelevant — each new file is a metadata
   operation (F2), and non-sequential small writes raise WAF (F2, F4). The
   accurate claim is "bytes written and their locality/granularity are the wear
   quantity; syscall count influences it indirectly."
3. *Blanket "SQLite writes each page twice" too strong.* Configuration-dependent:
   WAL+checkpoint ≈ 2× for append-shaped data; `journal_mode=OFF/MEMORY` +
   `synchronous=OFF` writes in place once but randomly and without crash safety
   (durability is waived here, so this is available). The safe statement is
   "SQLite's default and WAL modes write each changed byte ~twice or more."
4. *WAF 2–32× figures are vintage (2017) and include btrfs as an outlier.*
   Directional, not a prediction for APFS/ext4 today. Cited as such.
5. *Platform ambiguity.* The proxy may run on Linux (ext4/XFS, likely, next to
   vLLM) or on the macOS dev machine (APFS). The findings hold in direction for
   both; exact WAF differs. Flagged as a deferred question.
6. *The repo already commits to SQLite per request on the stats path.* With
   `--stats-db`, `db_upsert` commits on **every** upsert (defaults = delete +
   FULL), i.e. ~3–4 syncs/request — a larger write source than the dump's 2
   writes/request when enabled. Not enabled by default; out of this plan's scope,
   but it is the strongest in-repo evidence that "database" and "low wear" are
   not the same claim.
7. *Claim 2 ("will not write file by file").* Partly true and partly misleading:
   a DB does avoid many files (fewer inodes/dir entries), but it replaces them
   with page-granular transactions, journal/WAL, and index maintenance. Against a
   **single append-only file** it is the higher-write option.

**Corrected final answer:** The settled bulk-append, single collapsed, append-only,
no-fsync design is the lowest-write pattern of the candidates and is directly
supported by the sources (large sequential appends → WAF ≈ 1; no fsync → no
barrier amplification; append-only → bounded O(N) bytes). File-per-turn is the
worst. A database is **not** a wear mitigation for this workload: at best it
matches the append's locality while writing each changed byte ~twice; at worst
(default journal, per-tick commits, retention churn) it produces the
quadratic, TB-per-week amplification seen in the Codex and NordVPN incidents.
The user's "really harmful" premise is not supported at the estimated payload
volume; the *design pattern*, not the payload size, is what determines harm.

## Relevance to the current plan (per design element)

| Plan element | Research verdict | Notes |
|---|---|---|
| `FK-002` collapsed single artifact | Supported | Append-only large sequential writes → WAF ≈ 1 (F1/F4) |
| One write per flush (`G-001`) | Supported as a proxy | Count is secondary to bytes; keep witness B (F1) |
| Append-only, not rewrite-per-flush | Strongly supported | Rewrite → `O(N²)` bytes; the NordVPN quadratic case is the real-world form (F1/F5) |
| No fsync (durability waived) | Strongly supported | Largest single WAF reduction; already protected (F3) |
| File-per-turn (displaced layout) | Confirmed as the worst option | Metadata churn + small writes (F2) |
| "Use a database" (user's candidate) | **Not supported as a wear win** | Defaults are worse; WAL ≈ 2×; documented blow-ups (F5). Would need WAL + `synchronous=NORMAL` + batched txns to merely match |
| `G-006` record content (successor FK) | Constraint surfaced | Larger per-record payloads amortize page/sector overhead; a machine record with tiny records is the least FTL-friendly choice (F2). Readability vs write size tension is unchanged |

**Optional mitigations surfaced (not decisions):**
- Compression of the appended records (JSON compresses well) would cut bytes
  written — the only lever that reduces the physical quantity further — at the
  cost of a reader/format change. `RB-003` currently records "no compression
  facility is warranted"; the research does not overturn that, but notes it is
  the one remaining byte-reduction lever.
- Keep the target volume well below full (F6). The dev machine is at 88%.
- If a DB is ever chosen: WAL + `synchronous=NORMAL`, batch transactions, no
  retention churn, no `auto_vacuum` (F5).

## Existing Invariants In Scope

No new invariant surfaced. The six session-local `EI:` entries and their
dispositions are unchanged by this research. The contradiction surface is
unchanged.

## Behaviors to Protect (Regression Testing)

No change. The research does not add or alter observable behavior. The existing
`B-*`/`VP-*` set (plan-summary §Behavior Disposition Matrix) stands. One nuance
recorded for decide: `VP-T002-write-count-reduced` proves the proxy metric; a
byte-volume observation (or the witness-B form) is the physical quantity, and the
two should not be conflated.

## Anchor Map Draft

No change. The research adds no code referent. A future `Mode: NEW` row would be
required only if decide selects a database, a compression codec, or a viewer;
none is selected here.

## Deferred Questions

- **Which device/filesystem hosts `DUMP_DIR` in production** (Linux ext4/XFS
  beside vLLM, or macOS APFS on the dev box)?
  - Status: DEFERRED for /codify-review
  - Impact: exact WAF differs (ext4 measured 4× on small overwrite; APFS is CoW);
    the directional verdict does not.
  - Agent analysis: `none` — a factual premise for any numeric claim, not a user
    preference; source-checkable by inspecting the deployment host.
- **Should the write-count VP be paired with a byte-volume check?**
  - Status: DEFERRED for /codify-review
  - Impact: the count metric alone would not catch a rewrite-per-flush regression
    if witness B were dropped.
  - Agent analysis: recommendation is "yes, keep the existing witness-B form";
    decide owns the final VP set.
- **Is compression acceptable** (changes the read path/format)?
  - Status: DEFERRED for /codify-review
  - Impact: the only remaining lever to reduce bytes written.
  - Agent analysis: `none` — a user-visible format trade-off, not a factual one.

## Sources

- https://en.wikipedia.org/wiki/Write_amplification
- https://en.wikipedia.org/wiki/Solid-state_drive
- https://en.wikipedia.org/wiki/Log-structured_file_system
- https://en.wikipedia.org/wiki/Apple_File_System
- https://ar5iv.labs.arxiv.org/html/1707.08514 (Analyzing IO Amplification in Linux File Systems)
- https://lwn.net/Articles/322823/ (Ensuring data reaches disk)
- https://lwn.net/Articles/353411/ (Log-structured file systems: There's one in every SSD)
- https://sqlite.org/atomiccommit.html
- https://sqlite.org/wal.html
- https://sqlite.org/pragma.html#pragma_synchronous
- https://www.sqlite.org/whentouse.html
- https://avi.im/blag/2025/sqlite-fsync/
- https://lilting.ch/en/articles/codex-sqlite-log-ssd-writes
- https://github.com/openai/codex/issues/27911
- https://github.com/NordSecurity/nordvpn-linux/issues/1477
- https://smarthometroubleshoot.com/mitigating-sqlite-write-amplification-preventing-emmc-flash-wear-out-and-file-system-corruption-in-smart-home-gateways/
- https://www.xda-developers.com/small-files-are-the-bane-of-your-ssd/
- https://search.brave.com/search?q=number+of+write+syscalls+vs+bytes+written+SSD+wear+does+write+count+matter (snippet: SO 21735511 — write combining; sequential small writes tolerable; many tiny commits harmful)

## Questions Asked

No questions asked — this was a deferred-mode, research-only dispatch with a clear
scope. Findings and three deferred questions are recorded above.
