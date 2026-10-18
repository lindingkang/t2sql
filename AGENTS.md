# AGENTS.md

Text2SQL agent for the Spider benchmark, built milestone-by-milestone (M0 baseline → M1 retry loop → M1.5 voting → M2 prompt/eval → M3 hardening → M4 SFT). Pure Python research/evals code; **no package manifest, no pytest, no README at root**. The source of truth for results and design decisions is `learning_notes.md` — read it before changing eval logic or claiming numbers.

## Setup

- `.env` (gitignored, create it yourself) holds `LLM_API_KEY`, `LLM_BASE_URL` (Baidu Qianfan OpenAI-compatible `https://qianfan.baidubce.com/v2`), `LLM_MODEL`. There is no `.env.example`.
- Core deps only: `openai`, `python-dotenv`, `sqlglot`. Installed manually — no `requirements.txt`.
- M4 training/eval deps (`transformers peft trl bitsandbytes accelerate datasets sentencepiece`) install via `python m4/install_env.py`.
- `spider/` data is gitignored and must be downloaded per machine. Expected layout: `spider/dev.json`, `spider/tables.json`, `spider/train_spider.json`, `spider/train_others.json`, `spider/database/<db_id>/<db_id>.sqlite` (166 sqlite DBs).
- M4 local-model paths are hardcoded (autodl env): base `/root/autodl-tmp/hf_models/Qwen2.5-Coder-7B-Instruct`, adapter `/root/autodl-tmp/m4_lora/sft_cons_v1`.

## Verification / commands (no pytest)

Each module is its own entrypoint with a `_self_test()` run by `python <module>.py`. This **is** the test framework:

- `python safety.py` and `python degrade.py` and `python cache.py` and `python logger.py` — module self-tests.
- `python schema.py --db <id> --verify` — DDL generation + in-memory sqlite validity check.
- `python run_baseline.py --dry-run` — verify exec chain with fixed/gold SQL, **no API key needed**; `--limit N` (default 100).
- `python eval_agent.py --limit 20` then full `python eval_agent.py` — M1/M3 agent eval. Requires `baseline_full.json` for rescued/regressed diff.
- `python eval_harness.py --variant baseline|improved --runs N --tag X` and `--summarize --runs-dir eval_runs` — M2 A/B + stats.
- `python agent.py --db <db> --question "..." [--n-samples N --adaptive --cache --log-dir DIR]` — single-question agent.
- M4: `python m4/eval_local_base.py --mode m0|m2|m15|m3`, `m4/eval_sft_adapter.py`, `m4/eval_sft_hybrid.py`, `m4/train_sft.py`, `m4/build_train_set.py`, `m4/data_audit.py`.

Always smoke-test with a small `--limit` (e.g. `--limit 10`) before a full 1034 run.

## Metrics & correctness conventions

- Primary metric is **Execution Accuracy (EA)**. Strict = `results_equal_strict` (set compare, **column-order sensitive**, Spider official). Loose = `results_equal_loose` (ignores column order).
- `execute_sql` in `run_baseline.py` is the only sanctioned execution path: read-only `mode=ro` sqlite, `progress_handler` timeout, row limit, Latin-1/cp1252 `text_factory` fallback (some Spider DBs like `wta_1` are not UTF-8). Never execute with a plain `sqlite3.connect`.
- **temperature=0 on Qianfan is still non-deterministic** (~±0.7% noise). Single-run differences are meaningless; report mean±std over multiple runs, treat only >2σ as significant.

## Code structure

- `run_baseline.py` — shared utilities everything imports: `SPIDER_DIR`, `build_prompt`, `extract_sql`, `execute_sql`, `results_equal_strict/loose`, `classify_failure`, `call_llm`/`call_llm_with_usage`, `DEFAULT_EXEC_TIMEOUT`/`DEFAULT_ROW_LIMIT`.
- `schema.py` — `tables.json` → DDL text (`load_schemas`, `get_ddl`); filters the phantom `sqlite_sequence` table.
- `agent.py` — `Text2SQLAgent`: voting → retry loop → empty-result confirm. Imports `build_prompt` from `run_baseline`, plus `safety`/`degrade`.
- `safety.py` (AST whitelist, fail-closed), `degrade.py` (error classify + retry policy/hints), `cache.py` (L1/L2 cache + token budget), `logger.py` (JSONL trace), `metrics.py` (cost board).
- `prompt_variants.py` — M2 prompt variants (`build_prompt(db_id, question, schema, variant)`); distinct from `run_baseline.build_prompt`.
- `m4/` scripts add `sys.path.insert(0, ROOT)` to import root modules; they run from repo root or `m4/` either way.

## Repo / git conventions

- Only `.py` files + `learning_notes.md` are committed. Result JSONs, logs, and M4 generated data are gitignored (`m4/*` except `*.py`, `_*.txt`, `*.pid`, eval JSONs). Conclusions belong in `learning_notes.md`, not in committed data files.
- `--limit 10` (no space after `--`) — `-- limit 10` is parsed as a positional arg and errors.
- `<db_id>` → DB file convention: `spider/database/<db_id>/<db_id>.sqlite`.
- `extract_sql` already handles markdown fences and multi-line SQL — do not re-implement or "helpfully" trim model output (that would corrupt M2 failure attribution).