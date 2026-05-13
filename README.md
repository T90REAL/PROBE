# PROBE

**PROBE** is a framework for automatically generating Property-Based Tests (PBTs) via **Adversarial Refinement**. Rather than treating PBT generation as a one-shot translation task, PROBE frames it as a game of semantic asymmetry between a **Generator** and a **Validator**, iteratively hardening the properties.

## Requirements

- Python 3.11+
- An OpenAI-compatible API endpoint (OpenAI, DeepSeek, Qwen, SGLang, etc.)
- The target package installed in the Python environment used for test execution

Install dependencies:

```bash
pip install openai hypothesis pytest
```

## Configuration

Copy the example config and fill in your settings:

```bash
cp config.example.toml config.toml
```

```toml
[runtime]
model_name = "deepseek-chat"          # Recommended: DeepSeek. Any OpenAI-compatible model name works.
base_url = "https://api.openai.com/v1"
api_key = "sk-..."                    # Or leave as placeholder and set env var
python_executable = "python"         # Python used to execute generated tests
llm_concurrency = 1
request_timeout_seconds = 300
pytest_timeout_seconds = 120
temperature = 0.4
enforce_strength_checks = false       # Reject generated tests deemed too weak
reject_weak_suites = false            # Reject final suite if all properties are narrow
```

All fields can also be set via environment variables:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` / `PROBE_API_KEY` | API key |
| `OPENAI_BASE_URL` | Base URL override |
| `PROBE_MODEL` | Model name override |
| `PROBE_PYTHON` | Python executable override |
| `PROBE_LLM_CONCURRENCY` | Concurrent LLM requests |
| `PROBE_LLM_TIMEOUT_SECONDS` | Per-request timeout |
| `PROBE_PYTEST_TIMEOUT_SECONDS` | Per-test-run timeout |
| `PROBE_LLM_TEMPERATURE` | Sampling temperature |
| `PROBE_ENFORCE_STRENGTH_CHECKS` | `true`/`false` |
| `PROBE_REJECT_WEAK_SUITES` | `true`/`false` |

## Output

`run.py` prints a single JSON object to stdout:

```json
{
  "status": "success",
  "target_function": "sortedcontainers.sortedlist.SortedList.add",
  "strengthening_rounds": 2,
  "final_pbt_code": "from hypothesis import given, settings\n...",
  "properties": [
    {
      "property": "Adding an element and then removing it leaves the list unchanged",
      "status": "passed",
      "evidence": "docstring: 'Add value to sorted list.'",
      "evidence_type": "cross_function_contract",
      "confidence": "high",
      "mode": "state invariant",
      "oracle_hint": "use discard as inverse",
      "code": "..."
    }
  ],
  "bug_report": null,
  "adversarial_checks": [...]
}
```

Possible `status` values:

| Status | Meaning |
|---|---|
| `success` | One or more properties validated; final PBT emitted |
| `no_properties` | No property passed validation (or suite rejected as too weak) |
| `library_bug_found` | A property exposed a likely defect in the target library |

When `status` is `library_bug_found`, the `bug_report` field contains the property, reasoning, and error output as evidence for filing an issue.

## Bugs Found

The full bugs found are listed in [bugs](https://raw.githubusercontent.com/T90REAL/PROBE/main/bugs.txt)
