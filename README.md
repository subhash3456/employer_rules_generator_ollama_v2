# Recruiter Policy → Existing Java `rules.json` (Ollama, local)

This version is aligned to the rule DSL and API examples supplied for your Java application. It is a local Python/Streamlit tool: upload a recruiter policy PDF, extract policy requirements with Ollama, compile them into your existing rule JSON shape, validate the output, and download a draft `rules.json` array.

## What is included

- `reference/existing_rules_reference.json`: the 47 rules from your supplied `rules.json`, preserved as the baseline/reference.
- `config/rule_catalog.json`: extracted supported search types, fields, operators, condition grammar, scopes, severity values, domains, parameters, `evaluateOn` values, modifier effects and compliance flags.
- `config/candidate_report_schema.json`: mapping between the `CandidateReportResponse` API shape and the rule-engine field aliases.
- `config/generation_config.json`: generation defaults and safety rules.
- `app/pipeline.py`: PDF extraction, Ollama calls, retrieval, DSL validation, duplicate-ID checks and source-quote verification.
- `app/main.py`: Streamlit UI.
- `tests/test_baseline.py`: checks that all 47 supplied baseline rules pass the validator.
- `samples/sample_recruiter_policy.pdf`: sample PDF from the previous starter project.
- `samples/redacted_candidate_report_example.json`: synthetic/redacted example using the supplied report structure. No candidate PII is included.

## 1. Prerequisites

- Python 3.10 or newer
- Ollama installed and running locally
- Enough RAM for the selected local model

Open a terminal and pull the models (defaults):

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

You can use another Ollama chat model by changing `OLLAMA_CHAT_MODEL` in `.env`. The chat model should follow JSON output reliably. Embeddings are used for passage retrieval; if `/api/embed` is unavailable, the app falls back to lexical retrieval.

## 2. Create the Python environment

Windows PowerShell:

```powershell
cd path\to\employer_rules_generator_ollama_v2
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cd employer_rules_generator_ollama_v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 3. Run the compatibility test

```bash
python -m unittest discover -s tests
```

All 47 reference rules should pass. If you edit the catalog or validator, rerun this test.

## 4. Start the app

```bash
streamlit run app/main.py
```

The browser UI normally opens at `http://localhost:8501`.

## 5. Output contract

The downloadable `rules.json` is a **JSON array of rule definitions**, not the runtime selection response returned by `/api/orders/{orderId}/risk/rules`.

The compiler supports the shapes observed in your 47-rule baseline:

- scopes: `PER_SEARCH`, `CROSS`, `ORDER`
- conditions: leaf `{field, op, value}` and nested `all`, `any`, `not`
- operators: only those present in the supplied baseline
- optional properties: `domain`, `scope`, `requires`, `evaluateOn`, `modifiers`, `complianceFlags`
- modifier effect: `ESCALATE`
- parameter references: only those found in the supplied baseline

Runtime properties such as `suggested`, `runnable`, and `waitingFor` are intentionally not generated. Your Java backend calculates those from the current order/report state.

## 6. Candidate report mapping

The supplied report response exposes `searchTypes[]`, with `result`, `status`, `matchConfidence`, and `details`. The existing rule DSL uses aliases such as `attributes.foo`, `result`, and `job.roleCategories`.

The mapping contract is in `config/candidate_report_schema.json`. Typical mappings:

| Existing rule DSL path | Candidate report path |
|---|---|
| `attributes.matchType` | `searchTypes[SEARCH_TYPE].details.matchType` |
| `result` | `searchTypes[SEARCH_TYPE].result` |
| `status` | `searchTypes[SEARCH_TYPE].status` |
| `matchConfidence` | `searchTypes[SEARCH_TYPE].matchConfidence` |
| `job.roleCategories` | `application.roleCategories` |
| `order.daysToStart` | `application.daysToStart` |
| `order.pendingCount` | derived from `coverage.pending.length` |
| `order.notOrderedCount` | derived from `coverage.notOrdered.length` |
| `meta.*` | not directly present in the supplied DTO/sample; requires rule-engine runtime context |

**Important:** The sample report is not an exhaustive schema of all possible `details` fields. A field not seen in one sample is not automatically invalid. Conversely, a field used in the rule DSL may require runtime enrichment not shown in the DTO.

## 7. Validation behavior

The app blocks the validated download if any of these fail:

- unknown property, search type, scope, operator, severity, domain or compliance flag
- condition field outside the catalog for the rule's scope/search type
- malformed nested condition or modifier
- unknown parameter reference
- duplicate rule ID or reuse of an existing rule ID
- missing source-evidence entry or a quote not found verbatim in extracted PDF text

The UI may still let you download a `rules.draft.json` for debugging when validation fails. Do not load that draft into your Java service.

## 8. Recommended Java integration flow

1. Download the generated `rules.json` array.
2. Review every rule against the cited policy page/quote.
3. Have your compliance/legal reviewer approve the policy interpretation, severity, and any compliance flags.
4. Confirm all generated field paths are supported by your Java rule engine and its runtime context adapter.
5. Run your Java rule-engine unit/integration tests with representative reports, including missing/pending/unable-to-verify states.
6. Deploy only through your normal configuration/versioning process with rollback.

The Python app does not call your `localhost:8080` Java API and does not change the Java backend. It generates a file for review and integration.

## Limitations

- Scanned/image-only PDFs are not OCR'd in this build; provide searchable PDFs.
- The catalog is grounded in the supplied 47 rules and the two supplied API response examples. If your Java engine supports additional fields/operators, add them explicitly to the catalog and tests before asking the model to use them.
- A successful structural/evidence check does not establish legal compliance or policy correctness.
- Employment-screening rules are high impact. This tool must not automatically hire, reject, or rank candidates. Human review and jurisdiction-specific legal/compliance approval remain mandatory.
