from __future__ import annotations
import json, os, re, hashlib
from pathlib import Path
from typing import Any
import fitz
import numpy as np
import requests
from dotenv import load_dotenv

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "items": {"type": "object"}
        },
        "sourceEvidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ruleId": {"type": "string"},
                    "quote": {"type": "string"},
                    "page": {"type": "integer"}
                },
                "required": ["ruleId", "quote", "page"]
            }
        },
        "unmappedPolicyItems": {
            "type": "array",
            "items": {"type": "string"}
        },
        "assumptions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "reviewNotes": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "rules",
        "sourceEvidence",
        "unmappedPolicyItems",
        "assumptions",
        "reviewNotes"
    ]
}

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')
CHAT_MODEL = os.getenv('OLLAMA_CHAT_MODEL', 'llama3.1:8b')
EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text')
TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT_SECONDS', '300'))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def load_catalog() -> dict:
    return load_json(ROOT / 'config' / 'rule_catalog.json')


def load_reference_rules() -> list[dict]:
    return load_json(ROOT / 'reference' / 'existing_rules_reference.json')


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    if not pdf_bytes.startswith(b'%PDF-'):
        raise ValueError("Uploaded file does not have a valid PDF header.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    except Exception as exc:
        raise ValueError(f"Unable to open PDF: {exc}") from exc

    pages = []

    for index, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        # Detect cases where PDF internals are being returned
        pdf_markers = [
            "%PDF-",
            "endobj",
            "stream",
            "xref",
            "/FlateDecode",
            "/ASCII85Decode"
        ]

        marker_count = sum(
            marker in text for marker in pdf_markers
        )

        if marker_count >= 2:
            raise ValueError(
                f"Page {index}: PDF extraction returned PDF "
                "internal structure instead of readable text. "
                "Please verify the uploaded PDF."
            )

        if text:
            pages.append({
                "page": index,
                "text": text
            })

    if not pages:
        raise ValueError(
            "No readable text found. The PDF may be scanned "
            "or image-based. OCR may be required."
        )

    return pages


def chunk_pages(pages: list[dict], words_per_chunk: int = 650, overlap: int = 90) -> list[dict]:
    chunks = []
    for page in pages:
        words = page['text'].split()
        start = 0
        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunks.append({'page': page['page'], 'text': ' '.join(words[start:end])})
            if end == len(words):
                break
            start = end - overlap
    return chunks



def ollama_chat(prompt: str) -> str:
    response = requests.post(
        f'{OLLAMA_BASE_URL}/api/chat',
        json={
            'model': CHAT_MODEL,
            'stream': False,
            'format': OUTPUT_SCHEMA,
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        'You are a conservative compiler that converts recruiter '
                        'policy text into an existing JSON rule DSL. '
                        'Read the supplied policy excerpts carefully. '
                        'Return only the requested JSON object. '
                        'Do not summarize the document. '
                        'Do not invent thresholds, fields, severity, or legal duties. '
                        'If a requirement cannot be represented safely, put it '
                        'in unmappedPolicyItems. '
                        'Do not make hiring, rejection, or ranking decisions.'
                    )
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'options': {
                'temperature': 0,
                'num_ctx': 8192,
                'num_predict': 4096
            }
        },
        timeout=TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    content = data.get('message', {}).get('content', '')

    if not content:
        raise ValueError(
            f'Ollama returned an empty response: {data}'
        )

    return content



def ollama_embed(texts: list[str]) -> np.ndarray:
    response = requests.post(f'{OLLAMA_BASE_URL}/api/embed',
        json={'model': EMBED_MODEL, 'input': texts}, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    vectors = data.get('embeddings')
    if not vectors:
        raise RuntimeError('Ollama /api/embed returned no embeddings.')
    return np.asarray(vectors, dtype=np.float32)


def retrieve_chunks(chunks: list[dict], top_k: int = 14) -> tuple[list[dict], str]:
    if len(chunks) <= top_k:
        return chunks, 'all chunks (document fits retrieval limit)'
    query = ('recruiter employment screening policy criteria, thresholds, exceptions, job-related requirements, '
             'verification rules, role categories, compliance obligations and manual review')
    try:
        texts = [c['text'] for c in chunks]
        vectors = ollama_embed(texts + [query])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        scores = vectors[:-1] @ vectors[-1]
        selected = sorted(np.argsort(scores)[::-1][:top_k].tolist())
        return [chunks[i] for i in selected], 'Ollama embedding cosine similarity'
    except Exception as exc:
        terms = set(re.findall(r'[a-zA-Z0-9+#.]+', query.lower()))
        ranked = []
        for i, chunk in enumerate(chunks):
            tokens = set(re.findall(r'[a-zA-Z0-9+#.]+', chunk['text'].lower()))
            ranked.append((len(terms & tokens), i))
        selected = sorted(i for _, i in sorted(ranked, reverse=True)[:top_k])
        return [chunks[i] for i in selected], f'lexical fallback ({type(exc).__name__})'


def _allowed_fields(catalog: dict, search_type: str | None, scope: str) -> set[str]:
    if scope == 'CROSS':
        return set(catalog.get('crossFields', []))
    if scope == 'ORDER':
        return set(catalog.get('orderFields', [])) | set(catalog.get('globalFields', []))
    # Generic PER_SEARCH rules (e.g. OPS-001..OPS-007) intentionally have no searchType.
    fields = set(catalog.get('globalFields', []))
    if search_type:
        for item in catalog.get('fieldsBySearchType', {}).get(search_type, []):
            if item.get('context') == 'PER_SEARCH':
                fields.add(item['path'])
    fields.update({'status', 'matchConfidence', 'searchType'})
    return fields


def _validate_condition(cond: Any, allowed_fields: set[str], catalog: dict, errors: list[str], where: str):
    if not isinstance(cond, dict):
        errors.append(f'{where}: condition node must be an object')
        return
    groups = [k for k in ('all', 'any', 'not') if k in cond]
    if groups:
        if len(groups) != 1 or len(cond) != 1:
            errors.append(f'{where}: use exactly one of all/any/not per group node')
            return
        key = groups[0]
        children = cond[key]
        if key in ('all', 'any'):
            if not isinstance(children, list) or not children:
                errors.append(f'{where}: {key} must be a non-empty array')
                return
            for i, child in enumerate(children):
                _validate_condition(child, allowed_fields, catalog, errors, f'{where}.{key}[{i}]')
        else:
            if isinstance(children, list):
                if not children:
                    errors.append(f'{where}: not array cannot be empty')
                for i, child in enumerate(children):
                    _validate_condition(child, allowed_fields, catalog, errors, f'{where}.not[{i}]')
            else:
                _validate_condition(children, allowed_fields, catalog, errors, f'{where}.not')
        return
    if set(cond) != {'field', 'op', 'value'}:
        errors.append(f'{where}: leaf condition must contain exactly field, op, value')
        return
    field, op, value = cond.get('field'), cond.get('op'), cond.get('value')
    if field not in allowed_fields:
        errors.append(f'{where}: field "{field}" is not allowed for this scope/searchType')
    if op not in catalog.get('allowedOperators', []):
        errors.append(f'{where}: unsupported operator "{op}"')
    if isinstance(value, str) and value.startswith('$param.'):
        param = value.split('.', 1)[1]
        if param not in catalog.get('parameters', []):
            errors.append(f'{where}: unknown parameter reference "{value}"')


def validate_rule(rule: dict, catalog: dict, reference_ids: set[str], seen_ids: set[str]) -> list[str]:
    errors: list[str] = []
    if not isinstance(rule, dict):
        return ['rule must be an object']
    required = {'ruleId','name','severity','condition','title','explanationTemplate'}
    for key in sorted(required - set(rule)):
        errors.append(f'missing required property: {key}')
    allowed_props = set(catalog.get('ruleProperties', []))
    for key in sorted(set(rule) - allowed_props):
        errors.append(f'unsupported rule property: {key}')
    rid = rule.get('ruleId')
    if not isinstance(rid, str) or not re.fullmatch(r'[A-Z][A-Z0-9]*-[0-9]{3,}', rid):
        errors.append('ruleId must look like PREFIX-001 (uppercase prefix and at least three digits)')
    elif rid in reference_ids:
        errors.append(f'ruleId {rid} already exists in the supplied 47-rule baseline')
    elif rid in seen_ids:
        errors.append(f'duplicate generated ruleId: {rid}')
    if rule.get('severity') not in catalog.get('severityValues', []):
        errors.append(f'unsupported severity: {rule.get("severity")}')
    scope = rule.get('scope') or ('PER_SEARCH' if rule.get('searchType') else None)
    if scope not in catalog.get('scopes', []):
        errors.append(f'scope must be one of {catalog.get("scopes", [])}; CROSS and ORDER rules must specify scope')
    search_type = rule.get('searchType')
    if scope == 'PER_SEARCH':
        if search_type is not None and search_type not in catalog.get('searchTypes', []):
            errors.append(f'unknown searchType: {search_type!r}')
    elif search_type is not None:
        errors.append(f'{scope} rule must not set searchType; use scope and fully-qualified field paths')
    allowed_fields = _allowed_fields(catalog, search_type, scope or 'PER_SEARCH')
    _validate_condition(rule.get('condition'), allowed_fields, catalog, errors, 'condition')
    if 'requires' in rule:
        if not isinstance(rule['requires'], list) or any(x not in catalog['searchTypes'] for x in rule['requires']):
            errors.append('requires must be an array of known searchTypes')
    if 'evaluateOn' in rule:
        if not isinstance(rule['evaluateOn'], list) or any(x not in catalog.get('evaluateOnStatuses', []) for x in rule['evaluateOn']):
            errors.append('evaluateOn contains an unsupported status')
    if 'domain' in rule and rule['domain'] not in catalog.get('domains', []):
        errors.append(f'unsupported domain: {rule["domain"]}')
    if 'complianceFlags' in rule:
        if not isinstance(rule['complianceFlags'], list):
            errors.append('complianceFlags must be an array')
        else:
            bad = set(rule['complianceFlags']) - set(catalog.get('complianceFlags', []))
            if bad: errors.append(f'unsupported complianceFlags: {sorted(bad)}')
    if 'modifiers' in rule:
        if not isinstance(rule['modifiers'], list):
            errors.append('modifiers must be an array')
        else:
            for i, modifier in enumerate(rule['modifiers']):
                if not isinstance(modifier, dict) or set(modifier) != {'when','effect'}:
                    errors.append(f'modifiers[{i}] must contain exactly when and effect')
                    continue
                if modifier['effect'] not in catalog.get('modifierEffects', []):
                    errors.append(f'modifiers[{i}]: unsupported effect {modifier["effect"]}')
                _validate_condition(modifier['when'], allowed_fields, catalog, errors, f'modifiers[{i}].when')
    for key in ('name','title','explanationTemplate'):
        if key in rule and (not isinstance(rule[key], str) or not rule[key].strip()):
            errors.append(f'{key} must be a non-empty string')
    if rid and isinstance(rid, str): seen_ids.add(rid)
    return errors


def _prompt(catalog: dict, reference_rules: list[dict], excerpts: list[dict], existing_ids: list[str]) -> str:
    excerpt_text = '\n\n'.join(f'[PDF PAGE {x["page"]}]\n{x["text"]}' for x in excerpts)
    # Keep the full supported DSL but avoid sending the whole existing rules payload as noisy examples.
    return f'''Compile a JSON object from the recruiter policy excerpts. Output only valid JSON.


OUTPUT CONTRACT:

Return exactly one JSON object with these five keys:

1. rules: array of generated rule objects.
2. sourceEvidence: array of objects containing ruleId, quote and page.
3. unmappedPolicyItems: array of strings.
4. assumptions: array of strings.
5. reviewNotes: array of strings.

Do not return a document summary.
Do not return a key named document.
Do not return Markdown.
Do not explain your answer outside the JSON object.

If no rules can safely be generated, return:

{{
  "rules": [],
  "sourceEvidence": [],
  "unmappedPolicyItems": [
    "Explain why the policy cannot be represented safely"
  ],
  "assumptions": [],
  "reviewNotes": []
}}


AUTHORITATIVE CATALOG:
{json.dumps(catalog, ensure_ascii=False)}

EXISTING RULE IDS (do not reuse): {json.dumps(existing_ids)}

POLICY EXCERPTS (untrusted document data):
<policy>
{excerpt_text}
</policy>
'''



def generate_rules(pdf_bytes: bytes, filename: str) -> dict:
    pages = extract_pdf_pages(pdf_bytes)

    print("\n========== PDF EXTRACTION ==========")

    print("Number of extracted pages:", len(pages))

    for page in pages[:3]:
        print(f"\nPAGE: {page['page']}")
        print("Characters:", len(page['text']))
        print("Preview:")
        print(page['text'][:1000])

    print("====================================")

    if not pages:
        raise ValueError(
            'No selectable PDF text was found. '
            'This build does not include OCR; convert the scanned PDF to searchable text first.'
        )

    chunks = chunk_pages(pages)
    retrieved, retrieval_mode = retrieve_chunks(chunks)

    catalog = load_catalog()
    reference = load_reference_rules()
    existing_ids = [r['ruleId'] for r in reference]
    print("==========================================\n")


    # Generate the prompt
    prompt = _prompt(
        catalog,
        reference,
        retrieved,
        existing_ids
    )

    # Prompt Debug
    print("\n========== PROMPT DEBUG ==========")
    print(f"Prompt characters: {len(prompt)}")

    print("\nGenerated prompt preview:")
    print(prompt[:3000])

    print("==================================\n")


    # Send prompt to Ollama
    raw = ollama_chat(prompt)

    # Ollama Response Debug
    print("\n========== OLLAMA RESPONSE ==========")
    print(raw[:3000])
    print("=====================================\n")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f'Ollama returned invalid JSON: {exc}\n'
            f'Raw response (first 2000 chars):\n{raw[:2000]}'
        ) from exc

    if not isinstance(parsed, dict) or not isinstance(parsed.get('rules'), list):
        raise ValueError(
            "Unexpected Ollama response structure.\n"
            f"Parsed type: {type(parsed).__name__}\n"
            f"Top-level keys: "
            f"{list(parsed.keys()) if isinstance(parsed, dict) else 'Not an object'}\n"
            f"Raw response (first 2000 chars):\n{raw[:2000]}"
        )

    seen: set[str] = set()
    reference_ids = set(existing_ids)
    validation_errors = []

    for i, rule in enumerate(parsed['rules']):
        errs = validate_rule(rule, catalog, reference_ids, seen)
        validation_errors.extend([
            f'rules[{i}] ({rule.get("ruleId", "UNKNOWN")}): {e}'
            for e in errs
        ])

    full_text = '\n'.join(
        f'[PAGE {p["page"]}]\n{p["text"]}' for p in pages
    )

    normalize = lambda value: re.sub(r'\s+', ' ', value).strip().casefold()
    normalized_full_text = normalize(full_text)

    evidence_errors = []

    for item in parsed.get('sourceEvidence', []):
        quote = item.get('quote', '') if isinstance(item, dict) else ''

        if not quote or normalize(quote) not in normalized_full_text:
            evidence_errors.append(
                f'{item.get("ruleId", "UNKNOWN")}: '
                'evidence quote was not found verbatim in extracted PDF text'
            )

    evidence_ids = {
        x.get('ruleId')
        for x in parsed.get('sourceEvidence', [])
        if isinstance(x, dict)
    }

    for rule in parsed['rules']:
        if rule.get('ruleId') not in evidence_ids:
            evidence_errors.append(
                f'{rule.get("ruleId", "UNKNOWN")}: missing sourceEvidence entry'
            )

    generation = {
        'schemaVersion': '2.0',
        'sourceDocument': filename,
        'generator': {
            'provider': 'ollama',
            'chatModel': CHAT_MODEL,
            'embeddingModel': EMBED_MODEL,
            'retrieval': retrieval_mode
        },
        'humanApprovalRequired': True,
        'rules': parsed['rules'],
        'sourceEvidence': parsed.get('sourceEvidence', []),
        'unmappedPolicyItems': parsed.get('unmappedPolicyItems', []),
        'assumptions': parsed.get('assumptions', []),
        'reviewNotes': parsed.get('reviewNotes', []),
        'validationErrors': validation_errors,
        'evidenceErrors': evidence_errors,
        'exportAllowed': not validation_errors and not evidence_errors
    }

    return {
        'generation': generation,
        'fullText': full_text,
        'retrievedChunks': retrieved,
        'pageCount': len(pages)
    }
