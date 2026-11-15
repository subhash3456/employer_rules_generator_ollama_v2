from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import fitz
import numpy as np
import requests
from dotenv import load_dotenv


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

# Supports either:
#   project_root/pipeline.py
# or:
#   project_root/some_package/pipeline.py
_THIS_DIR = Path(__file__).resolve().parent
ROOT = _THIS_DIR if (_THIS_DIR / "config").exists() else _THIS_DIR.parent

load_dotenv(ROOT / ".env")

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
).rstrip("/")

CHAT_MODEL = os.getenv(
    "OLLAMA_CHAT_MODEL",
    "qwen2.5:3b"
)

EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "nomic-embed-text"
)

TIMEOUT = int(
    os.getenv("OLLAMA_TIMEOUT_SECONDS", "300")
)


# ============================================================
# STAGE 2: EXISTING JAVA RULE DSL OUTPUT SCHEMA
# ============================================================

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


# ============================================================
# STAGE 1: POLICY REQUIREMENT EXTRACTION SCHEMA
# ============================================================

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "category": {"type": "string"},
                    "role": {
                        "type": ["string", "null"]
                    },
                    "requirementType": {"type": "string"},
                    "statement": {"type": "string"},
                    "threshold": {
                        "type": ["number", "null"]
                    },
                    "unit": {
                        "type": ["string", "null"]
                    },
                    "sourceEvidence": {
                        "type": "object",
                        "properties": {
                            "quote": {"type": "string"},
                            "page": {"type": "integer"}
                        },
                        "required": ["quote", "page"]
                    }
                },
                "required": [
                    "id",
                    "category",
                    "role",
                    "requirementType",
                    "statement",
                    "threshold",
                    "unit",
                    "sourceEvidence"
                ]
            }
        },
        "unresolvedItems": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "requirements",
        "unresolvedItems"
    ]
}


# ============================================================
# COMMON JSON / FILE HELPERS
# ============================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def load_catalog() -> dict:
    return load_json(ROOT / "config" / "rule_catalog.json")


def load_reference_rules() -> list[dict]:
    return load_json(
        ROOT / "reference" / "existing_rules_reference.json"
    )


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_pages(pdf_bytes: bytes) -> list[dict]:
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError(
            "Uploaded file does not have a valid PDF header."
        )

    try:
        doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )
    except Exception as exc:
        raise ValueError(
            f"Unable to open PDF: {exc}"
        ) from exc

    pages = []

    try:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()

            pdf_markers = [
                "%PDF-",
                "endobj",
                "stream",
                "xref",
                "/FlateDecode",
                "/ASCII85Decode"
            ]

            marker_count = sum(
                marker in text
                for marker in pdf_markers
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
    finally:
        doc.close()

    if not pages:
        raise ValueError(
            "No readable text found. The PDF may be scanned "
            "or image-based. OCR may be required."
        )

    return pages


def chunk_pages(
    pages: list[dict],
    words_per_chunk: int = 650,
    overlap: int = 90
) -> list[dict]:

    if words_per_chunk <= 0:
        raise ValueError("words_per_chunk must be greater than zero.")

    if overlap < 0 or overlap >= words_per_chunk:
        raise ValueError(
            "overlap must be >= 0 and smaller than words_per_chunk."
        )

    chunks = []

    for page in pages:
        words = page["text"].split()
        start = 0

        while start < len(words):
            end = min(
                start + words_per_chunk,
                len(words)
            )

            chunks.append({
                "page": page["page"],
                "text": " ".join(words[start:end])
            })

            if end == len(words):
                break

            start = end - overlap

    return chunks


# ============================================================
# OLLAMA CHAT AND EMBEDDINGS
# ============================================================

def ollama_chat(
    prompt: str,
    output_schema: dict | None = None,
    system_prompt: str | None = None
) -> str:

    if output_schema is None:
        output_schema = OUTPUT_SCHEMA

    if system_prompt is None:
        system_prompt = (
            "You are a conservative compiler that converts recruiter "
            "policy text into an existing JSON rule DSL. "
            "Return only the requested JSON object. "
            "Do not invent thresholds, fields, severity, or legal duties. "
            "If a requirement cannot be represented safely, put it "
            "in unmappedPolicyItems. "
            "Do not make hiring, rejection, or ranking decisions."
        )

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "stream": False,
                "format": output_schema,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "options": {
                    "temperature": 0,
                    "num_ctx": 8192,
                    "num_predict": 4096
                }
            },
            timeout=TIMEOUT
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Unable to communicate with Ollama at "
            f"{OLLAMA_BASE_URL}: {exc}"
        ) from exc

    data = response.json()
    content = data.get("message", {}).get("content", "")

    if not content:
        raise ValueError(
            f"Ollama returned an empty response: {data}"
        )

    return content


def ollama_embed(texts: list[str]) -> np.ndarray:
    if not texts:
        raise ValueError("Cannot create embeddings for an empty text list.")

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": EMBED_MODEL,
                "input": texts
            },
            timeout=TIMEOUT
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Unable to create Ollama embeddings: {exc}"
        ) from exc

    data = response.json()
    vectors = data.get("embeddings")

    if not vectors:
        raise RuntimeError(
            "Ollama /api/embed returned no embeddings."
        )

    return np.asarray(
        vectors,
        dtype=np.float32
    )


def retrieve_chunks(
    chunks: list[dict],
    top_k: int = 14
) -> tuple[list[dict], str]:

    if not chunks:
        return [], "no chunks"

    if len(chunks) <= top_k:
        return (
            chunks,
            "all chunks (document fits retrieval limit)"
        )

    query = (
        "recruiter employment screening policy criteria, thresholds, "
        "exceptions, job-related requirements, verification rules, "
        "role categories, compliance obligations and manual review"
    )

    try:
        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        vectors = ollama_embed(
            texts + [query]
        )

        vectors /= (
            np.linalg.norm(
                vectors,
                axis=1,
                keepdims=True
            ) + 1e-12
        )

        scores = vectors[:-1] @ vectors[-1]

        selected = sorted(
            np.argsort(scores)[::-1][:top_k].tolist()
        )

        return (
            [chunks[index] for index in selected],
            "Ollama embedding cosine similarity"
        )

    except Exception as exc:
        terms = set(
            re.findall(
                r"[a-zA-Z0-9+#.]+",
                query.lower()
            )
        )

        ranked = []

        for index, chunk in enumerate(chunks):
            tokens = set(
                re.findall(
                    r"[a-zA-Z0-9+#.]+",
                    chunk["text"].lower()
                )
            )

            ranked.append((
                len(terms & tokens),
                index
            ))

        selected = sorted(
            index
            for _, index in sorted(
                ranked,
                reverse=True
            )[:top_k]
        )

        return (
            [chunks[index] for index in selected],
            f"lexical fallback ({type(exc).__name__})"
        )


# ============================================================
# STAGE 1: POLICY REQUIREMENT EXTRACTION
# ============================================================

def extract_policy_requirements(full_text: str) -> dict:
    """
    Extract explicit policy requirements from the complete PDF text.

    This stage does not know about or generate the Java rule DSL.
    """

    prompt = f"""
Extract EVERY explicit requirement from the recruiter policy.

YOUR TASK:
Policy requirement extraction only.

Do not generate Java DSL rules.
Do not decide whether requirements are supported by the Java engine.
Do not invent requirements or missing information.

EXTRACTION RULES:

1. Extract every explicit policy obligation.
2. Split compound requirements into independent requirements.
3. Preserve the original meaning of each requirement.
4. Capture numeric and non-numeric requirements.
5. For a numeric requirement, extract its threshold and unit.
6. If there is no numeric threshold, use null.
7. Preserve the applicable role, if one is specified.
8. Every requirement must have a verbatim evidence quote.
9. The page number must match the supplied PDF page marker.
10. If a requirement cannot be confidently extracted,
    add an explanation to unresolvedItems.
11. Do not omit procedural requirements such as verification,
    approvals, interviews, and documentation.
12. Do not convert recommendations into mandatory requirements.
13. Do not infer requirements from general industry practices.
14. Do not combine different thresholds into one requirement.
15. Preserve the difference between mandatory language
    ("must", "required") and advisory language ("should", "may").

SPLITTING EXAMPLE:

Policy:
Engineering Manager: Minimum 8 years of relevant experience,
including 2 years of leadership experience.

This must produce TWO independent requirements:
- Minimum 8 years of relevant experience.
- Minimum 2 years of leadership experience.

Return only the JSON structure required by the schema.

POLICY TEXT:

<policy>
{full_text}
</policy>
"""

    system_prompt = (
        "You are a precise recruiter policy extraction engine. "
        "Extract all explicit requirements without summarizing or "
        "inventing information. Split compound requirements into "
        "independent requirements. Preserve exact evidence quotes "
        "and original PDF page numbers. Return only valid JSON "
        "matching the supplied schema."
    )

    raw = ollama_chat(
        prompt=prompt,
        output_schema=EXTRACTION_SCHEMA,
        system_prompt=system_prompt
    )

    print("\n========== STAGE 1 RAW RESPONSE ==========")
    print(raw[:5000])
    print("==========================================\n")

    try:
        parsed = json.loads(raw)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Stage 1 returned invalid JSON: {exc}\n"
            f"Raw response:\n{raw[:3000]}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            "Stage 1 response must be a JSON object."
        )

    requirements = parsed.get("requirements")
    unresolved_items = parsed.get("unresolvedItems")

    if not isinstance(requirements, list):
        raise ValueError(
            "Stage 1 response does not contain a valid requirements array."
        )

    if not isinstance(unresolved_items, list):
        raise ValueError(
            "Stage 1 response does not contain a valid unresolvedItems array."
        )

    # Assign stable IDs in Python instead of trusting model-generated IDs.
    for index, requirement in enumerate(
        requirements,
        start=1
    ):

        if not isinstance(requirement, dict):
            raise ValueError(
                f"Requirement at index {index} is not a JSON object."
            )

        requirement["id"] = f"REQ-{index:03d}"

        required_fields = [
            "category",
            "role",
            "requirementType",
            "statement",
            "threshold",
            "unit",
            "sourceEvidence"
        ]

        for field in required_fields:
            if field not in requirement:
                raise ValueError(
                    f"{requirement['id']} is missing field: {field}"
                )

        for field in ("category", "requirementType", "statement"):
            if not isinstance(requirement[field], str):
                raise ValueError(
                    f"{requirement['id']} has invalid {field}."
                )

            if not requirement[field].strip():
                raise ValueError(
                    f"{requirement['id']} has an empty {field}."
                )

        role = requirement["role"]

        if role is not None and not isinstance(role, str):
            raise ValueError(
                f"{requirement['id']} has an invalid role."
            )

        unit = requirement["unit"]

        if unit is not None and not isinstance(unit, str):
            raise ValueError(
                f"{requirement['id']} has an invalid unit."
            )

        threshold = requirement["threshold"]

        if (
            threshold is not None
            and (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
            )
        ):
            raise ValueError(
                f"{requirement['id']} has an invalid threshold."
            )

        evidence = requirement["sourceEvidence"]

        if not isinstance(evidence, dict):
            raise ValueError(
                f"{requirement['id']} has invalid sourceEvidence."
            )

        quote = evidence.get("quote")
        page = evidence.get("page")

        if not isinstance(quote, str) or not quote.strip():
            raise ValueError(
                f"{requirement['id']} has an empty evidence quote."
            )

        if (
            not isinstance(page, int)
            or isinstance(page, bool)
            or page < 1
        ):
            raise ValueError(
                f"{requirement['id']} has an invalid page number."
            )

    for item in unresolved_items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                "unresolvedItems contains an invalid or empty item."
            )

    return {
        "requirements": requirements,
        "unresolvedItems": unresolved_items
    }


def validate_extraction_evidence(
    requirements: list[dict],
    pages: list[dict]
) -> list[str]:
    """
    Check each Stage 1 evidence quote against its claimed PDF page.
    """

    pages_by_number = {
        page["page"]: normalize_text(page["text"])
        for page in pages
    }

    errors = []

    for requirement in requirements:
        requirement_id = requirement.get(
            "id",
            "UNKNOWN"
        )

        evidence = requirement.get(
            "sourceEvidence",
            {}
        )

        quote = evidence.get("quote", "")
        page_number = evidence.get("page")

        if not quote:
            errors.append(
                f"{requirement_id}: missing evidence quote."
            )
            continue

        page_text = pages_by_number.get(page_number)

        if page_text is None:
            errors.append(
                f"{requirement_id}: PDF page {page_number} does not exist."
            )
            continue

        if normalize_text(quote) not in page_text:
            errors.append(
                f"{requirement_id}: evidence quote was not found "
                f"on PDF page {page_number}."
            )

    return errors


# ============================================================
# EXISTING JAVA DSL VALIDATION
# ============================================================

def _allowed_fields(
    catalog: dict,
    search_type: str | None,
    scope: str
) -> set[str]:

    if scope == "CROSS":
        return set(catalog.get("crossFields", []))

    if scope == "ORDER":
        return (
            set(catalog.get("orderFields", []))
            | set(catalog.get("globalFields", []))
        )

    # Generic PER_SEARCH rules intentionally have no searchType.
    fields = set(catalog.get("globalFields", []))

    if search_type:
        for item in catalog.get(
            "fieldsBySearchType",
            {}
        ).get(search_type, []):

            if item.get("context") == "PER_SEARCH":
                fields.add(item["path"])

    fields.update({
        "status",
        "matchConfidence",
        "searchType"
    })

    return fields


def _validate_condition(
    cond: Any,
    allowed_fields: set[str],
    catalog: dict,
    errors: list[str],
    where: str
):
    if not isinstance(cond, dict):
        errors.append(
            f"{where}: condition node must be an object"
        )
        return

    groups = [
        key
        for key in ("all", "any", "not")
        if key in cond
    ]

    if groups:
        if len(groups) != 1 or len(cond) != 1:
            errors.append(
                f"{where}: use exactly one of all/any/not per group node"
            )
            return

        key = groups[0]
        children = cond[key]

        if key in ("all", "any"):
            if not isinstance(children, list) or not children:
                errors.append(
                    f"{where}: {key} must be a non-empty array"
                )
                return

            for index, child in enumerate(children):
                _validate_condition(
                    child,
                    allowed_fields,
                    catalog,
                    errors,
                    f"{where}.{key}[{index}]"
                )

        else:
            if isinstance(children, list):
                if not children:
                    errors.append(
                        f"{where}: not array cannot be empty"
                    )

                for index, child in enumerate(children):
                    _validate_condition(
                        child,
                        allowed_fields,
                        catalog,
                        errors,
                        f"{where}.not[{index}]"
                    )

            else:
                _validate_condition(
                    children,
                    allowed_fields,
                    catalog,
                    errors,
                    f"{where}.not"
                )

        return

    if set(cond) != {"field", "op", "value"}:
        errors.append(
            f"{where}: leaf condition must contain exactly field, op, value"
        )
        return

    field = cond.get("field")
    operator = cond.get("op")
    value = cond.get("value")

    if field not in allowed_fields:
        errors.append(
            f'{where}: field "{field}" is not allowed '
            "for this scope/searchType"
        )

    if operator not in catalog.get("allowedOperators", []):
        errors.append(
            f'{where}: unsupported operator "{operator}"'
        )

    if isinstance(value, str) and value.startswith("$param."):
        parameter = value.split(".", 1)[1]

        if parameter not in catalog.get("parameters", []):
            errors.append(
                f'{where}: unknown parameter reference "{value}"'
            )


def validate_rule(
    rule: dict,
    catalog: dict,
    reference_ids: set[str],
    seen_ids: set[str]
) -> list[str]:

    errors: list[str] = []

    if not isinstance(rule, dict):
        return ["rule must be an object"]

    required = {
        "ruleId",
        "name",
        "severity",
        "condition",
        "title",
        "explanationTemplate"
    }

    for key in sorted(required - set(rule)):
        errors.append(
            f"missing required property: {key}"
        )

    allowed_props = set(
        catalog.get("ruleProperties", [])
    )

    for key in sorted(set(rule) - allowed_props):
        errors.append(
            f"unsupported rule property: {key}"
        )

    rule_id = rule.get("ruleId")

    if (
        not isinstance(rule_id, str)
        or not re.fullmatch(
            r"[A-Z][A-Z0-9]*-[0-9]{3,}",
            rule_id
        )
    ):
        errors.append(
            "ruleId must look like PREFIX-001 "
            "(uppercase prefix and at least three digits)"
        )

    elif rule_id in reference_ids:
        errors.append(
            f"ruleId {rule_id} already exists in the supplied baseline"
        )

    elif rule_id in seen_ids:
        errors.append(
            f"duplicate generated ruleId: {rule_id}"
        )

    if rule.get("severity") not in catalog.get(
        "severityValues",
        []
    ):
        errors.append(
            f'unsupported severity: {rule.get("severity")}'
        )

    scope = rule.get("scope") or (
        "PER_SEARCH"
        if rule.get("searchType")
        else None
    )

    if scope not in catalog.get("scopes", []):
        errors.append(
            f'scope must be one of {catalog.get("scopes", [])}; '
            "CROSS and ORDER rules must specify scope"
        )

    search_type = rule.get("searchType")

    if scope == "PER_SEARCH":
        if (
            search_type is not None
            and search_type not in catalog.get("searchTypes", [])
        ):
            errors.append(
                f"unknown searchType: {search_type!r}"
            )

    elif search_type is not None:
        errors.append(
            f"{scope} rule must not set searchType; "
            "use scope and fully-qualified field paths"
        )

    allowed_fields = _allowed_fields(
        catalog,
        search_type,
        scope or "PER_SEARCH"
    )

    _validate_condition(
        rule.get("condition"),
        allowed_fields,
        catalog,
        errors,
        "condition"
    )

    if "requires" in rule:
        if (
            not isinstance(rule["requires"], list)
            or any(
                item not in catalog["searchTypes"]
                for item in rule["requires"]
            )
        ):
            errors.append(
                "requires must be an array of known searchTypes"
            )

    if "evaluateOn" in rule:
        if (
            not isinstance(rule["evaluateOn"], list)
            or any(
                item not in catalog.get("evaluateOnStatuses", [])
                for item in rule["evaluateOn"]
            )
        ):
            errors.append(
                "evaluateOn contains an unsupported status"
            )

    if (
        "domain" in rule
        and rule["domain"] not in catalog.get("domains", [])
    ):
        errors.append(
            f'unsupported domain: {rule["domain"]}'
        )

    if "complianceFlags" in rule:
        if not isinstance(rule["complianceFlags"], list):
            errors.append(
                "complianceFlags must be an array"
            )
        else:
            bad = (
                set(rule["complianceFlags"])
                - set(catalog.get("complianceFlags", []))
            )

            if bad:
                errors.append(
                    f"unsupported complianceFlags: {sorted(bad)}"
                )

    if "modifiers" in rule:
        if not isinstance(rule["modifiers"], list):
            errors.append(
                "modifiers must be an array"
            )
        else:
            for index, modifier in enumerate(rule["modifiers"]):
                if (
                    not isinstance(modifier, dict)
                    or set(modifier) != {"when", "effect"}
                ):
                    errors.append(
                        f"modifiers[{index}] must contain exactly when and effect"
                    )
                    continue

                if modifier["effect"] not in catalog.get(
                    "modifierEffects",
                    []
                ):
                    errors.append(
                        f"modifiers[{index}]: unsupported effect "
                        f"{modifier['effect']}"
                    )

                _validate_condition(
                    modifier["when"],
                    allowed_fields,
                    catalog,
                    errors,
                    f"modifiers[{index}].when"
                )

    for key in (
        "name",
        "title",
        "explanationTemplate"
    ):
        if key in rule and (
            not isinstance(rule[key], str)
            or not rule[key].strip()
        ):
            errors.append(
                f"{key} must be a non-empty string"
            )

    if isinstance(rule_id, str) and rule_id:
        seen_ids.add(rule_id)

    return errors


# ============================================================
# STAGE 2: JAVA DSL COMPILATION PROMPT
# ============================================================

def _prompt(
    catalog: dict,
    reference_rules: list[dict],
    excerpts: list[dict],
    existing_ids: list[str],
    extracted_requirements: list[dict]
) -> str:

    excerpt_text = "\n\n".join(
        f'[PDF PAGE {item["page"]}]\n{item["text"]}'
        for item in excerpts
    )

    return f"""
You are a conservative compiler that maps extracted recruiter
policy requirements to an EXISTING Java JSON rule DSL.

YOUR TASK IS RULE COMPILATION ONLY.

The requirements below were extracted in Stage 1.
Treat them as the primary representation of policy meaning.

RULES:

1. Generate only rules supported by the authoritative catalog.
2. Do not invent Java DSL fields, operators, search types or scopes.
3. Do not invent thresholds or severity values.
4. Do not convert procedural requirements into unsupported rules.
5. Every generated rule must have exact supporting source evidence.
6. Use the original PDF excerpts to verify evidence quotes and pages.
7. Explain each unsupported requirement individually in unmappedPolicyItems.
8. Do not reuse existing rule IDs.
9. Do not make hiring, rejection, ranking or eligibility decisions.
10. An empty rules array is allowed when no extracted requirement
    can safely be represented.
11. Do not treat the absence of a generated rule as proof that
    the policy requirement does not exist.
12. Preserve the distinction between mandatory and advisory language.
13. Generated rules are drafts and require authorized human review.

OUTPUT CONTRACT:

Return one JSON object with these keys:
- rules: array of supported Java DSL rules
- sourceEvidence: array of ruleId, quote and page objects
- unmappedPolicyItems: array of explanations for unsupported requirements
- assumptions: array of assumptions, normally empty
- reviewNotes: array of human review notes

AUTHORITATIVE CATALOG:

{json.dumps(catalog, ensure_ascii=False)}

EXISTING RULE IDS (do not reuse):

{json.dumps(existing_ids)}

EXTRACTED POLICY REQUIREMENTS:

{json.dumps(extracted_requirements, ensure_ascii=False)}

ORIGINAL PDF EXCERPTS FOR EVIDENCE VERIFICATION:

<policy>
{excerpt_text}
</policy>
"""


# ============================================================
# COMPLETE TWO-STAGE PIPELINE
# ============================================================

def generate_rules(
    pdf_bytes: bytes,
    filename: str
) -> dict:
    """
    Full pipeline:

    1. Extract text from PDF.
    2. Extract structured policy requirements (Stage 1).
    3. Validate Stage 1 evidence against source PDF pages.
    4. Compile verified requirements into Java DSL (Stage 2).
    5. Validate generated DSL rules and their evidence.
    6. Return generation metadata for the Streamlit UI.
    """

    # --------------------------------------------------------
    # STEP 1: PDF TEXT EXTRACTION
    # --------------------------------------------------------

    pages = extract_pdf_pages(pdf_bytes)

    print("\n========== PDF EXTRACTION ==========")
    print("Number of extracted pages:", len(pages))

    for page in pages[:3]:
        print(f"\nPAGE: {page['page']}")
        print("Characters:", len(page["text"]))
        print("Preview:")
        print(page["text"][:1000])

    print("====================================\n")

    if not pages:
        raise ValueError(
            "No selectable PDF text was found. "
            "This build does not include OCR; convert scanned PDFs "
            "to searchable text first."
        )

    full_text = "\n".join(
        f'[PAGE {page["page"]}]\n{page["text"]}'
        for page in pages
    )

    # --------------------------------------------------------
    # STEP 2: STAGE 1 - POLICY REQUIREMENT EXTRACTION
    # --------------------------------------------------------

    print("\n========== STAGE 1: EXTRACTION ==========")

    extraction = extract_policy_requirements(
        full_text
    )

    extracted_requirements = extraction["requirements"]
    unresolved_items = extraction["unresolvedItems"]

    print(
        "Extracted requirements:",
        len(extracted_requirements)
    )

    for requirement in extracted_requirements:
        print(
            requirement["id"],
            "|",
            requirement["category"],
            "|",
            requirement["statement"]
        )

    print("\nUnresolved items:")
    for item in unresolved_items:
        print("-", item)

    print("========================================\n")

    # --------------------------------------------------------
    # STEP 3: VALIDATE STAGE 1 EVIDENCE
    # --------------------------------------------------------

    extraction_errors = validate_extraction_evidence(
        extracted_requirements,
        pages
    )

    print("\n========== STAGE 1 EVIDENCE VALIDATION ==========")

    for error in extraction_errors:
        print(error)

    print(
        "Extraction evidence errors:",
        len(extraction_errors)
    )

    print("=================================================\n")

    # Only pass requirements with source quotes that were
    # verified against the exact page claimed by the model.
    pages_by_number = {
        page["page"]: normalize_text(page["text"])
        for page in pages
    }

    verified_requirements = []

    for requirement in extracted_requirements:
        evidence = requirement["sourceEvidence"]
        page_number = evidence["page"]
        quote = normalize_text(evidence["quote"])

        page_text = pages_by_number.get(page_number, "")

        if quote and quote in page_text:
            verified_requirements.append(requirement)

    # --------------------------------------------------------
    # STEP 4: LOAD JAVA DSL CATALOG AND PREPARE RETRIEVAL
    # --------------------------------------------------------

    chunks = chunk_pages(pages)

    retrieved, retrieval_mode = retrieve_chunks(chunks)

    catalog = load_catalog()
    reference_rules = load_reference_rules()

    existing_ids = [
        rule["ruleId"]
        for rule in reference_rules
        if isinstance(rule, dict) and "ruleId" in rule
    ]

    print("\n========== STAGE 2 INPUT ==========")
    print("Verified requirements:", len(verified_requirements))
    print("Retrieval mode:", retrieval_mode)
    print("==================================\n")

    # --------------------------------------------------------
    # STEP 5: STAGE 2 - JAVA DSL RULE COMPILATION
    # --------------------------------------------------------

    prompt = _prompt(
        catalog,
        reference_rules,
        retrieved,
        existing_ids,
        verified_requirements
    )

    print("\n========== STAGE 2 PROMPT DEBUG ==========")
    print("Prompt characters:", len(prompt))
    print(prompt[:3000])
    print("==========================================\n")

    raw = ollama_chat(
        prompt=prompt,
        output_schema=OUTPUT_SCHEMA
    )

    print("\n========== STAGE 2 OLLAMA RESPONSE ==========")
    print(raw[:5000])
    print("=============================================\n")

    try:
        parsed = json.loads(raw)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Stage 2 returned invalid JSON: {exc}\n"
            f"Raw response:\n{raw[:3000]}"
        ) from exc

    if (
        not isinstance(parsed, dict)
        or not isinstance(parsed.get("rules"), list)
    ):
        raise ValueError(
            "Unexpected Stage 2 response structure. "
            f"Top-level keys: "
            f"{list(parsed.keys()) if isinstance(parsed, dict) else 'Not an object'}"
        )

    # --------------------------------------------------------
    # STEP 6: VALIDATE GENERATED JAVA DSL RULES
    # --------------------------------------------------------

    seen_ids: set[str] = set()
    reference_ids = set(existing_ids)
    validation_errors = []

    for index, rule in enumerate(parsed["rules"]):
        rule_errors = validate_rule(
            rule,
            catalog,
            reference_ids,
            seen_ids
        )

        validation_errors.extend([
            f'rules[{index}] ({rule.get("ruleId", "UNKNOWN")}): {error}'
            for error in rule_errors
        ])

    # --------------------------------------------------------
    # STEP 7: VALIDATE STAGE 2 RULE EVIDENCE
    # --------------------------------------------------------

    normalized_full_text = normalize_text(full_text)

    evidence_errors = []

    source_evidence = parsed.get(
        "sourceEvidence",
        []
    )

    if not isinstance(source_evidence, list):
        evidence_errors.append(
            "sourceEvidence must be an array."
        )
        source_evidence = []

    for item in source_evidence:
        if not isinstance(item, dict):
            evidence_errors.append(
                "Invalid sourceEvidence entry: expected an object."
            )
            continue

        quote = item.get("quote", "")
        page_number = item.get("page")

        if (
            not isinstance(quote, str)
            or not quote.strip()
            or normalize_text(quote) not in normalized_full_text
        ):
            evidence_errors.append(
                f'{item.get("ruleId", "UNKNOWN")}: evidence quote '
                "was not found in extracted PDF text."
            )

        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
            or page_number > len(pages)
        ):
            evidence_errors.append(
                f'{item.get("ruleId", "UNKNOWN")}: invalid evidence page.'
            )

    evidence_ids = {
        item.get("ruleId")
        for item in source_evidence
        if isinstance(item, dict)
    }

    for rule in parsed["rules"]:
        if not isinstance(rule, dict):
            continue

        if rule.get("ruleId") not in evidence_ids:
            evidence_errors.append(
                f'{rule.get("ruleId", "UNKNOWN")}: '
                "missing sourceEvidence entry."
            )

    # --------------------------------------------------------
    # STEP 8: PLACEHOLDER AND RESPONSE VALIDATION
    # --------------------------------------------------------

    placeholder_values = {
        "explain why the policy cannot be represented safely",
        "todo",
        "n/a",
        "not specified"
    }

    unmapped_items = parsed.get(
        "unmappedPolicyItems",
        []
    )

    if not isinstance(unmapped_items, list):
        validation_errors.append(
            "unmappedPolicyItems must be an array."
        )
        unmapped_items = [
            "Invalid unmappedPolicyItems response."
        ]

    placeholder_errors = [
        item
        for item in unmapped_items
        if (
            isinstance(item, str)
            and item.strip().casefold() in placeholder_values
        )
    ]

    if placeholder_errors:
        validation_errors.append(
            "Stage 2 returned placeholder text in unmappedPolicyItems."
        )

    assumptions = parsed.get("assumptions", [])
    review_notes = parsed.get("reviewNotes", [])

    if not isinstance(assumptions, list):
        validation_errors.append(
            "assumptions must be an array."
        )
        assumptions = []

    if not isinstance(review_notes, list):
        validation_errors.append(
            "reviewNotes must be an array."
        )
        review_notes = []

    # --------------------------------------------------------
    # STEP 9: DETERMINE TECHNICAL EXPORT STATUS
    # --------------------------------------------------------

    # This is a technical validation gate, not compliance approval.
    # An empty rules array can be valid if requirements were extracted
    # and Stage 2 explicitly explains why they are unsupported.
    export_allowed = (
        bool(extracted_requirements)
        and not unresolved_items
        and not extraction_errors
        and not validation_errors
        and not evidence_errors
        and not placeholder_errors
    )

    # --------------------------------------------------------
    # STEP 10: BUILD FINAL RESPONSE
    # --------------------------------------------------------

    generation = {
        "schemaVersion": "2.0",

        "sourceDocument": filename,

        "generator": {
            "provider": "ollama",
            "chatModel": CHAT_MODEL,
            "embeddingModel": EMBED_MODEL,
            "retrieval": retrieval_mode
        },

        "humanApprovalRequired": True,

        # Stage 1 results
        "extractedRequirements": extracted_requirements,
        "unresolvedPolicyItems": unresolved_items,
        "extractionErrors": extraction_errors,

        # Stage 2 results
        "rules": parsed["rules"],
        "sourceEvidence": source_evidence,
        "unmappedPolicyItems": unmapped_items,
        "assumptions": assumptions,
        "reviewNotes": review_notes,

        # Validation results
        "validationErrors": validation_errors,
        "evidenceErrors": evidence_errors,

        # Technical validation only; human approval is still required.
        "exportAllowed": export_allowed
    }

    return {
        "generation": generation,
        "fullText": full_text,
        "retrievedChunks": retrieved,
        "pageCount": len(pages)
    }
