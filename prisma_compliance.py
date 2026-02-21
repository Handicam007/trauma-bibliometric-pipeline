#!/usr/bin/env python3
"""
PRISMA-trAIce Compliance Checklist Generator
=============================================
Auto-generates a PRISMA-trAIce (Preferred Reporting Items for Systematic
reviews and Meta-Analyses — AI Transparency and Compliance Evaluation)
checklist from the pipeline configuration.

Extends standard PRISMA 2020 with AI-specific transparency items:
  - LLM provider, model, version, temperature
  - Prompt text (SHA-256 hash + excerpt)
  - Schema enforcement details
  - Validation framework description
  - Confidence thresholds and HITL mechanism
  - Cost transparency

Usage:
    python prisma_compliance.py
    python prisma_compliance.py --output my_checklist.json

Output:
    stats_output/prisma_traice_checklist.json (machine-readable)
    stats_output/prisma_traice_checklist.md   (formatted supplement)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent / "stats_output"
OUTPUT_JSON = OUTPUT_DIR / "prisma_traice_checklist.json"
OUTPUT_MD = OUTPUT_DIR / "prisma_traice_checklist.md"


def _sha256_excerpt(text: str, max_chars: int = 200) -> dict:
    """Return SHA-256 hash and truncated excerpt of a text block."""
    full_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    excerpt = text[:max_chars].replace("\n", " ").strip()
    if len(text) > max_chars:
        excerpt += "..."
    return {"sha256": full_hash, "excerpt": excerpt, "length_chars": len(text)}


def _safe_import(module_name: str):
    """Try to import a module, return None on failure."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def generate_prisma_traice_checklist(
    validation_report_path: Path = None,
    output_json: Path = None,
    output_md: Path = None,
) -> dict:
    """
    Generate PRISMA-trAIce checklist from pipeline configuration.

    Auto-populates fields from the codebase. Items requiring manual input
    are marked with status="MANUAL_REQUIRED".

    Returns:
        dict with complete checklist
    """
    if output_json is None:
        output_json = OUTPUT_JSON
    if output_md is None:
        output_md = OUTPUT_MD

    print(f"\n{'=' * 70}")
    print("PRISMA-trAIce COMPLIANCE CHECKLIST")
    print(f"{'=' * 70}")

    checklist = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "framework": "PRISMA-trAIce (AI Transparency and Compliance Evaluation)",
        "items": [],
    }

    # Import config
    config = _safe_import("config")

    # ── 1. TITLE ──────────────────────────────────────────────────────
    checklist["items"].append({
        "number": 1,
        "section": "Title",
        "item": "Identify the report as a systematic review, meta-analysis, or both",
        "reported": "MANUAL_REQUIRED",
        "evidence": "Indicate in title that AI/LLM was used for screening/classification",
    })

    # ── 2. REGISTRATION ──────────────────────────────────────────────
    checklist["items"].append({
        "number": 2,
        "section": "Registration",
        "item": "Registration number and registry name",
        "reported": "MANUAL_REQUIRED",
        "evidence": "PROSPERO or similar registration (if applicable)",
    })

    # ── 3. ELIGIBILITY CRITERIA ──────────────────────────────────────
    screening_module = _safe_import("llm_screening")
    if screening_module and hasattr(screening_module, "SYSTEM_PROMPT"):
        prompt_info = _sha256_excerpt(screening_module.SYSTEM_PROMPT)
        checklist["items"].append({
            "number": 3,
            "section": "Eligibility Criteria",
            "item": "Inclusion/exclusion criteria with AI screening prompt",
            "reported": "Yes",
            "evidence": {
                "prompt_hash": prompt_info["sha256"],
                "prompt_excerpt": prompt_info["excerpt"],
                "prompt_length": prompt_info["length_chars"],
                "confidence_thresholds": {
                    "auto_accept": getattr(config, "LLM_CONFIDENCE_AUTO", None),
                    "human_review": getattr(config, "LLM_CONFIDENCE_REVIEW", None),
                },
            },
        })
    else:
        # Try to get the prompt from a different attribute name
        checklist["items"].append({
            "number": 3,
            "section": "Eligibility Criteria",
            "item": "Inclusion/exclusion criteria with AI screening prompt",
            "reported": "Partial",
            "evidence": {
                "note": "Screening prompt defined in llm_screening.py",
                "confidence_thresholds": {
                    "auto_accept": getattr(config, "LLM_CONFIDENCE_AUTO", None),
                    "human_review": getattr(config, "LLM_CONFIDENCE_REVIEW", None),
                },
            },
        })

    # ── 4. INFORMATION SOURCES ────────────────────────────────────────
    checklist["items"].append({
        "number": 4,
        "section": "Information Sources",
        "item": "Databases and sources searched",
        "reported": "Yes",
        "evidence": {
            "primary_search": "Scopus Search API",
            "abstract_retrieval": ["Scopus Abstract Retrieval API", "PubMed E-utilities"],
            "geographic_filter": getattr(config, "GEO_FILTER_SCOPUS", None),
            "year_range": f"{getattr(config, 'YEAR_MIN', '?')}-{getattr(config, 'YEAR_MAX', '?')}",
            "field": getattr(config, "FIELD_NAME", "Unknown"),
        },
    })

    # ── 5. SEARCH STRATEGY ───────────────────────────────────────────
    search_meta_path = Path(__file__).parent / "results_refined" / "search_metadata.json"
    if search_meta_path.exists():
        try:
            search_meta = json.loads(search_meta_path.read_text())
            checklist["items"].append({
                "number": 5,
                "section": "Search Strategy",
                "item": "Full search strategy with reproducibility metadata",
                "reported": "Yes",
                "evidence": {
                    "search_date": search_meta.get("search_date_utc"),
                    "n_queries": search_meta.get("n_queries"),
                    "total_results": search_meta.get("total_results_raw"),
                    "truncation_warnings": search_meta.get("n_truncation_events", 0),
                    "metadata_file": str(search_meta_path),
                },
            })
        except Exception:
            checklist["items"].append({
                "number": 5,
                "section": "Search Strategy",
                "item": "Full search strategy",
                "reported": "Partial",
                "evidence": "search_metadata.json exists but could not be parsed",
            })
    else:
        checklist["items"].append({
            "number": 5,
            "section": "Search Strategy",
            "item": "Full search strategy",
            "reported": "MANUAL_REQUIRED",
            "evidence": "Record search queries and dates",
        })

    # ── 6. AI MODEL SPECIFICATION (AI-specific) ──────────────────────
    providers_module = _safe_import("llm_providers")
    model_info = {
        "provider": getattr(config, "LLM_PROVIDER", "unknown"),
        "model": getattr(config, "LLM_MODEL", None) or "(provider default)",
        "temperature": getattr(config, "LLM_TEMPERATURE", None),
        "max_retries": getattr(config, "LLM_MAX_RETRIES", None),
        "seed": 42,
    }
    if providers_module and hasattr(providers_module, "DEFAULT_MODELS"):
        model_info["default_models"] = providers_module.DEFAULT_MODELS

    checklist["items"].append({
        "number": 6,
        "section": "AI Model Specification",
        "item": "LLM provider, model, version, and hyperparameters",
        "reported": "Yes",
        "evidence": model_info,
        "ai_specific": True,
    })

    # ── 7. SCHEMA ENFORCEMENT (AI-specific) ──────────────────────────
    schemas_module = _safe_import("llm_schemas")
    schema_info = {}
    if schemas_module:
        import pydantic
        schema_info["pydantic_version"] = pydantic.__version__
        schema_info["schemas"] = []
        for name in ["ScreeningResult", "DomainClassification",
                      "ConceptClassification", "ExtractionResult"]:
            cls = getattr(schemas_module, name, None)
            if cls:
                schema_info["schemas"].append({
                    "name": name,
                    "fields": list(cls.model_fields.keys()),
                })

    checklist["items"].append({
        "number": 7,
        "section": "Schema Enforcement",
        "item": "Pydantic models for structured LLM output validation",
        "reported": "Yes" if schema_info else "No",
        "evidence": schema_info,
        "ai_specific": True,
    })

    # ── 8. CLASSIFICATION APPROACH (AI-specific) ─────────────────────
    classification_module = _safe_import("llm_classification")
    class_info = {}
    if classification_module and hasattr(classification_module, "DOMAIN_SYSTEM_PROMPT"):
        prompt_info = _sha256_excerpt(classification_module.DOMAIN_SYSTEM_PROMPT)
        class_info["domain_prompt_hash"] = prompt_info["sha256"]
        class_info["domain_prompt_excerpt"] = prompt_info["excerpt"]

    concepts_module = _safe_import("concept_definitions")
    if concepts_module and hasattr(concepts_module, "CLINICAL_CONCEPTS"):
        class_info["n_concepts"] = len(concepts_module.CLINICAL_CONCEPTS)
        class_info["concept_names"] = sorted(concepts_module.CLINICAL_CONCEPTS.keys())
    if concepts_module and hasattr(concepts_module, "DOMAIN_GROUPS"):
        class_info["n_domains"] = len(concepts_module.DOMAIN_GROUPS)
        class_info["domain_names"] = list(concepts_module.DOMAIN_GROUPS.keys())

    checklist["items"].append({
        "number": 8,
        "section": "Classification Approach",
        "item": "Hierarchical two-stage concept classification (domains → concepts)",
        "reported": "Yes" if class_info else "Partial",
        "evidence": class_info,
        "ai_specific": True,
    })

    # ── 9. DATA EXTRACTION PROMPT (AI-specific) ──────────────────────
    extraction_module = _safe_import("llm_extraction")
    extract_info = {}
    if extraction_module:
        for attr_name in ["GOLDEN_PROMPT", "SYSTEM_PROMPT", "EXTRACTION_SYSTEM_PROMPT"]:
            prompt_text = getattr(extraction_module, attr_name, None)
            if prompt_text:
                prompt_info = _sha256_excerpt(prompt_text)
                extract_info["prompt_hash"] = prompt_info["sha256"]
                extract_info["prompt_excerpt"] = prompt_info["excerpt"]
                extract_info["prompt_length"] = prompt_info["length_chars"]
                break

    checklist["items"].append({
        "number": 9,
        "section": "Data Extraction",
        "item": "Golden Prompt with anti-hallucination rules for structured extraction",
        "reported": "Yes" if extract_info else "Partial",
        "evidence": extract_info,
        "ai_specific": True,
    })

    # ── 10. VALIDATION FRAMEWORK (AI-specific) ───────────────────────
    validation_info = {
        "layers": [
            "Semantic Gap Audit (Wilson CI)",
            "LLM vs Regex Concept Agreement (Cohen's kappa per concept)",
            "Screening Agreement (LLM vs keyword filter with kappa)",
            "Human-in-the-Loop Dispute Export + Override Import",
            "Self-Consistency Check (temp=0)",
            "Three-Run Consensus Check (temperature variation)",
            "Validation Sample Generator (stratified for human annotation)",
            "Human-LLM Agreement (P/R/F1 against gold standard)",
        ],
        "n_layers": 8,
    }

    # Check if validation report exists
    val_report_path = validation_report_path or (
        Path(__file__).parent / "llm_cache" / "validation_report.json"
    )
    if val_report_path.exists():
        try:
            val_report = json.loads(val_report_path.read_text())
            validation_info["validation_report_exists"] = True
            validation_info["report_keys"] = list(val_report.keys())
        except Exception:
            pass

    checklist["items"].append({
        "number": 10,
        "section": "Validation Framework",
        "item": "Multi-layer validation with inter-rater reliability",
        "reported": "Yes",
        "evidence": validation_info,
        "ai_specific": True,
    })

    # ── 11. CONFIDENCE THRESHOLDS (AI-specific) ──────────────────────
    checklist["items"].append({
        "number": 11,
        "section": "Confidence Thresholds",
        "item": "Human-in-the-loop confidence thresholds for auto-accept/review/reject",
        "reported": "Yes",
        "evidence": {
            "auto_accept_threshold": getattr(config, "LLM_CONFIDENCE_AUTO", None),
            "human_review_threshold": getattr(config, "LLM_CONFIDENCE_REVIEW", None),
            "below_review": "auto-reject for screening, auto-skip for classification",
        },
        "ai_specific": True,
    })

    # ── 12. COST TRANSPARENCY (AI-specific) ──────────────────────────
    if providers_module and hasattr(providers_module, "COST_PER_1M_TOKENS"):
        cost_info = providers_module.COST_PER_1M_TOKENS
    else:
        cost_info = {}

    # Check for actual cost report
    cost_report_path = Path(__file__).parent / "llm_cache" / "cost_report.json"
    actual_costs = None
    if cost_report_path.exists():
        try:
            actual_costs = json.loads(cost_report_path.read_text())
        except Exception:
            pass

    checklist["items"].append({
        "number": 12,
        "section": "Cost Transparency",
        "item": "LLM API costs (estimated and actual)",
        "reported": "Yes" if actual_costs else "Partial",
        "evidence": {
            "pricing_table": {k: dict(v) for k, v in cost_info.items()} if cost_info else {},
            "actual_costs": actual_costs,
            "note": "Costs per 1M tokens (input, output) in USD",
        },
        "ai_specific": True,
    })

    # ── 13. REPRODUCIBILITY (AI-specific) ─────────────────────────────
    checklist["items"].append({
        "number": 13,
        "section": "Reproducibility",
        "item": "Reproducibility measures for LLM-based analysis",
        "reported": "Yes",
        "evidence": {
            "temperature": getattr(config, "LLM_TEMPERATURE", None),
            "seed": 42,
            "consensus_runs": getattr(config, "CONSENSUS_RUNS", None),
            "consensus_temperatures": list(getattr(config, "CONSENSUS_TEMPERATURES", ())),
            "caching": "Incremental JSON cache with atomic writes (crash-proof)",
            "schema_enforcement": "Pydantic models with auto-retry on validation failure",
            "code_repository": "MANUAL_REQUIRED — deposit code + prompts in supplementary",
        },
        "ai_specific": True,
    })

    # ── 14. EXCLUSION TAXONOMY (AI-specific) ──────────────────────────
    if schemas_module and hasattr(schemas_module, "ScreeningResult"):
        exclusion_field = schemas_module.ScreeningResult.model_fields.get("exclusion_category")
        if exclusion_field and hasattr(exclusion_field, "annotation"):
            # Extract Literal values
            import typing
            annotation = exclusion_field.annotation
            # Handle Optional[Literal[...]]
            args = getattr(annotation, "__args__", [])
            categories = []
            for arg in args:
                if hasattr(arg, "__args__"):
                    categories.extend(arg.__args__)
            if not categories:
                categories = ["See llm_schemas.py ScreeningResult.exclusion_category"]
        else:
            categories = ["See llm_schemas.py"]
    else:
        categories = []

    checklist["items"].append({
        "number": 14,
        "section": "Exclusion Taxonomy",
        "item": "Standardized exclusion categories for screening",
        "reported": "Yes" if categories else "Partial",
        "evidence": {"categories": categories},
        "ai_specific": True,
    })

    # ── 15-20: STANDARD PRISMA ITEMS (manual) ────────────────────────
    manual_items = [
        (15, "Study Selection", "Process of selecting studies (with AI role)"),
        (16, "Data Collection", "How data were extracted (automated + manual)"),
        (17, "Risk of Bias", "Risk of bias assessment in included studies"),
        (18, "Synthesis Methods", "Methods for combining results"),
        (19, "Funding", "Sources of funding and conflicts of interest"),
        (20, "Data Availability", "Code, data, prompts deposited in repository"),
    ]
    for num, section, item in manual_items:
        checklist["items"].append({
            "number": num,
            "section": section,
            "item": item,
            "reported": "MANUAL_REQUIRED",
            "evidence": "Complete in manuscript",
        })

    # ── Save JSON ────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(checklist, f, indent=2, default=str)

    # ── Generate Markdown ────────────────────────────────────────────
    md_lines = [
        "# PRISMA-trAIce Compliance Checklist",
        "",
        f"Generated: {checklist['generated_at']}",
        "",
        "| # | Section | Item | Reported | Evidence |",
        "|---|---------|------|----------|----------|",
    ]

    for item in checklist["items"]:
        reported = item["reported"]
        evidence = item.get("evidence", "")
        if isinstance(evidence, dict):
            # Summarize complex evidence
            evidence_str = "; ".join(
                f"{k}: {v}" for k, v in evidence.items()
                if not isinstance(v, (dict, list)) and v is not None
            )[:120]
        else:
            evidence_str = str(evidence)[:120]

        ai_tag = " 🤖" if item.get("ai_specific") else ""
        md_lines.append(
            f"| {item['number']} | {item['section']}{ai_tag} | "
            f"{item['item'][:60]} | {reported} | {evidence_str} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "Items marked 🤖 are AI-specific extensions to standard PRISMA 2020.",
        "",
        "Items marked `MANUAL_REQUIRED` must be completed by the authors.",
        "",
        f"Full machine-readable checklist: `{output_json.name}`",
    ])

    output_md.write_text("\n".join(md_lines))

    # ── Summary ──────────────────────────────────────────────────────
    n_auto = sum(1 for i in checklist["items"] if i["reported"] == "Yes")
    n_partial = sum(1 for i in checklist["items"] if i["reported"] == "Partial")
    n_manual = sum(1 for i in checklist["items"] if i["reported"] == "MANUAL_REQUIRED")
    n_ai = sum(1 for i in checklist["items"] if i.get("ai_specific"))

    print(f"\n  Checklist items: {len(checklist['items'])}")
    print(f"    Auto-populated:    {n_auto}")
    print(f"    Partially filled:  {n_partial}")
    print(f"    Manual required:   {n_manual}")
    print(f"    AI-specific items: {n_ai}")
    print(f"\n  Saved JSON: {output_json}")
    print(f"  Saved Markdown: {output_md}")

    return checklist


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate PRISMA-trAIce checklist")
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=OUTPUT_MD)
    parser.add_argument("--validation-report", type=Path, default=None)
    args = parser.parse_args()

    generate_prisma_traice_checklist(
        validation_report_path=args.validation_report,
        output_json=args.output_json,
        output_md=args.output_md,
    )


if __name__ == "__main__":
    main()
