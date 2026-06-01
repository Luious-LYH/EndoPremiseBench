#!/usr/bin/env python3
"""Profile candidate EndoPremiseBench v2 probes.

This script is a data reality check. It reads local annotation files, builds
auditable candidate probes when rules are deterministic, and reports rejected
or diagnostic cases separately. It does not run any model inference.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


NA_TEXT = "not applicable / no such entity is visible"
DEFAULT_CONFIG = Path("configs/premise_probe_v2.json")
DEFAULT_OUTPUT_JSON = Path("results/data_reality_check_v2.json")
DEFAULT_CANDIDATES_JSONL = Path("results/premise_candidates_v2.jsonl")
DEFAULT_REPORT_MD = Path("tables/data_reality_check_v2.md")

LETTERS = ["A", "B", "C", "D"]

ONTOLOGY = {
    "color": ["pink", "red", "white"],
    "location": ["upper region", "central region", "lower region"],
    "morphology_type": ["polypoid lesion", "flat lesion", "pedunculated lesion"],
    "count": ["one", "two", "three or more"],
    "removal_status": ["completely removed", "residual tissue remains", "not removed"],
    "presence": ["yes", "no", "unclear"],
}

ABSENCE_RE = re.compile(
    r"\b(no|none|not relevant|not applicable|absent|without|no evidence|not identified|"
    r"not observed|not detected|not visible|0)\b",
    re.IGNORECASE,
)


def read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    for key in ("data", "annotations", "samples", "records"):
        value = data.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    raise ValueError(f"Unsupported JSON shape: {path}")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def is_absence(answer: Any) -> bool:
    text = norm_text(answer).lower()
    return bool(ABSENCE_RE.search(text))


def has_image(project_root: Path, dataset_dir: str, image_path: str) -> bool:
    if not image_path:
        return False
    return (project_root / dataset_dir / image_path).exists()


def map_question_class(question_class: str) -> Tuple[Optional[str], Optional[str]]:
    """Map one atomic Kvasir-VQA-x1 class to one target/attribute pair."""
    cls = question_class.lower()
    if cls.startswith("polyp_"):
        target = "polyp"
    elif cls.startswith("instrument_"):
        target = "instrument"
    elif cls.startswith("abnormality_") or cls.startswith("finding_"):
        target = "abnormality"
    elif cls.startswith("landmark_"):
        target = "landmark"
    else:
        return None, None

    if cls.endswith("_color"):
        return target, "color"
    if cls.endswith("_location"):
        return target, "location"
    if cls.endswith("_type"):
        return target, "morphology_type"
    if cls.endswith("_count"):
        return target, "count"
    if cls.endswith("_removal_status"):
        return target, "removal_status"
    if cls.endswith("_presence"):
        return target, "presence"
    if cls.endswith("_size"):
        return target, "size"
    return None, None


def target_absent(answer: str, target: str) -> bool:
    text = norm_text(answer).lower()
    def negated_near(terms: Sequence[str]) -> bool:
        for term in terms:
            if re.search(rf"\b(no|without)\b[^.;,]{{0,60}}\b{term}\b", text):
                return True
            if re.search(rf"\b{term}\b[^.;,]{{0,50}}\b(not|not\s+visible|not\s+identified|not\s+observed|not\s+detected|absent)\b", text):
                return True
        return False

    if target == "polyp":
        return negated_near(["polyp", "polyps", "polypoid lesion", "polypoid lesions"]) or "no evidence of polyp" in text
    if target == "instrument":
        return negated_near(["instrument", "instruments", "device", "devices", "tool", "tools"])
    if target == "abnormality":
        return negated_near(["abnormality", "abnormalities", "finding", "findings", "lesion", "lesions"]) or bool(re.search(r"\bnormal\b", text))
    if target == "landmark":
        return negated_near(["landmark", "landmarks"])
    return is_absence(text)


def target_present(answer: str, target: str) -> bool:
    text = norm_text(answer).lower()
    if target_absent(text, target):
        return False
    if target == "polyp":
        return any(x in text for x in ["polyp", "polypoid", "paris"])
    if target == "instrument":
        return any(x in text for x in ["instrument", "device", "tool", "tube"])
    if target == "abnormality":
        return any(x in text for x in ["abnormal", "finding", "lesion", "esophagitis", "colitis", "ulcer"])
    if target == "landmark":
        return any(x in text for x in ["landmark", "z-line", "ileocecal", "pylorus"])
    return not is_absence(text)


def infer_from_question(question: str, source: str = "") -> Tuple[Optional[str], Optional[str], str]:
    q = question.lower()
    s = source.lower()
    if "polyp" in q or "polyp" in s:
        target = "polyp"
    elif "instrument" in q or "instrument" in s or "tool" in q:
        target = "instrument"
    elif "abnormal" in q or "finding" in q or "lesion" in q:
        target = "abnormality"
    elif "landmark" in q or "z-line" in q:
        target = "landmark"
    else:
        target = None

    if "color" in q:
        return target, "color", "question_text"
    if "where" in q or "location" in q or "located" in q:
        return target, "location", "question_text"
    if "type" in q or "morphology" in q or "classification" in q:
        return target, "morphology_type", "question_text"
    if "how many" in q or "count" in q or "number" in q:
        return target, "count", "question_text"
    if "removed" in q or "removal" in q:
        return target, "removal_status", "question_text"
    if "present" in q or "visible" in q or "are there" in q or "is there" in q:
        return target, "presence", "question_text"
    if "size" in q:
        return target, "size", "question_text"
    return target, None, "question_text"


def normalize_answer_option(answer: str, attribute_type: str) -> Optional[str]:
    text = norm_text(answer).lower()
    if not text:
        return None
    if attribute_type == "count":
        if text in {"1", "one", "one abnormal finding present", "one surgical instrument present"}:
            return "one"
        if text in {"2", "two"}:
            return "two"
        if text in {"3", "4", "5", "three", "three or more", ">2"}:
            return "three or more"
    if attribute_type == "presence":
        if text in {"yes", "present", "visible"} or text.startswith("evidence of"):
            return "yes"
        if is_absence(text):
            return "no"
    if attribute_type == "color":
        for color in ONTOLOGY["color"]:
            if text == color:
                return color
    if attribute_type == "location":
        if any(x in text for x in ["upper", "top"]):
            return "upper region"
        if any(x in text for x in ["lower", "bottom"]):
            return "lower region"
        if any(x in text for x in ["center", "central", "middle"]):
            return "central region"
    if attribute_type == "morphology_type":
        if "pedunculated" in text or "paris ip" in text:
            return "pedunculated lesion"
        if "flat" in text or "paris iia" in text:
            return "flat lesion"
        if "polyp" in text or "paris is" in text:
            return "polypoid lesion"
    if attribute_type == "removal_status":
        if "no residual" in text or "removed" in text and "not" not in text:
            return "completely removed"
        if "residual" in text or "remain present" in text or "remains present" in text:
            return "residual tissue remains"
        if "not removed" in text:
            return "not removed"
    return None


def make_options(correct_text: str, attribute_type: str, rng: random.Random) -> Tuple[Dict[str, str], str]:
    pool = [x for x in ONTOLOGY.get(attribute_type, []) if x != correct_text]
    distractors = pool[:]
    rng.shuffle(distractors)
    values = [correct_text] + distractors[:2] + [NA_TEXT]
    values = values[:4]
    while len(values) < 4:
        values.append(f"other option {len(values)}")
    rng.shuffle(values)
    options = dict(zip(LETTERS, values))
    answer = next(k for k, v in options.items() if v == correct_text)
    return options, answer


def make_false_premise_options(attribute_type: str, rng: random.Random) -> Tuple[Dict[str, str], str]:
    concrete = ONTOLOGY.get(attribute_type, ["yes", "no", "unclear"])[:3]
    values = concrete + [NA_TEXT]
    rng.shuffle(values)
    options = dict(zip(LETTERS, values))
    answer = next(k for k, v in options.items() if v == NA_TEXT)
    return options, answer


def display_entity(target: str, plural: bool = False) -> str:
    if target == "abnormality":
        return "abnormal findings" if plural else "abnormal finding"
    if target == "landmark":
        return "anatomical landmarks" if plural else "anatomical landmark"
    if target == "instrument":
        return "instruments" if plural else "instrument"
    if target == "polyp":
        return "polyps" if plural else "polyp"
    return f"{target}s" if plural else target


def probe_question(target: str, attribute_type: str) -> str:
    entity = display_entity(target)
    if attribute_type == "color":
        return f"What color is the {entity} in the image?"
    if attribute_type == "location":
        return f"Where is the {entity} located in the image?"
    if attribute_type == "morphology_type":
        return f"What type or morphology is the {entity} in the image?"
    if attribute_type == "count":
        return f"How many {display_entity(target, plural=True)} are visible in the image?"
    if attribute_type == "removal_status":
        return f"What is the removal status of the {entity} in the image?"
    if attribute_type == "presence":
        return f"Is the {entity} visible in the image?"
    return f"What attribute best describes the {entity} in the image?"


def candidate_record(
    *,
    cid: str,
    image_path: str,
    question: str,
    options: Dict[str, str],
    answer: str,
    premise_type: str,
    target_entity: str,
    attribute_type: str,
    source_dataset: str,
    label_evidence: str,
    scoring_type: str,
    confidence: str,
    construction_rule: str,
    source_id: Any,
) -> Dict[str, Any]:
    return {
        "id": cid,
        "image_path": image_path,
        "question": question,
        "options": options,
        "answer": answer,
        "premise_type": premise_type,
        "target_entity": target_entity,
        "attribute_type": attribute_type,
        "source_dataset": source_dataset,
        "label_evidence": label_evidence,
        "scoring_type": scoring_type,
        "confidence": confidence,
        "construction_rule": construction_rule,
        "source_id": source_id,
    }


def add_rejection(rejected: List[Dict[str, Any]], source_dataset: str, source_id: Any, reason: str, record: Dict[str, Any]) -> None:
    if len(rejected) < 5000:
        rejected.append(
            {
                "source_dataset": source_dataset,
                "source_id": source_id,
                "reason": reason,
                "question": norm_text(record.get("question", ""))[:300],
                "answer": norm_text(record.get("answer", record.get("gt", "")))[:300],
            }
        )


def profile_kvasir_x1(project_root: Path, data_dir: str, records: Iterable[Dict[str, Any]], rng: random.Random) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for rec in records:
        source_id = rec.get("id")
        image_path = str(Path(data_dir) / rec.get("image_path", ""))
        if not has_image(project_root, data_dir, rec.get("image_path", "")):
            add_rejection(rejected, "Kvasir-VQA-x1", source_id, "missing_image", rec)
            continue
        classes = rec.get("question_class") or []
        mapped_classes = []
        for cls in classes:
            target, attribute_type = map_question_class(str(cls))
            if target and attribute_type:
                mapped_classes.append((str(cls), target, attribute_type))
        if not mapped_classes:
            add_rejection(rejected, "Kvasir-VQA-x1", source_id, "unsupported_or_unresolved_question_class", rec)
            continue
        answer_text = norm_text(rec.get("answer"))
        made_any = False
        for cls, target, attribute_type in mapped_classes:
            evidence = f"question_class={cls}; answer={answer_text}"
            if attribute_type == "size":
                add_rejection(rejected, "Kvasir-VQA-x1", source_id, "size_low_confidence_not_main_table", rec)
                continue
            if target_absent(answer_text, target):
                if attribute_type == "presence":
                    add_rejection(rejected, "Kvasir-VQA-x1", source_id, "presence_absence_is_object_probe_not_attribute_trap", rec)
                    continue
                options, answer = make_false_premise_options(attribute_type, rng)
                candidates.append(
                    candidate_record(
                        cid=f"kx1_fp_{source_id}_{cls}",
                        image_path=image_path,
                        question=probe_question(target, attribute_type),
                        options=options,
                        answer=answer,
                        premise_type="false",
                        target_entity=target,
                        attribute_type=attribute_type,
                        source_dataset="Kvasir-VQA-x1",
                        label_evidence=evidence,
                        scoring_type="converted_mcq",
                        confidence="high",
                        construction_rule="false-premise trap from x1 atomic class plus target-specific absence answer",
                        source_id=source_id,
                    )
                )
                made_any = True
                continue
            if not target_present(answer_text, target):
                add_rejection(rejected, "Kvasir-VQA-x1", source_id, "no_target_specific_positive_or_absence_evidence", rec)
                continue
            mapped = normalize_answer_option(answer_text, attribute_type)
            if mapped is None:
                add_rejection(rejected, "Kvasir-VQA-x1", source_id, "answer_not_in_controlled_ontology", rec)
                continue
            options, answer = make_options(mapped, attribute_type, rng)
            candidates.append(
                candidate_record(
                    cid=f"kx1_tp_{source_id}_{cls}",
                    image_path=image_path,
                    question=probe_question(target, attribute_type),
                    options=options,
                    answer=answer,
                    premise_type="true",
                    target_entity=target,
                    attribute_type=attribute_type,
                    source_dataset="Kvasir-VQA-x1",
                    label_evidence=evidence,
                    scoring_type="converted_mcq",
                    confidence="medium",
                    construction_rule="true-premise MCQ conversion from x1 atomic class and controlled answer",
                    source_id=source_id,
                )
            )
            made_any = True
        if not made_any:
            if not any(x["source_dataset"] == "Kvasir-VQA-x1" and x["source_id"] == source_id for x in rejected[-len(mapped_classes or [None]) - 1 :]):
                add_rejection(rejected, "Kvasir-VQA-x1", source_id, "presence_absence_is_object_probe_not_attribute_trap", rec)
    return candidates, rejected


def profile_kvasir_vqa(project_root: Path, data_dir: str, records: Iterable[Dict[str, Any]], rng: random.Random) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    normal_seen = Counter()
    for rec in records:
        source_id = rec.get("id")
        source = norm_text(rec.get("source"))
        rel_image = rec.get("image_path", "")
        image_path = str(Path(data_dir) / rel_image)
        if not has_image(project_root, data_dir, rel_image):
            add_rejection(rejected, "Kvasir-VQA", source_id, "missing_image", rec)
            continue
        target, attribute_type, evidence_source = infer_from_question(norm_text(rec.get("question")), source)
        answer_text = norm_text(rec.get("answer"))
        if source.lower() == "normal":
            # Normal images support high-confidence absence traps. Cap repeated
            # traps per image-target-attribute to keep the profile realistic.
            for target_attr in [("polyp", "color"), ("polyp", "morphology_type"), ("instrument", "location"), ("abnormality", "color")]:
                key = (rec.get("img_id"),) + target_attr
                if normal_seen[key]:
                    continue
                normal_seen[key] += 1
                target2, attr2 = target_attr
                options, answer = make_false_premise_options(attr2, rng)
                candidates.append(
                    candidate_record(
                        cid=f"kvqa_normal_fp_{source_id}_{target2}_{attr2}",
                        image_path=image_path,
                        question=probe_question(target2, attr2),
                        options=options,
                        answer=answer,
                        premise_type="false",
                        target_entity=target2,
                        attribute_type=attr2,
                        source_dataset="Kvasir-VQA",
                        label_evidence=f"source=Normal; original_answer={answer_text}",
                        scoring_type="converted_mcq",
                        confidence="high",
                        construction_rule="false-premise trap from Kvasir-VQA Normal source",
                        source_id=source_id,
                    )
                )
            continue
        if target is None or attribute_type is None:
            add_rejection(rejected, "Kvasir-VQA", source_id, "unsupported_or_unresolved_question_text", rec)
            continue
        if attribute_type == "size":
            add_rejection(rejected, "Kvasir-VQA", source_id, "size_low_confidence_not_main_table", rec)
            continue
        if is_absence(answer_text):
            add_rejection(rejected, "Kvasir-VQA", source_id, "absence_answer_without_normal_source_kept_out_for_conservatism", rec)
            continue
        mapped = normalize_answer_option(answer_text, attribute_type)
        if mapped is None:
            add_rejection(rejected, "Kvasir-VQA", source_id, "answer_not_in_controlled_ontology", rec)
            continue
        options, answer = make_options(mapped, attribute_type, rng)
        candidates.append(
            candidate_record(
                cid=f"kvqa_tp_{source_id}",
                image_path=image_path,
                question=probe_question(target, attribute_type),
                options=options,
                answer=answer,
                premise_type="true",
                target_entity=target,
                attribute_type=attribute_type,
                source_dataset="Kvasir-VQA",
                label_evidence=f"{evidence_source}; source={source}; answer={answer_text}",
                scoring_type="converted_mcq",
                confidence="medium",
                construction_rule="true-premise MCQ conversion from Kvasir-VQA answer",
                source_id=source_id,
            )
        )
    return candidates, rejected


def infer_endobench_attribute(rec: Dict[str, Any]) -> str:
    text = " ".join(norm_text(rec.get(k)) for k in ("category", "task", "subtask", "question")).lower()
    if "count" in text or "quantification" in text:
        return "count"
    if "type" in text or "classification" in text:
        return "morphology_type"
    if "region" in text or "grounding" in text or "localization" in text or "landmark" in text:
        return "location"
    if "instrument" in text:
        return "presence"
    return "presence"


def profile_endobench(project_root: Path, data_dir: str, records: Iterable[Dict[str, Any]], rng: random.Random) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for rec in records:
        source_id = rec.get("id")
        rel_image = rec.get("image_path", "")
        image_path = str(Path(data_dir) / "EndoBench-Images" / rel_image)
        if not (project_root / image_path).exists():
            add_rejection(rejected, "EndoBench", source_id, "missing_image", rec)
            continue
        options_list = rec.get("options") or []
        answer_letter = norm_text(rec.get("answer"))
        if answer_letter not in LETTERS or len(options_list) < 3:
            add_rejection(rejected, "EndoBench", source_id, "unsupported_answer_or_too_few_options", rec)
            continue
        correct_idx = LETTERS.index(answer_letter)
        if correct_idx >= len(options_list):
            add_rejection(rejected, "EndoBench", source_id, "answer_index_out_of_range", rec)
            continue
        correct_text = norm_text(options_list[correct_idx])
        distractors = [norm_text(x) for i, x in enumerate(options_list) if i != correct_idx and norm_text(x)]
        rng.shuffle(distractors)
        values = [correct_text] + distractors[:2] + [NA_TEXT]
        rng.shuffle(values)
        options = dict(zip(LETTERS, values))
        answer = next(k for k, v in options.items() if v == correct_text)
        attribute_type = infer_endobench_attribute(rec)
        target = "clinical finding"
        if "polyp" in " ".join([norm_text(rec.get("question")), norm_text(rec.get("subtask"))]).lower():
            target = "polyp"
        elif "instrument" in " ".join([norm_text(rec.get("question")), norm_text(rec.get("subtask"))]).lower():
            target = "instrument"
        elif "landmark" in " ".join([norm_text(rec.get("question")), norm_text(rec.get("subtask"))]).lower():
            target = "landmark"
        candidates.append(
            candidate_record(
                cid=f"endobench_tp_{source_id}",
                image_path=image_path,
                question=norm_text(rec.get("question")),
                options=options,
                answer=answer,
                premise_type="true",
                target_entity=target,
                attribute_type=attribute_type,
                source_dataset="EndoBench",
                label_evidence=f"EndoBench reviewed MCQ; original_answer={answer_letter}; gt={norm_text(rec.get('gt'))}",
                scoring_type="converted_mcq",
                confidence="high",
                construction_rule="true-premise control from EndoBench MCQ with added randomized N/A option",
                source_id=source_id,
            )
        )
    return candidates, rejected


def profile_endobench_extended(records: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    diagnostics: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for rec in records:
        diagnostics.append(
            {
                "id": f"endobench_ext_diag_{rec.get('id')}",
                "image_path": str(Path("data/EndoBench-Extended/Extended-Images") / norm_text(rec.get("image"))),
                "question": norm_text(rec.get("question")),
                "answer": norm_text(rec.get("answer")),
                "premise_type": "unknown",
                "target_entity": "clinical finding",
                "attribute_type": "open_diagnostic",
                "source_dataset": "EndoBench-Extended",
                "label_evidence": "open-ended expert QA; no deterministic false-premise label",
                "scoring_type": "open_ended",
                "confidence": "low",
                "construction_rule": "diagnostic-only open-ended sample",
            }
        )
        add_rejection(rejected, "EndoBench-Extended", rec.get("id"), "open_ended_diagnostic_only", rec)
    return diagnostics, rejected


def counter_dict(items: Iterable[Any]) -> Dict[str, int]:
    return {str(k): v for k, v in Counter(items).most_common()}


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_markdown(summary: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# EndoPremiseBench Data Reality Check v2")
    lines.append("")
    lines.append("更新时间：`2026-05-22 Asia/Shanghai`")
    lines.append("")
    lines.append("## Node A Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key in [
        "total_candidate_probes",
        "strict_mcq_count",
        "converted_mcq_count",
        "open_ended_diagnostic_count",
        "true_premise_count",
        "false_premise_count",
        "rejected_count",
    ]:
        lines.append(f"| `{key}` | {summary[key]} |")
    lines.append("")
    lines.append("## Attribute Distribution")
    lines.append("")
    lines.append("| Attribute | Count |")
    lines.append("|---|---:|")
    for k, v in summary["attribute_type_distribution"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Source Distribution")
    lines.append("")
    lines.append("| Source | Count |")
    lines.append("|---|---:|")
    for k, v in summary["source_dataset_distribution"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## N/A Option Position")
    lines.append("")
    lines.append("| Option | Count |")
    lines.append("|---|---:|")
    for k, v in summary["na_position_distribution"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Gate Assessment")
    lines.append("")
    for item in summary["gate_assessment"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Accepted Examples")
    lines.append("")
    for ex in summary["accepted_examples"]:
        lines.append(f"- `{ex['id']}` [{ex['premise_type']}/{ex['attribute_type']}/{ex['source_dataset']}]: {ex['question']} -> {ex['answer']}")
    lines.append("")
    lines.append("## Rejected Examples")
    lines.append("")
    for ex in summary["rejected_examples"]:
        lines.append(f"- `{ex['source_dataset']}:{ex['source_id']}` {ex['reason']}: {ex['question']}")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(summary["decision"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def gate_assessment(summary: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    main_count = summary["strict_mcq_count"] + summary["converted_mcq_count"]
    if main_count < 500:
        notes.append("FAIL: strict + converted MCQ < 500; must shrink to a small diagnostic probe or convert more samples.")
    else:
        notes.append("PASS: strict + converted MCQ >= 500.")
    true_count = summary["true_premise_count"]
    false_count = summary["false_premise_count"]
    if min(true_count, false_count) == 0 or max(true_count, false_count) / max(1, min(true_count, false_count)) > 3:
        notes.append("WARN: true/false premise imbalance exceeds 3:1; use balanced subset for main experiments.")
    else:
        notes.append("PASS: true/false premise imbalance is within 3:1.")
    attr = summary["attribute_type_distribution"]
    if attr:
        top_attr, top_attr_n = next(iter(attr.items()))
        if top_attr_n / max(1, main_count) > 0.60:
            notes.append(f"WARN: attribute `{top_attr}` exceeds 60%; rebalance attribute types.")
        else:
            notes.append("PASS: no attribute type exceeds 60%.")
    src = summary["source_dataset_distribution"]
    if src:
        top_src, top_src_n = next(iter(src.items()))
        if top_src_n / max(1, main_count) > 0.70:
            notes.append(f"WARN: source `{top_src}` exceeds 70%; use source-balanced subset.")
        else:
            notes.append("PASS: no dataset source exceeds 70%.")
    na_pos = summary["na_position_distribution"]
    if len([k for k, v in na_pos.items() if v]) <= 1:
        notes.append("FAIL: N/A appears in a fixed option position; randomize before inference.")
    else:
        notes.append("PASS: N/A positions are randomized across options.")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-candidates", type=Path, default=DEFAULT_CANDIDATES_JSONL)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--max-records-per-dataset", type=int, default=0, help="0 means all records")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = Path(cfg["project_root"])
    rng = random.Random(int(cfg.get("seed", 20260522)))
    paths = cfg["data_paths"]

    all_candidates: List[Dict[str, Any]] = []
    all_rejected: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    dataset_counts: Dict[str, int] = {}

    def limited(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if args.max_records_per_dataset and args.max_records_per_dataset > 0:
            return records[: args.max_records_per_dataset]
        return records

    kx1_train = limited(read_json(project_root / paths["kvasir_vqa_x1_train"]))
    kx1_test = limited(read_json(project_root / paths["kvasir_vqa_x1_test"]))
    kx1_records = kx1_train + kx1_test
    dataset_counts["Kvasir-VQA-x1"] = len(kx1_records)
    cands, rej = profile_kvasir_x1(project_root, "data/Kvasir-VQA-x1", kx1_records, rng)
    all_candidates.extend(cands)
    all_rejected.extend(rej)

    kvqa_records = limited(read_json(project_root / paths["kvasir_vqa"]))
    dataset_counts["Kvasir-VQA"] = len(kvqa_records)
    cands, rej = profile_kvasir_vqa(project_root, "data/Kvasir-VQA", kvqa_records, rng)
    all_candidates.extend(cands)
    all_rejected.extend(rej)

    endobench_records = limited(read_json(project_root / paths["endobench"]))
    dataset_counts["EndoBench"] = len(endobench_records)
    cands, rej = profile_endobench(project_root, "data/EndoBench", endobench_records, rng)
    all_candidates.extend(cands)
    all_rejected.extend(rej)

    ext_records = limited(read_json(project_root / paths["endobench_extended"]))
    dataset_counts["EndoBench-Extended"] = len(ext_records)
    diag, rej = profile_endobench_extended(ext_records)
    diagnostics.extend(diag)
    all_rejected.extend(rej)

    write_jsonl(all_candidates, args.output_candidates)

    na_positions = []
    for rec in all_candidates:
        for letter, text in rec["options"].items():
            if text == NA_TEXT:
                na_positions.append(letter)
                break

    strict_count = sum(1 for x in all_candidates if x["scoring_type"] == "strict_mcq")
    converted_count = sum(1 for x in all_candidates if x["scoring_type"] == "converted_mcq")
    summary: Dict[str, Any] = {
        "version": "data_reality_check_v2",
        "updated_at": "2026-05-22 Asia/Shanghai",
        "dataset_record_counts": dataset_counts,
        "total_candidate_probes": len(all_candidates) + len(diagnostics),
        "strict_mcq_count": strict_count,
        "converted_mcq_count": converted_count,
        "open_ended_diagnostic_count": len(diagnostics),
        "true_premise_count": sum(1 for x in all_candidates if x["premise_type"] == "true"),
        "false_premise_count": sum(1 for x in all_candidates if x["premise_type"] == "false"),
        "attribute_type_distribution": counter_dict(x["attribute_type"] for x in all_candidates),
        "source_dataset_distribution": counter_dict(x["source_dataset"] for x in all_candidates),
        "na_position_distribution": counter_dict(na_positions),
        "rejected_count": len(all_rejected),
        "rejected_reason_distribution": counter_dict(x["reason"] for x in all_rejected),
        "accepted_examples": all_candidates[:10],
        "rejected_examples": all_rejected[:10],
        "diagnostic_examples": diagnostics[:10],
        "candidate_jsonl": str(args.output_candidates),
    }
    summary["gate_assessment"] = gate_assessment(summary)
    warnings = [x for x in summary["gate_assessment"] if x.startswith(("WARN", "FAIL"))]
    if warnings:
        summary["decision"] = (
            "Do not launch full GPU inference yet. Build a balanced high-confidence subset, "
            "then run parser/model smoke. Warnings: " + " ".join(warnings)
        )
    else:
        summary["decision"] = "Data gate passes for parser/scorer smoke on a balanced subset."

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, args.output_md)
    print(json.dumps({"summary": str(args.output_json), "candidates": str(args.output_candidates), "report": str(args.output_md), "candidate_count": len(all_candidates)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
