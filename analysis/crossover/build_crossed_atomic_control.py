#!/usr/bin/env python3
"""Build the test-annotation-only 2x2 crossed premise control."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import networkx as nx
import pyarrow.parquet as pq


SEED = 20260716
LETTERS = ("A", "B", "C", "D")
NA_TEXT = "not applicable / no such entity is visible"
COUNT_OPTIONS = ("one", "two", "three or more", NA_TEXT)
LOCATION_OPTIONS = ("upper region", "central region", "lower region", NA_TEXT)
TOKEN = re.compile(r"^(upper|center|lower)(?:$|[-\s])")

QUESTIONS = {
    ("instrument", "count"): "How many instruments are visible in the image?",
    ("polyp", "count"): "How many polyps are visible in the image?",
    ("instrument", "location"): "Where is the instrument located in the image?",
    ("landmark", "location"): "Where is the anatomical landmark located in the image?",
}


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def parse_integer(answer: str) -> int | None:
    text = normalize(answer)
    return int(text) if re.fullmatch(r"\d+", text) else None


def parse_count_presence(answer: str) -> bool | None:
    value = parse_integer(answer)
    return None if value is None else value > 0


def parse_not_none_presence(answer: str) -> bool | None:
    text = normalize(answer)
    return None if not text else text != "none"


def parse_polyp_removal_presence(answer: str) -> bool | None:
    text = normalize(answer)
    return None if not text else text != "not relevant"


ENTITY_SIGNALS: Dict[str, Sequence[Tuple[str, Callable[[str], bool | None]]]] = {
    "instrument": (
        ("instrument_count", parse_count_presence),
        ("instrument_presence", parse_not_none_presence),
        ("instrument_location", parse_not_none_presence),
    ),
    "polyp": (
        ("polyp_count", parse_count_presence),
        ("polyp_type", parse_not_none_presence),
        ("polyp_removal_status", parse_polyp_removal_presence),
    ),
    "landmark": (
        ("landmark_presence", parse_not_none_presence),
        ("landmark_location", parse_not_none_presence),
        ("landmark_color", lambda answer: True if answer else None),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*values: Any) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def load_atomic_annotations(
    parquet_path: Path,
) -> Tuple[Dict[Tuple[str, str], Set[str]], Dict[Tuple[str, str, str], Set[int]], Set[str], int, int]:
    rows = pq.read_table(
        parquet_path,
        columns=["img_id", "original", "question_class"],
    ).to_pylist()
    values: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    source_rows: Dict[Tuple[str, str, str], Set[int]] = defaultdict(set)
    image_ids: Set[str] = set()
    atomic_count = 0
    for row_index, row in enumerate(rows):
        image_id = str(row["img_id"])
        image_ids.add(image_id)
        qa = json.loads(str(row["original"]))
        if isinstance(qa, dict):
            qa = [qa]
        classes = list(row.get("question_class") or [])
        if len(qa) != len(classes):
            raise ValueError(f"row {row_index}: original/question_class length mismatch")
        for question_class, atom in zip(classes, qa):
            if not isinstance(atom, dict) or "a" not in atom:
                raise ValueError(f"row {row_index}: malformed atomic QA")
            cls = str(question_class)
            answer = normalize(atom["a"])
            values[(image_id, cls)].add(answer)
            source_rows[(image_id, cls, answer)].add(row_index)
            atomic_count += 1
    return values, source_rows, image_ids, len(rows), atomic_count


def entity_status(values: Mapping[Tuple[str, str], Set[str]], image_id: str, entity: str) -> str:
    signals: Set[bool] = set()
    for question_class, parser in ENTITY_SIGNALS[entity]:
        for answer in values.get((image_id, question_class), set()):
            signal = parser(answer)
            if signal is not None:
                signals.add(signal)
    if not signals:
        return "unknown"
    if len(signals) > 1:
        return "conflict"
    return "present" if True in signals else "absent"


def count_label(values: Mapping[Tuple[str, str], Set[str]], image_id: str, entity: str) -> str | None:
    question_class = {"instrument": "instrument_count", "polyp": "polyp_count"}[entity]
    labels: Set[str] = set()
    for answer in values.get((image_id, question_class), set()):
        value = parse_integer(answer)
        if value is None:
            return None
        labels.add("absent" if value == 0 else "one" if value == 1 else "two" if value == 2 else "three or more")
    if len(labels) != 1:
        return None
    label = next(iter(labels))
    return None if label == "absent" else label


def location_label(values: Mapping[Tuple[str, str], Set[str]], image_id: str, entity: str) -> str | None:
    question_class = {"instrument": "instrument_location", "landmark": "landmark_location"}[entity]
    answers = values.get((image_id, question_class), set())
    if not answers or "none" in answers:
        return None
    all_bands: Set[str] = set()
    for answer in answers:
        bands: Set[str] = set()
        for token in re.split(r"\s*;\s*", answer):
            match = TOKEN.match(token)
            if match is None:
                return None
            bands.add(match.group(1))
        if len(bands) != 1:
            return None
        all_bands.update(bands)
    if len(all_bands) != 1:
        return None
    return {"upper": "upper region", "center": "central region", "lower": "lower region"}[next(iter(all_bands))]


def atomic_evidence(
    values: Mapping[Tuple[str, str], Set[str]],
    source_rows: Mapping[Tuple[str, str, str], Set[int]],
    image_id: str,
    entity: str,
    attribute: str | None = None,
) -> List[Dict[str, Any]]:
    if attribute == "count":
        classes = [{"instrument": "instrument_count", "polyp": "polyp_count"}[entity]]
    elif attribute == "location":
        classes = [{"instrument": "instrument_location", "landmark": "landmark_location"}[entity]]
    else:
        classes = [question_class for question_class, _ in ENTITY_SIGNALS[entity]]
    evidence = []
    for question_class in classes:
        for answer in sorted(values.get((image_id, question_class), set())):
            evidence.append(
                {
                    "question_class": question_class,
                    "answer": answer,
                    "test_row_indices": sorted(source_rows.get((image_id, question_class, answer), set())),
                }
            )
    return evidence


def build_candidate_sets(
    values: Mapping[Tuple[str, str], Set[str]], image_ids: Set[str]
) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    count_values = {"one", "two", "three or more"}
    location_values = {"upper region", "central region", "lower region"}
    instrument_polyp = {
        image_id
        for image_id in image_ids
        if count_label(values, image_id, "instrument") in count_values
        and entity_status(values, image_id, "instrument") == "present"
        and entity_status(values, image_id, "polyp") == "absent"
    }
    polyp_instrument = {
        image_id
        for image_id in image_ids
        if count_label(values, image_id, "polyp") in count_values
        and entity_status(values, image_id, "polyp") == "present"
        and entity_status(values, image_id, "instrument") == "absent"
    }
    instrument_landmark = {
        image_id
        for image_id in image_ids
        if location_label(values, image_id, "instrument") in location_values
        and entity_status(values, image_id, "instrument") == "present"
        and entity_status(values, image_id, "landmark") == "absent"
    }
    landmark_instrument = {
        image_id
        for image_id in image_ids
        if location_label(values, image_id, "landmark") in location_values
        and entity_status(values, image_id, "landmark") == "present"
        and entity_status(values, image_id, "instrument") == "absent"
    }
    return instrument_polyp, polyp_instrument, instrument_landmark, landmark_instrument


def maximum_blocks(
    instrument_polyp: Set[str],
    polyp_instrument: Set[str],
    instrument_landmark: Set[str],
    landmark_instrument: Set[str],
) -> List[Tuple[str, str, str]]:
    graph = nx.Graph()
    for left, right, stratum in (
        (instrument_polyp, polyp_instrument, "count"),
        (instrument_landmark, landmark_instrument, "location"),
    ):
        for image_a in sorted(left):
            for image_b in sorted(right):
                if image_a == image_b:
                    continue
                if not graph.has_edge(image_a, image_b):
                    graph.add_edge(image_a, image_b, strata=set())
                graph[image_a][image_b]["strata"].add(stratum)
                graph[image_a][image_b]["weight"] = int("location" in graph[image_a][image_b]["strata"])
    matching = nx.max_weight_matching(graph, maxcardinality=True, weight="weight")
    result = []
    for first, second in matching:
        strata = graph[first][second]["strata"]
        stratum = "location" if "location" in strata else "count"
        if stratum == "count":
            image_a, image_b = (first, second) if first in instrument_polyp else (second, first)
        else:
            image_a, image_b = (first, second) if first in instrument_landmark else (second, first)
        result.append((stratum, image_a, image_b))
    return sorted(result, key=lambda item: (item[0], stable_hash(SEED, *item)))


def option_layouts(attribute: str, count: int) -> List[Dict[str, str]]:
    vocabulary = COUNT_OPTIONS if attribute == "count" else LOCATION_OPTIONS
    permutations = list(itertools.permutations(vocabulary))
    random.Random(int(stable_hash(SEED, attribute, "layouts")[:16], 16)).shuffle(permutations)
    layouts = [permutations[index % len(permutations)] for index in range(count)]
    return [{letter: value for letter, value in zip(LETTERS, layout)} for layout in layouts]


def answer_letter(options: Mapping[str, str], answer_text: str) -> str:
    matches = [letter for letter, text in options.items() if text == answer_text]
    if len(matches) != 1:
        raise ValueError(f"answer {answer_text!r} has {len(matches)} matches")
    return matches[0]


def make_probe(
    block: Mapping[str, Any],
    image_role: str,
    query_role: str,
    image_id: str,
    entity: str,
    premise_type: str,
    answer_text: str,
    evidence: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    options = block["options"]
    return {
        "id": f"{block['block_id']}::I{image_role}_Q{query_role}",
        "block_id": block["block_id"],
        "cell": f"I{image_role}_Q{query_role}",
        "image_role": image_role,
        "query_role": query_role,
        "query_id": f"{block['block_id']}::Q{query_role}",
        "image_path": f"data/Kvasir-VQA-x1/images/{image_id}.jpg",
        "source_image_id": image_id,
        "question": QUESTIONS[(entity, block["attribute_type"])],
        "options": options,
        "answer": answer_letter(options, answer_text),
        "answer_text": answer_text,
        "premise_type": premise_type,
        "target_entity": entity,
        "attribute_type": block["attribute_type"],
        "na_letter": answer_letter(options, NA_TEXT),
        "source_dataset": "Kvasir-VQA-x1",
        "source_split": "test",
        "atomic_evidence": list(evidence),
        "scoring_type": "crossed_real_image_premise_control",
        "construction_rule": "test atomic-annotation 2x2 image-by-query crossover",
    }


def materialize(
    matches: Sequence[Tuple[str, str, str]],
    values: Mapping[Tuple[str, str], Set[str]],
    source_rows: Mapping[Tuple[str, str, str], Set[int]],
    project_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
    stratum_sizes = Counter(stratum for stratum, _, _ in matches)
    layouts = {attribute: iter(option_layouts(attribute, count)) for attribute, count in stratum_sizes.items()}
    blocks: List[Dict[str, Any]] = []
    probes: List[Dict[str, Any]] = []
    image_hashes: Dict[str, str] = {}
    for stratum, image_a, image_b in matches:
        entity_b = "polyp" if stratum == "count" else "landmark"
        label_a = count_label(values, image_a, "instrument") if stratum == "count" else location_label(values, image_a, "instrument")
        label_b = count_label(values, image_b, entity_b) if stratum == "count" else location_label(values, image_b, entity_b)
        if label_a is None or label_b is None:
            raise AssertionError("matched image lost its supported label")
        block_id = "xctrl_" + stable_hash(stratum, image_a, image_b)[:16]
        block = {
            "block_id": block_id,
            "attribute_type": stratum,
            "entity_a": "instrument",
            "entity_b": entity_b,
            "image_a_id": image_a,
            "image_b_id": image_b,
            "image_a_supported_text": label_a,
            "image_b_supported_text": label_b,
            "options": next(layouts[stratum]),
            "image_a_supported_evidence": atomic_evidence(values, source_rows, image_a, "instrument", stratum),
            "image_a_entity_b_absence_evidence": atomic_evidence(values, source_rows, image_a, entity_b),
            "image_b_supported_evidence": atomic_evidence(values, source_rows, image_b, entity_b, stratum),
            "image_b_instrument_absence_evidence": atomic_evidence(values, source_rows, image_b, "instrument"),
            "source_split": "test",
            "model_outputs_consulted": False,
        }
        for image_id in (image_a, image_b):
            path = project_root / "data" / "Kvasir-VQA-x1" / "images" / f"{image_id}.jpg"
            if not path.exists():
                raise FileNotFoundError(path)
            image_hashes[image_id] = sha256(path)
        probes.extend(
            [
                make_probe(block, "A", "A", image_a, "instrument", "true", label_a, block["image_a_supported_evidence"]),
                make_probe(block, "A", "B", image_a, entity_b, "false", NA_TEXT, block["image_a_entity_b_absence_evidence"]),
                make_probe(block, "B", "A", image_b, "instrument", "false", NA_TEXT, block["image_b_instrument_absence_evidence"]),
                make_probe(block, "B", "B", image_b, entity_b, "true", label_b, block["image_b_supported_evidence"]),
            ]
        )
        blocks.append(block)
    probes.sort(key=lambda row: stable_hash(SEED, "manifest-order", row["id"]))
    return blocks, probes, image_hashes


def group_by(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, List[Mapping[str, Any]]]:
    result: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row[field])].append(row)
    return result


def validate(
    blocks: Sequence[Mapping[str, Any]], probes: Sequence[Mapping[str, Any]], image_hashes: Mapping[str, str]
) -> Dict[str, Any]:
    if len(blocks) != 165 or len(probes) != 660:
        raise AssertionError(f"unexpected size: blocks={len(blocks)} probes={len(probes)}")
    stratum_counts = Counter(block["attribute_type"] for block in blocks)
    if stratum_counts != {"count": 159, "location": 6}:
        raise AssertionError(f"unexpected strata: {stratum_counts}")
    image_counts = Counter(row["source_image_id"] for row in probes)
    if len(image_counts) != 330 or set(image_counts.values()) != {2}:
        raise AssertionError("images are not globally unique across blocks")
    if len(set(image_hashes.values())) != 330:
        raise AssertionError("selected images contain duplicate file contents")
    query_counts = Counter(row["query_id"] for row in probes)
    if len(query_counts) != 330 or set(query_counts.values()) != {2}:
        raise AssertionError("queries are not paired exactly twice")
    if len({row["id"] for row in probes}) != 660:
        raise AssertionError("probe IDs are not unique")
    for block_id, rows in group_by(probes, "block_id").items():
        if len(rows) != 4 or {row["cell"] for row in rows} != {"IA_QA", "IA_QB", "IB_QA", "IB_QB"}:
            raise AssertionError(f"invalid cells: {block_id}")
        if Counter(row["premise_type"] for row in rows) != {"true": 2, "false": 2}:
            raise AssertionError(f"invalid premises: {block_id}")
        if len({json.dumps(row["options"], sort_keys=True) for row in rows}) != 1:
            raise AssertionError(f"option mismatch: {block_id}")
    for image_id, rows in group_by(probes, "source_image_id").items():
        if Counter(row["premise_type"] for row in rows) != {"true": 1, "false": 1}:
            raise AssertionError(f"image is not crossed: {image_id}")
    for query_id, rows in group_by(probes, "query_id").items():
        if Counter(row["premise_type"] for row in rows) != {"true": 1, "false": 1}:
            raise AssertionError(f"query is not crossed: {query_id}")
        for field in ("question", "options", "target_entity", "attribute_type", "na_letter"):
            if rows[0][field] != rows[1][field]:
                raise AssertionError(f"query mismatch {query_id}: {field}")
    na_positions = Counter(answer_letter(block["options"], NA_TEXT) for block in blocks)
    return {
        "blocks": len(blocks),
        "probes": len(probes),
        "unique_images": len(image_counts),
        "unique_image_hashes": len(set(image_hashes.values())),
        "unique_queries": len(query_counts),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "na_position_counts": dict(sorted(na_positions.items())),
        "premise_counts": dict(sorted(Counter(row["premise_type"] for row in probes).items())),
        "entity_premise_counts": {
            "|".join(key): value
            for key, value in sorted(Counter((row["target_entity"], row["premise_type"]) for row in probes).items())
        },
        "model_outputs_consulted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    values, source_rows, image_ids, test_rows, atomic_count = load_atomic_annotations(args.test_parquet)
    candidates = build_candidate_sets(values, image_ids)
    matches = maximum_blocks(*candidates)
    blocks, probes, image_hashes = materialize(matches, values, source_rows, args.project_root.resolve())
    validation = validate(blocks, probes, image_hashes)

    output_dir = args.output_dir.resolve()
    block_path = output_dir / "crossed_blocks_165.jsonl"
    manifest_path = output_dir / "crossed_manifest_660.jsonl"
    image_hash_path = output_dir / "crossed_image_sha256.json"
    smoke_blocks = [item for item in blocks if item["attribute_type"] == "count"][:5]
    smoke_blocks += [item for item in blocks if item["attribute_type"] == "location"][:1]
    smoke_ids = {block["block_id"] for block in smoke_blocks}
    smoke_rows = [row for row in probes if row["block_id"] in smoke_ids]
    smoke_path = output_dir / "crossed_smoke_24.jsonl"
    write_jsonl(block_path, blocks)
    write_jsonl(manifest_path, probes)
    write_jsonl(smoke_path, smoke_rows)
    write_json(image_hash_path, image_hashes)

    validation.update(
        {
            "schema": "crossed_atomic_test_control_validation_v1",
            "seed": SEED,
            "test_rows": test_rows,
            "test_images": len(image_ids),
            "atomic_annotations": atomic_count,
            "candidate_set_sizes": {
                "instrument_present_polyp_absent_count": len(candidates[0]),
                "polyp_present_instrument_absent_count": len(candidates[1]),
                "instrument_present_landmark_absent_location": len(candidates[2]),
                "landmark_present_instrument_absent_location": len(candidates[3]),
            },
            "test_parquet": args.test_parquet.name,
            "test_parquet_sha256": sha256(args.test_parquet),
            "block_manifest": block_path.name,
            "block_manifest_sha256": sha256(block_path),
            "probe_manifest": manifest_path.name,
            "probe_manifest_sha256": sha256(manifest_path),
            "image_hash_manifest": image_hash_path.name,
            "image_hash_manifest_sha256": sha256(image_hash_path),
            "smoke_manifest": smoke_path.name,
            "smoke_manifest_sha256": sha256(smoke_path),
            "smoke_rows": len(smoke_rows),
        }
    )
    validation_path = output_dir / "crossed_validation.json"
    write_json(validation_path, validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
