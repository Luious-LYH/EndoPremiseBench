#!/usr/bin/env python3
"""
Balanced Subset Builder for EndoPremiseBench v2

Constructs source-balanced, premise-balanced, attribute-capped subsets from premise candidates.

Usage:
    python build_balanced_subset.py --input results/premise_candidates_v2.jsonl

Outputs:
    - results/premise_balanced_main_v2.jsonl (main subset)
    - results/premise_balanced_smoke_v2.jsonl (smoke test subset)
    - tables/balanced_subset_report_v2.md (distribution report)

Balancing constraints:
    - Source balance: no single source > 70%
    - Premise balance: true/false imbalance <= 3:1
    - Attribute cap: no single attribute > 60%
    - Presence exclusion: 'presence' excluded or capped at <= 10% (medical POPE risk)
    - N/A positions: tracked in statistics
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any
import random


def load_candidates(input_path: Path) -> List[Dict[str, Any]]:
    """Load premise candidates from JSONL."""
    candidates = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def compute_distribution(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute distribution statistics for a sample set."""
    total = len(samples)

    # Source distribution
    sources = Counter(s['source_dataset'] for s in samples)

    # Premise distribution
    premises = Counter(s['premise_type'] for s in samples)

    # Attribute distribution
    attributes = Counter(s['attribute_type'] for s in samples)

    # N/A position distribution (answer == "C" or "not applicable" in options)
    na_positions = []
    for s in samples:
        answer = s['answer']
        options = s['options']
        # Find which option is N/A
        na_key = None
        for key, val in options.items():
            if 'not applicable' in val.lower():
                na_key = key
                break
        if na_key:
            na_positions.append(na_key)

    na_position_counts = Counter(na_positions)

    return {
        'total': total,
        'sources': dict(sources),
        'premises': dict(premises),
        'attributes': dict(attributes),
        'na_positions': dict(na_position_counts)
    }


def check_balance_constraints(dist: Dict[str, Any]) -> Dict[str, Any]:
    """Check if distribution meets balance constraints."""
    total = dist['total']
    gates = {}

    # Source balance: no single source > 70%
    max_source_pct = max(count / total for count in dist['sources'].values()) if dist['sources'] else 0
    gates['source_balance'] = max_source_pct <= 0.70
    gates['max_source_pct'] = max_source_pct

    # Premise balance: true/false imbalance <= 3:1
    true_count = dist['premises'].get('true', 0)
    false_count = dist['premises'].get('false', 0)
    if false_count > 0:
        premise_ratio = true_count / false_count
    elif true_count > 0:
        premise_ratio = float('inf')
    else:
        premise_ratio = 1.0

    gates['premise_balance'] = premise_ratio <= 3.0 and premise_ratio >= 1/3.0
    gates['premise_ratio'] = premise_ratio

    # Attribute cap: no single attribute > 60%
    max_attr_pct = max(count / total for count in dist['attributes'].values()) if dist['attributes'] else 0
    gates['attribute_cap'] = max_attr_pct <= 0.60
    gates['max_attr_pct'] = max_attr_pct

    # Presence cap: 'presence' <= 10% (or excluded)
    presence_count = dist['attributes'].get('presence', 0)
    presence_pct = presence_count / total if total > 0 else 0
    gates['presence_cap'] = presence_pct <= 0.10
    gates['presence_pct'] = presence_pct

    # Overall gate
    gates['all_pass'] = all([
        gates['source_balance'],
        gates['premise_balance'],
        gates['attribute_cap'],
        gates['presence_cap']
    ])

    return gates


def build_balanced_subset(
    candidates: List[Dict[str, Any]],
    target_size: int,
    exclude_presence: bool = True,
    max_presence_pct: float = 0.10
) -> List[Dict[str, Any]]:
    """
    Build a balanced subset using stratified sampling with soft constraints.

    Strategy:
    1. Filter out 'presence' attribute if exclude_presence=True
    2. Group by (source, premise, attribute)
    3. Compute target quotas based on balance constraints
    4. Sample from each group proportionally
    """
    # Filter presence if needed
    if exclude_presence:
        filtered = [c for c in candidates if c['attribute_type'] != 'presence']
    else:
        filtered = candidates.copy()

    # Group by (source, premise, attribute)
    groups = defaultdict(list)
    for c in filtered:
        key = (c['source_dataset'], c['premise_type'], c['attribute_type'])
        groups[key].append(c)

    # Compute source distribution in filtered data.
    source_counts = Counter(c['source_dataset'] for c in filtered)
    sources = sorted(source_counts.keys())

    # Compute source quotas with an equal-first policy. EndoBench has fewer
    # non-presence samples, so we fill the smaller source first and redistribute
    # the remaining budget without letting one source dominate.
    source_quotas = {}
    equal_quota = target_size // max(1, len(sources))
    for source in sources:
        source_quotas[source] = min(equal_quota, source_counts[source])

    max_source_quota = int(target_size * 0.60)
    current_total = sum(source_quotas.values())
    if current_total < target_size:
        remainder = target_size - current_total
        while remainder > 0:
            progressed = False
            for source in sorted(sources, key=lambda s: source_quotas[s]):
                if remainder == 0:
                    break
                if source_quotas[source] >= max_source_quota:
                    continue
                if source_quotas[source] >= source_counts[source]:
                    continue
                source_quotas[source] += 1
                remainder -= 1
                progressed = True
            if not progressed:
                break

    # Premise: aim for 50/50 within each source quota, but adapt to availability
    subset = []
    for source in sources:
        source_quota = source_quotas.get(source, 0)
        if source_quota == 0:
            continue

        source_samples = [c for c in filtered if c['source_dataset'] == source]
        true_samples = [c for c in source_samples if c['premise_type'] == 'true']
        false_samples = [c for c in source_samples if c['premise_type'] == 'false']

        # Compute ideal 50/50 split
        ideal_true = source_quota // 2
        ideal_false = source_quota - ideal_true

        # Adjust based on availability
        actual_true = min(ideal_true, len(true_samples))
        actual_false = min(ideal_false, len(false_samples))

        # If one premise type is insufficient, compensate with the other
        if actual_true < ideal_true and len(false_samples) > actual_false:
            # Need more false to compensate
            shortfall = ideal_true - actual_true
            actual_false = min(actual_false + shortfall, len(false_samples))
        elif actual_false < ideal_false and len(true_samples) > actual_true:
            # Need more true to compensate
            shortfall = ideal_false - actual_false
            actual_true = min(actual_true + shortfall, len(true_samples))

        # Sample
        if actual_true > 0:
            subset.extend(random.sample(true_samples, actual_true))
        if actual_false > 0:
            subset.extend(random.sample(false_samples, actual_false))

    # Shuffle final subset
    random.shuffle(subset)

    # Trim to exact target size if needed
    return subset[:target_size]


def build_smoke_subset(
    candidates: List[Dict[str, Any]],
    target_size: int = 100,
    exclude_presence: bool = True
) -> List[Dict[str, Any]]:
    """
    Build a smoke test subset with coverage across sources, premises, and attributes.

    Strategy: stratified sampling to ensure diversity.
    """
    # Filter presence if needed
    if exclude_presence:
        filtered = [c for c in candidates if c['attribute_type'] != 'presence']
    else:
        filtered = candidates

    # Group by (source, premise, attribute)
    groups = defaultdict(list)
    for c in filtered:
        key = (c['source_dataset'], c['premise_type'], c['attribute_type'])
        groups[key].append(c)

    # Sample from each group
    smoke = []
    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    samples_per_group = max(1, target_size // len(group_keys))

    for key in group_keys:
        group_samples = groups[key]
        sample_count = min(samples_per_group, len(group_samples))
        smoke.extend(random.sample(group_samples, sample_count))

        if len(smoke) >= target_size:
            break

    if len(smoke) < target_size:
        used_ids = {s.get('id') for s in smoke}
        remaining = [s for s in filtered if s.get('id') not in used_ids]
        random.shuffle(remaining)
        smoke.extend(remaining[: target_size - len(smoke)])

    # Trim to exact size
    random.shuffle(smoke)
    return smoke[:target_size]


def save_subset(subset: List[Dict[str, Any]], output_path: Path):
    """Save subset to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in subset:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def generate_report(
    main_dist: Dict[str, Any],
    main_gates: Dict[str, Any],
    smoke_dist: Dict[str, Any],
    smoke_gates: Dict[str, Any],
    output_path: Path
):
    """Generate markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Balanced Subset Report v2\n\n")
        f.write(f"Generated: {Path(__file__).name}\n\n")

        # Main subset
        f.write("## Main Subset\n\n")
        f.write(f"**Total samples:** {main_dist['total']}\n\n")

        f.write("### Source Distribution\n\n")
        f.write("| Source | Count | Percentage |\n")
        f.write("|--------|-------|------------|\n")
        for source, count in sorted(main_dist['sources'].items()):
            pct = count / main_dist['total'] * 100
            f.write(f"| {source} | {count} | {pct:.1f}% |\n")
        f.write("\n")

        f.write("### Premise Distribution\n\n")
        f.write("| Premise | Count | Percentage |\n")
        f.write("|---------|-------|------------|\n")
        for premise, count in sorted(main_dist['premises'].items()):
            pct = count / main_dist['total'] * 100
            f.write(f"| {premise} | {count} | {pct:.1f}% |\n")
        f.write("\n")

        f.write("### Attribute Distribution\n\n")
        f.write("| Attribute | Count | Percentage |\n")
        f.write("|-----------|-------|------------|\n")
        for attr, count in sorted(main_dist['attributes'].items(), key=lambda x: -x[1]):
            pct = count / main_dist['total'] * 100
            f.write(f"| {attr} | {count} | {pct:.1f}% |\n")
        f.write("\n")

        f.write("### N/A Position Distribution\n\n")
        f.write("| Position | Count | Percentage |\n")
        f.write("|----------|-------|------------|\n")
        for pos, count in sorted(main_dist['na_positions'].items()):
            pct = count / main_dist['total'] * 100
            f.write(f"| {pos} | {count} | {pct:.1f}% |\n")
        f.write("\n")

        f.write("### Balance Gates\n\n")
        f.write("| Gate | Status | Value |\n")
        f.write("|------|--------|-------|\n")
        f.write(f"| Source balance (<=70%) | {'PASS' if main_gates['source_balance'] else 'FAIL'} | {main_gates['max_source_pct']*100:.1f}% |\n")
        f.write(f"| Premise balance (<=3:1) | {'PASS' if main_gates['premise_balance'] else 'FAIL'} | {main_gates['premise_ratio']:.2f}:1 |\n")
        f.write(f"| Attribute cap (<=60%) | {'PASS' if main_gates['attribute_cap'] else 'FAIL'} | {main_gates['max_attr_pct']*100:.1f}% |\n")
        f.write(f"| Presence cap (<=10%) | {'PASS' if main_gates['presence_cap'] else 'FAIL'} | {main_gates['presence_pct']*100:.1f}% |\n")
        f.write(f"| **Overall** | {'PASS' if main_gates['all_pass'] else 'FAIL'} | - |\n")
        f.write("\n")

        # Smoke subset
        f.write("## Smoke Subset\n\n")
        f.write(f"**Total samples:** {smoke_dist['total']}\n\n")

        f.write("### Source Distribution\n\n")
        f.write("| Source | Count | Percentage |\n")
        f.write("|--------|-------|------------|\n")
        for source, count in sorted(smoke_dist['sources'].items()):
            pct = count / smoke_dist['total'] * 100
            f.write(f"| {source} | {count} | {pct:.1f}% |\n")
        f.write("\n")

        f.write("### Premise Distribution\n\n")
        f.write("| Premise | Count | Percentage |\n")
        f.write("|---------|-------|------------|\n")
        for premise, count in sorted(smoke_dist['premises'].items()):
            pct = count / smoke_dist['total'] * 100
            f.write(f"| {premise} | {count} | {pct:.1f}% |\n")
        f.write("\n")

        f.write("### Attribute Distribution\n\n")
        f.write("| Attribute | Count | Percentage |\n")
        f.write("|-----------|-------|------------|\n")
        for attr, count in sorted(smoke_dist['attributes'].items(), key=lambda x: -x[1]):
            pct = count / smoke_dist['total'] * 100
            f.write(f"| {attr} | {count} | {pct:.1f}% |\n")
        f.write("\n")

        f.write("### Balance Gates\n\n")
        f.write("| Gate | Status | Value |\n")
        f.write("|------|--------|-------|\n")
        f.write(f"| Source balance (<=70%) | {'PASS' if smoke_gates['source_balance'] else 'FAIL'} | {smoke_gates['max_source_pct']*100:.1f}% |\n")
        f.write(f"| Premise balance (<=3:1) | {'PASS' if smoke_gates['premise_balance'] else 'FAIL'} | {smoke_gates['premise_ratio']:.2f}:1 |\n")
        f.write(f"| Attribute cap (<=60%) | {'PASS' if smoke_gates['attribute_cap'] else 'FAIL'} | {smoke_gates['max_attr_pct']*100:.1f}% |\n")
        f.write(f"| Presence cap (<=10%) | {'PASS' if smoke_gates['presence_cap'] else 'FAIL'} | {smoke_gates['presence_pct']*100:.1f}% |\n")
        f.write(f"| **Overall** | {'PASS' if smoke_gates['all_pass'] else 'FAIL'} | - |\n")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description='Build balanced subsets for EndoPremiseBench v2')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file with premise candidates')
    parser.add_argument('--main-size', type=int, default=6000, help='Target size for main subset')
    parser.add_argument('--smoke-size', type=int, default=100, help='Target size for smoke subset')
    parser.add_argument('--exclude-presence', action='store_true', default=True, help='Exclude presence attribute')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Paths
    input_path = Path(args.input)
    output_dir = input_path.parent
    tables_dir = input_path.parent.parent / 'tables'

    main_output = output_dir / 'premise_balanced_main_v2.jsonl'
    smoke_output = output_dir / 'premise_balanced_smoke_v2.jsonl'
    report_output = tables_dir / 'balanced_subset_report_v2.md'

    print(f"Loading candidates from {input_path}...")
    candidates = load_candidates(input_path)
    print(f"Loaded {len(candidates)} candidates")

    print(f"\nBuilding main subset (target size: {args.main_size})...")
    main_subset = build_balanced_subset(
        candidates,
        target_size=args.main_size,
        exclude_presence=args.exclude_presence
    )
    print(f"Main subset size: {len(main_subset)}")

    print(f"\nBuilding smoke subset (target size: {args.smoke_size})...")
    smoke_subset = build_smoke_subset(
        candidates,
        target_size=args.smoke_size,
        exclude_presence=args.exclude_presence
    )
    print(f"Smoke subset size: {len(smoke_subset)}")

    # Compute distributions
    print("\nComputing distributions...")
    main_dist = compute_distribution(main_subset)
    main_gates = check_balance_constraints(main_dist)

    smoke_dist = compute_distribution(smoke_subset)
    smoke_gates = check_balance_constraints(smoke_dist)

    # Save outputs
    print(f"\nSaving main subset to {main_output}...")
    save_subset(main_subset, main_output)

    print(f"Saving smoke subset to {smoke_output}...")
    save_subset(smoke_subset, smoke_output)

    print(f"Generating report to {report_output}...")
    generate_report(main_dist, main_gates, smoke_dist, smoke_gates, report_output)

    print("\n=== Summary ===")
    print(f"Main subset: {len(main_subset)} samples")
    print(f"  Balance gates: {'PASS' if main_gates['all_pass'] else 'FAIL'}")
    print(f"Smoke subset: {len(smoke_subset)} samples")
    print(f"  Balance gates: {'PASS' if smoke_gates['all_pass'] else 'FAIL'}")
    print(f"\nOutputs:")
    print(f"  - {main_output}")
    print(f"  - {smoke_output}")
    print(f"  - {report_output}")
    print("\nDone.")


if __name__ == '__main__':
    main()
