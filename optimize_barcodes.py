"""
Barcode Optimizer
=================
Generates an optimal set of N barcodes of configurable length that:
  1. Have GC content within 40-60% (±1 base by default for a larger candidate pool)
  2. Maximize the minimum pairwise Hamming distance
  3. Avoid homopolymer runs >= 3
  4. Avoid dinucleotide repeats (e.g., ATATAT, GCGCGC)
  5. Check pairwise sequence complementarity (dimer risk)
  6. Report Levenshtein distance alongside Hamming distance

Note on GC content:
  By default, GC count is allowed to be ±1 base beyond strict 40-60%,
  giving a larger candidate pool. Use --strict to enforce exactly 40-60% GC.

Usage:
  python optimize_barcodes.py -n 96
  python optimize_barcodes.py -n 96 --length 10
  python optimize_barcodes.py -n 96 --strict
  python optimize_barcodes.py -n 96 --restarts 20
"""

import argparse
import math
import random
import sys
import time
from itertools import product

import numpy as np


def gc_content(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)


def has_homopolymer(seq, max_run=2):
    """Return True if seq contains a homopolymer run > max_run."""
    count = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            count += 1
            if count > max_run:
                return True
        else:
            count = 1
    return False


def has_dinucleotide_repeat(seq, max_repeat=2):
    """Return True if seq contains a dinucleotide repeated more than max_repeat times.
    E.g., ATATAT = 3 repeats of AT; max_repeat=2 would flag this."""
    for i in range(len(seq) - 1):
        dinuc = seq[i:i+2]
        if dinuc[0] == dinuc[1]:  # Skip homopolymer dinucs (handled separately)
            continue
        count = 1
        pos = i + 2
        while pos + 1 < len(seq) and seq[pos:pos+2] == dinuc:
            count += 1
            pos += 2
        if count > max_repeat:
            return True
    return False


def hamming(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def levenshtein(s1, s2):
    """Compute Levenshtein (edit) distance between two sequences."""
    n, m = len(s1), len(s2)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[m]


def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(comp[b] for b in reversed(seq))


def max_complementarity(s1, s2):
    """Compute maximum number of Watson-Crick base pairs between s1 and the
    reverse complement of s2, across all sliding alignments with min overlap of 3.
    This measures dimer formation potential."""
    rc2 = reverse_complement(s2)
    l = len(s1)
    max_bp = 0
    # Slide rc2 across s1 with min overlap of 3
    for offset in range(-(l - 3), l - 2):
        bp = 0
        for i in range(l):
            j = i - offset
            if 0 <= j < len(rc2):
                if s1[i] == rc2[j]:
                    bp += 1
        if bp > max_bp:
            max_bp = bp
    return max_bp


def generate_candidates(length=8, gc_min=0.375, gc_max=0.625, max_homopolymer=2, max_dinuc_repeat=2):
    """Generate all valid barcode candidates meeting GC, homopolymer, and dinucleotide constraints."""
    candidates = []
    for bases in product('ACGT', repeat=length):
        seq = ''.join(bases)
        gc = gc_content(seq)
        if gc < gc_min or gc > gc_max:
            continue
        if has_homopolymer(seq, max_homopolymer):
            continue
        if has_dinucleotide_repeat(seq, max_dinuc_repeat):
            continue
        candidates.append(seq)
    return candidates


def compute_distance_matrix(candidates):
    """Precompute pairwise Hamming distance matrix using numpy for speed."""
    n = len(candidates)
    # Encode sequences as numpy arrays of integers
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.array([[mapping[c] for c in seq] for seq in candidates], dtype=np.int8)
    # Compute pairwise distances in batches to save memory
    # For ~20k candidates, full matrix would be huge; we'll use on-the-fly computation
    return encoded


def greedy_select(candidates, encoded, n_barcodes, seed=None):
    """
    Unified multi-objective greedy selection.
    At each step, picks the candidate that yields the best lexicographic
    score (max min_HD, max min_LD, min max_CP) for the resulting set.

    Tracks per-candidate:
      - min_hd_to_set: minimum Hamming distance to any selected barcode
      - min_ld_to_set: minimum Levenshtein distance to any selected barcode
      - max_cp_to_set: maximum complementarity to any selected barcode

    Uses HD as a fast pre-filter (numpy) before computing expensive LD/CP.
    """
    rng = random.Random(seed)
    n_cand = len(candidates)

    # Start with a random barcode
    start_idx = rng.randint(0, n_cand - 1)
    selected_indices = [start_idx]
    selected_set = {start_idx}

    # Initialize per-candidate tracking arrays
    # HD via numpy (fast)
    min_hd_to_set = np.sum(encoded != encoded[start_idx], axis=1).astype(np.int16)
    # LD and CP via Python (only computed for viable candidates)
    min_ld_to_set = np.full(n_cand, 999, dtype=np.int16)
    max_cp_to_set = np.zeros(n_cand, dtype=np.int16)

    start_seq = candidates[start_idx]
    for i in range(n_cand):
        if i == start_idx:
            continue
        min_ld_to_set[i] = levenshtein(candidates[i], start_seq)
        max_cp_to_set[i] = max_complementarity(candidates[i], start_seq)

    # Global score of current set (only matters after >= 2 barcodes)
    global_min_hd = 999
    global_min_ld = 999
    global_max_cp = 0

    for step in range(1, n_barcodes):
        # For each unselected candidate, compute what the set score would be
        # if we added it:
        #   would_min_hd = min(global_min_hd, min_hd_to_set[i])
        #   would_min_ld = min(global_min_ld, min_ld_to_set[i])
        #   would_max_cp = max(global_max_cp, max_cp_to_set[i])
        # Pick best lexicographic (max would_min_hd, max would_min_ld, min would_max_cp)

        best_idx = -1
        best_would = (-1, -1, 999)  # (would_min_hd, would_min_ld, -would_max_cp proxy)

        # Pre-filter: only consider candidates with min_hd_to_set >= some threshold
        # First pass: find the max achievable min_hd
        temp_hd = min_hd_to_set.copy()
        for idx in selected_set:
            temp_hd[idx] = -1
        max_achievable_hd = int(np.max(temp_hd))

        if max_achievable_hd <= 0:
            print(f"  Warning: Could not find {n_barcodes} barcodes with HD > 0. Stopped at {step}.")
            break

        # The best possible would_min_hd = min(global_min_hd, max_achievable_hd)
        # We only need candidates whose min_hd_to_set >= min(global_min_hd, max_achievable_hd)
        # to have any chance of being optimal on the HD axis.
        # But we also want to explore candidates that tie on HD but win on LD/CP.
        # Strategy: consider all candidates with min_hd_to_set >= best HD threshold
        target_hd = min(global_min_hd, max_achievable_hd) if step > 1 else max_achievable_hd

        # Get viable candidates
        viable_mask = (temp_hd >= target_hd)
        viable_indices = np.where(viable_mask)[0]

        for i in viable_indices:
            would_min_hd = min(global_min_hd, int(min_hd_to_set[i])) if step > 1 else int(min_hd_to_set[i])
            would_min_ld = min(global_min_ld, int(min_ld_to_set[i])) if step > 1 else int(min_ld_to_set[i])
            would_max_cp = max(global_max_cp, int(max_cp_to_set[i])) if step > 1 else int(max_cp_to_set[i])

            # Lexicographic comparison: max hd, max ld, min cp
            cand_score = (would_min_hd, would_min_ld, -would_max_cp)
            if cand_score > best_would:
                best_would = cand_score
                best_idx = int(i)

        if best_idx < 0:
            print(f"  Warning: No viable candidate at step {step}. Stopped.")
            break

        selected_indices.append(best_idx)
        selected_set.add(best_idx)

        # Update global score
        if step == 1:
            global_min_hd = int(min_hd_to_set[best_idx])
            global_min_ld = int(min_ld_to_set[best_idx])
            global_max_cp = int(max_cp_to_set[best_idx])
        else:
            global_min_hd = min(global_min_hd, int(min_hd_to_set[best_idx]))
            global_min_ld = min(global_min_ld, int(min_ld_to_set[best_idx]))
            global_max_cp = max(global_max_cp, int(max_cp_to_set[best_idx]))

        # Update per-candidate arrays with distances to the newly added barcode
        new_seq = candidates[best_idx]
        new_hd = np.sum(encoded != encoded[best_idx], axis=1).astype(np.int16)
        min_hd_to_set = np.minimum(min_hd_to_set, new_hd)

        for i in range(n_cand):
            if i in selected_set:
                continue
            ld = levenshtein(candidates[i], new_seq)
            if ld < min_ld_to_set[i]:
                min_ld_to_set[i] = ld
            cp = max_complementarity(candidates[i], new_seq)
            if cp > max_cp_to_set[i]:
                max_cp_to_set[i] = cp

    return selected_indices, (global_min_hd, global_min_ld, global_max_cp)


def evaluate_set_hd(encoded, indices):
    """Return the minimum pairwise HD for the selected set (fast, for greedy phase)."""
    subset = encoded[indices]
    n = len(indices)
    min_hd = 999
    for i in range(n):
        for j in range(i + 1, n):
            hd = np.sum(subset[i] != subset[j])
            if hd < min_hd:
                min_hd = hd
                if min_hd <= 1:
                    return min_hd
    return min_hd


def evaluate_set_full(candidates, indices):
    """Evaluate a barcode set with the full objective.
    Returns (min_hd, min_ld, max_cp, mean_hd, mean_ld, mean_cp) tuple."""
    seqs = [candidates[i] for i in indices]
    n = len(seqs)
    min_hd = 999
    min_ld = 999
    max_cp = 0
    sum_hd = 0
    sum_ld = 0
    sum_cp = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            hd = hamming(seqs[i], seqs[j])
            ld = levenshtein(seqs[i], seqs[j])
            cp = max_complementarity(seqs[i], seqs[j])
            if hd < min_hd:
                min_hd = hd
            if ld < min_ld:
                min_ld = ld
            if cp > max_cp:
                max_cp = cp
            sum_hd += hd
            sum_ld += ld
            sum_cp += cp
            count += 1
    return (min_hd, min_ld, max_cp,
            sum_hd / count, sum_ld / count, sum_cp / count)


def score_is_better(new_score, old_score):
    """Lexicographic comparison using the full scalar.
    Scores are (min_hd, min_ld, max_cp, mean_hd, mean_ld, mean_cp) tuples."""
    return score_to_scalar(new_score) > score_to_scalar(old_score)


def compute_pair_metrics_for_idx(seq, other_seqs):
    """Compute min HD, min LD, and max CP between seq and a list of other sequences.
    Returns (min_hd, min_ld, max_cp)."""
    min_hd = 999
    min_ld = 999
    max_cp = 0
    for other in other_seqs:
        hd = hamming(seq, other)
        if hd < min_hd:
            min_hd = hd
        ld = levenshtein(seq, other)
        if ld < min_ld:
            min_ld = ld
        cp = max_complementarity(seq, other)
        if cp > max_cp:
            max_cp = cp
    return (min_hd, min_ld, max_cp)


def score_to_scalar(score):
    """Convert (min_hd, min_ld, max_cp, mean_hd, mean_ld, mean_cp) to a scalar.
    Weights ensure strict lexicographic priority:
      min_HD >> min_LD >> max_CP >> mean_HD >> mean_LD >> mean_CP
    Higher scalar = better."""
    min_hd, min_ld, max_cp, mean_hd, mean_ld, mean_cp = score
    return (10000.0 * min_hd
            + 100.0 * min_ld
            - 10.0 * max_cp
            + 1.0 * mean_hd
            + 0.5 * mean_ld
            - 0.1 * mean_cp)


def simulated_annealing(candidates, initial_indices, encoded,
                        iterations=50000, t_start=5.0, t_end=0.01,
                        seed=None):
    """Simulated annealing optimizer for barcode set selection.

    Objective: maximize min_HD, then maximize min_LD, then minimize max_CP.

    Key features:
      - Targeted swaps: 50% of the time, swap a barcode involved in the worst
        (bottleneck) pair instead of a random one.
      - Incremental evaluation: only recomputes the N-1 pairs involving the
        swapped position.
      - Cached bottleneck tracking with numpy arrays for fast min/max.

    Args:
        candidates: list of all candidate barcode sequences
        initial_indices: starting set of selected candidate indices (from greedy)
        encoded: numpy-encoded candidate sequences for fast HD
        iterations: number of SA iterations
        t_start: starting temperature
        t_end: ending temperature
        seed: random seed
    """
    rng = random.Random(seed)
    n_cand = len(candidates)
    n_sel = len(initial_indices)
    n_pairs = n_sel * (n_sel - 1) // 2

    # Cooling schedule: geometric
    alpha = (t_end / t_start) ** (1.0 / max(iterations - 1, 1))

    # Initialize selected set
    selected = list(initial_indices)
    selected_set = set(selected)

    # Use numpy arrays for pairwise metrics (faster min/max than dicts)
    # Map pair (i,j) where i<j to a flat index: idx = i*n_sel - i*(i+1)//2 + j - i - 1
    def pair_idx(i, j):
        a, b = (i, j) if i < j else (j, i)
        return a * n_sel - a * (a + 1) // 2 + b - a - 1

    hd_arr = np.zeros(n_pairs, dtype=np.int16)
    ld_arr = np.zeros(n_pairs, dtype=np.int16)
    cp_arr = np.zeros(n_pairs, dtype=np.int16)

    for i in range(n_sel):
        si = candidates[selected[i]]
        for j in range(i + 1, n_sel):
            sj = candidates[selected[j]]
            pidx = pair_idx(i, j)
            hd_arr[pidx] = hamming(si, sj)
            ld_arr[pidx] = levenshtein(si, sj)
            cp_arr[pidx] = max_complementarity(si, sj)

    # Build a mask for each position: which pair indices involve that position
    pos_masks = []
    for p in range(n_sel):
        mask = np.zeros(n_pairs, dtype=bool)
        for j in range(n_sel):
            if j != p:
                mask[pair_idx(p, j)] = True
        pos_masks.append(mask)

    # Running sums for fast mean computation
    sum_hd = int(np.sum(hd_arr))
    sum_ld = int(np.sum(ld_arr))
    sum_cp = int(np.sum(cp_arr))

    def get_score():
        return (int(np.min(hd_arr)), int(np.min(ld_arr)), int(np.max(cp_arr)),
                sum_hd / n_pairs, sum_ld / n_pairs, sum_cp / n_pairs)

    def find_bottleneck_positions():
        """Find positions involved in the worst pairs (lowest HD, then lowest LD, then highest CP)."""
        positions = set()
        # Find pair(s) with minimum HD
        min_hd = int(np.min(hd_arr))
        hd_worst = np.where(hd_arr == min_hd)[0]
        for pidx in hd_worst[:5]:  # Limit to avoid too many
            # Reverse map flat index to (i,j)
            for i in range(n_sel):
                for j in range(i + 1, n_sel):
                    if pair_idx(i, j) == pidx:
                        positions.add(i)
                        positions.add(j)
                        break
                else:
                    continue
                break
        # Also find pair(s) with minimum LD
        min_ld = int(np.min(ld_arr))
        ld_worst = np.where(ld_arr == min_ld)[0]
        for pidx in ld_worst[:5]:
            for i in range(n_sel):
                for j in range(i + 1, n_sel):
                    if pair_idx(i, j) == pidx:
                        positions.add(i)
                        positions.add(j)
                        break
                else:
                    continue
                break
        # Also highest CP
        max_cp = int(np.max(cp_arr))
        cp_worst = np.where(cp_arr == max_cp)[0]
        for pidx in cp_worst[:5]:
            for i in range(n_sel):
                for j in range(i + 1, n_sel):
                    if pair_idx(i, j) == pidx:
                        positions.add(i)
                        positions.add(j)
                        break
                else:
                    continue
                break
        return list(positions)

    current_score = get_score()
    current_scalar = score_to_scalar(current_score)
    best_score = current_score
    best_scalar = current_scalar
    best_selected = list(selected)

    # Precompute list of unselected indices for fast sampling
    unselected = list(set(range(n_cand)) - selected_set)

    accepted = 0
    improved = 0
    temp = t_start
    log_interval = max(1, iterations // 20)
    bottleneck_positions = find_bottleneck_positions()
    bottleneck_refresh = max(1, iterations // 100)

    def fmt_score(s):
        return f"HD>={s[0]} LD>={s[1]} CP<={s[2]} avgHD={s[3]:.2f} avgLD={s[4]:.2f} avgCP={s[5]:.2f}"

    print(f"    Initial: {fmt_score(current_score)} (scalar={current_scalar:.1f})")

    for it in range(iterations):
        # Refresh bottleneck positions periodically
        if it % bottleneck_refresh == 0:
            bottleneck_positions = find_bottleneck_positions()

        # 50% targeted swap (bottleneck position), 50% random
        if rng.random() < 0.5 and bottleneck_positions:
            pos = rng.choice(bottleneck_positions)
        else:
            pos = rng.randint(0, n_sel - 1)

        old_cand_idx = selected[pos]

        # Pick a random unselected candidate to swap in
        unsel_pos = rng.randint(0, len(unselected) - 1)
        new_cand_idx = unselected[unsel_pos]
        new_seq = candidates[new_cand_idx]

        # Compute new pair metrics for position `pos` with all others
        pos_mask = pos_masks[pos]
        not_pos_mask = ~pos_mask

        new_hd_vals = np.empty(n_sel - 1, dtype=np.int16)
        new_ld_vals = np.empty(n_sel - 1, dtype=np.int16)
        new_cp_vals = np.empty(n_sel - 1, dtype=np.int16)
        pair_indices = []  # flat indices for the pairs involving pos
        k = 0
        for j in range(n_sel):
            if j == pos:
                continue
            pidx = pair_idx(pos, j)
            pair_indices.append(pidx)
            other_seq = candidates[selected[j]]
            new_hd_vals[k] = hamming(new_seq, other_seq)
            new_ld_vals[k] = levenshtein(new_seq, other_seq)
            new_cp_vals[k] = max_complementarity(new_seq, other_seq)
            k += 1

        # Compute new global score incrementally using numpy
        # non-pos metrics
        non_pos_min_hd = int(np.min(hd_arr[not_pos_mask])) if np.any(not_pos_mask) else 999
        non_pos_min_ld = int(np.min(ld_arr[not_pos_mask])) if np.any(not_pos_mask) else 999
        non_pos_max_cp = int(np.max(cp_arr[not_pos_mask])) if np.any(not_pos_mask) else 0

        new_min_hd = min(non_pos_min_hd, int(np.min(new_hd_vals)))
        new_min_ld = min(non_pos_min_ld, int(np.min(new_ld_vals)))
        new_max_cp = max(non_pos_max_cp, int(np.max(new_cp_vals)))

        # Incremental mean: subtract old pos-pair sums, add new
        old_pos_sum_hd = int(np.sum(hd_arr[pos_mask]))
        old_pos_sum_ld = int(np.sum(ld_arr[pos_mask]))
        old_pos_sum_cp = int(np.sum(cp_arr[pos_mask]))
        new_pos_sum_hd = int(np.sum(new_hd_vals))
        new_pos_sum_ld = int(np.sum(new_ld_vals))
        new_pos_sum_cp = int(np.sum(new_cp_vals))

        new_sum_hd = sum_hd - old_pos_sum_hd + new_pos_sum_hd
        new_sum_ld = sum_ld - old_pos_sum_ld + new_pos_sum_ld
        new_sum_cp = sum_cp - old_pos_sum_cp + new_pos_sum_cp

        new_score = (new_min_hd, new_min_ld, new_max_cp,
                     new_sum_hd / n_pairs, new_sum_ld / n_pairs, new_sum_cp / n_pairs)
        new_scalar = score_to_scalar(new_score)

        # Acceptance decision
        delta = new_scalar - current_scalar
        accept = False
        if delta > 0:
            accept = True
        elif temp > 0 and delta > -50:  # Clip very bad moves
            prob = math.exp(delta / temp)
            if rng.random() < prob:
                accept = True

        if accept:
            # Apply the swap
            selected_set.discard(old_cand_idx)
            selected_set.add(new_cand_idx)
            selected[pos] = new_cand_idx

            # Update unselected list
            unselected[unsel_pos] = old_cand_idx

            # Update running sums
            sum_hd = new_sum_hd
            sum_ld = new_sum_ld
            sum_cp = new_sum_cp

            # Update pair metric arrays
            for k, pidx in enumerate(pair_indices):
                hd_arr[pidx] = new_hd_vals[k]
                ld_arr[pidx] = new_ld_vals[k]
                cp_arr[pidx] = new_cp_vals[k]

            current_score = new_score
            current_scalar = new_scalar
            accepted += 1

            # Track best
            if score_is_better(new_score, best_score):
                best_score = new_score
                best_scalar = new_scalar
                best_selected = list(selected)
                improved += 1

        # Cool
        temp *= alpha

        # Logging
        if (it + 1) % log_interval == 0 or it == 0:
            print(f"    iter {it+1:>6}/{iterations}: T={temp:.3f}  "
                  f"cur=({fmt_score(current_score)})  "
                  f"best=scalar={best_scalar:.1f}  "
                  f"accepted={accepted} improved={improved}")

    print(f"\n    SA complete: {accepted} accepted, {improved} improvements")
    print(f"    Best: {fmt_score(best_score)} (scalar={best_scalar:.1f})")
    return best_selected, best_score


def main():
    parser = argparse.ArgumentParser(description="Optimize a barcode set for maximum pairwise distance and minimum complementarity")
    parser.add_argument('-n', '--num-barcodes', type=int, default=96,
                        help='Number of barcodes to generate (default: 96)')
    parser.add_argument('-l', '--length', type=int, default=8,
                        help='Barcode length in nucleotides (default: 8)')
    parser.add_argument('--strict', action='store_true',
                        help='Strict GC filter: only barcodes with GC count in [ceil(L*0.4), floor(L*0.6)]')
    parser.add_argument('--max-homopolymer', type=int, default=2,
                        help='Maximum allowed homopolymer run length (default: 2)')
    parser.add_argument('--max-dinuc-repeat', type=int, default=2,
                        help='Maximum allowed dinucleotide repeat count (default: 2)')
    parser.add_argument('--restarts', type=int, default=3,
                        help='Number of random restarts for greedy initialization (default: 3)')
    parser.add_argument('--sa-iterations', type=int, default=50000,
                        help='Simulated annealing iterations (default: 50000)')
    parser.add_argument('--sa-temp', type=float, default=5.0,
                        help='SA starting temperature (default: 5.0)')
    parser.add_argument('--sa-cool', type=float, default=0.01,
                        help='SA ending temperature (default: 0.01)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file for barcode list (CSV)')
    args = parser.parse_args()

    # Compute GC bounds as integer counts for the given length, then convert to fractions.
    # This ensures sensible GC ranges for any barcode length.
    if args.strict:
        gc_count_min = math.ceil(args.length * 0.40)
        gc_count_max = math.floor(args.length * 0.60)
    else:
        # Allow ±1 GC base beyond strict 40-60% for a larger candidate pool
        gc_count_min = max(0, math.floor(args.length * 0.40) - 1)
        gc_count_max = min(args.length, math.ceil(args.length * 0.60) + 1)
    gc_min = gc_count_min / args.length
    gc_max = gc_count_max / args.length

    print(f"=== Barcode Optimizer ===")
    print(f"  Barcode length:    {args.length} nt")
    print(f"  Target count:      {args.num_barcodes}")
    print(f"  GC range:          {gc_min*100:.1f}% - {gc_max*100:.1f}%")
    print(f"  Max homopolymer:   {args.max_homopolymer}")
    print(f"  Max dinuc repeat:  {args.max_dinuc_repeat}")
    print(f"  Greedy restarts:   {args.restarts}")
    print(f"  SA iterations:     {args.sa_iterations}")
    print(f"  SA temp range:     {args.sa_temp} -> {args.sa_cool}")
    print()

    # Step 1: Generate candidate pool
    print("Generating candidate barcodes...", end=" ", flush=True)
    t0 = time.time()
    candidates = generate_candidates(
        length=args.length,
        gc_min=gc_min,
        gc_max=gc_max,
        max_homopolymer=args.max_homopolymer,
        max_dinuc_repeat=args.max_dinuc_repeat,
    )
    t1 = time.time()
    print(f"{len(candidates)} candidates in {t1-t0:.1f}s")

    if len(candidates) < args.num_barcodes:
        print(f"ERROR: Only {len(candidates)} candidates available, cannot select {args.num_barcodes}.")
        sys.exit(1)

    # Step 2: Encode for fast distance computation
    print("Encoding sequences...", end=" ", flush=True)
    encoded = compute_distance_matrix(candidates)
    print("done")

    # Step 3: Greedy initialization
    print(f"\nPhase 1: Greedy initialization ({args.restarts} restarts)...")
    base_seed = args.seed if args.seed is not None else random.randint(0, 999999)

    best_indices = None
    best_score = (-1, -1, 999, 0.0, 0.0, 999.0)  # (min_hd, min_ld, max_cp, mean_hd, mean_ld, mean_cp)

    for r in range(args.restarts):
        seed = base_seed + r
        indices, _ = greedy_select(candidates, encoded, args.num_barcodes, seed=seed)
        # Compute full 6-element score including means
        score = evaluate_set_full(candidates, indices)
        improved = ""
        if score_is_better(score, best_score):
            best_score = score
            best_indices = indices
            improved = " << NEW BEST"
        print(f"  Restart {r+1:>3}/{args.restarts}: HD>={score[0]} LD>={score[1]} CP<={score[2]} "
              f"avgHD={score[3]:.2f} avgLD={score[4]:.2f} avgCP={score[5]:.2f}{improved}")

    print(f"  Greedy best: HD>={best_score[0]} LD>={best_score[1]} CP<={best_score[2]} "
          f"avgHD={best_score[3]:.2f} avgLD={best_score[4]:.2f} avgCP={best_score[5]:.2f}")

    # Step 4: Simulated annealing refinement
    print(f"\nPhase 2: Simulated annealing ({args.sa_iterations} iterations)...")
    sa_seed = base_seed + args.restarts
    refined_indices, refined_score = simulated_annealing(
        candidates, best_indices, encoded,
        iterations=args.sa_iterations,
        t_start=args.sa_temp,
        t_end=args.sa_cool,
        seed=sa_seed,
    )
    print(f"\n  Final: HD>={refined_score[0]} LD>={refined_score[1]} CP<={refined_score[2]} "
          f"avgHD={refined_score[3]:.2f} avgLD={refined_score[4]:.2f} avgCP={refined_score[5]:.2f}")

    # Step 5: Collect and display results
    selected = [candidates[i] for i in refined_indices]

    # Compute full stats: Hamming, Levenshtein, and Complementarity
    from itertools import combinations
    from collections import Counter

    print("\nComputing pairwise Hamming & Levenshtein distances...", end=" ", flush=True)
    all_hd = []
    all_ld = []
    all_cp = []
    pair_data = []  # (i_name, j_name, s1, s2, hd, ld, cp)
    for i, j in combinations(range(len(selected)), 2):
        s1, s2 = selected[i], selected[j]
        hd = hamming(s1, s2)
        ld = levenshtein(s1, s2)
        cp = max_complementarity(s1, s2)
        all_hd.append(hd)
        all_ld.append(ld)
        all_cp.append(cp)
        pair_data.append((i+1, j+1, s1, s2, hd, ld, cp))
    print("done")

    hd_counter = Counter(all_hd)
    ld_counter = Counter(all_ld)
    cp_counter = Counter(all_cp)

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(selected)} optimized {args.length}-nt barcodes")
    print(f"{'='*60}")

    # --- Hamming Distance ---
    print(f"\n  HAMMING DISTANCE")
    print(f"  Min pairwise HD:  {min(all_hd)}")
    print(f"  Max pairwise HD:  {max(all_hd)}")
    print(f"  Mean pairwise HD: {sum(all_hd)/len(all_hd):.2f}")
    print(f"  Distribution:")
    for hd_val in sorted(hd_counter.keys()):
        print(f"    HD={hd_val}: {hd_counter[hd_val]:>5} pairs")

    # --- Levenshtein Distance ---
    print(f"\n  LEVENSHTEIN (EDIT) DISTANCE")
    print(f"  Min pairwise LD:  {min(all_ld)}")
    print(f"  Max pairwise LD:  {max(all_ld)}")
    print(f"  Mean pairwise LD: {sum(all_ld)/len(all_ld):.2f}")
    print(f"  Distribution:")
    for ld_val in sorted(ld_counter.keys()):
        print(f"    LD={ld_val}: {ld_counter[ld_val]:>5} pairs")

    # --- Complementarity ---
    print(f"\n  SEQUENCE COMPLEMENTARITY (dimer risk)")
    print(f"  Max complementarity: {max(all_cp)} bp")
    print(f"  Mean complementarity: {sum(all_cp)/len(all_cp):.2f} bp")
    print(f"  Distribution:")
    for cp_val in sorted(cp_counter.keys()):
        print(f"    CP={cp_val}: {cp_counter[cp_val]:>5} pairs")

    # Flag high-complementarity pairs (>= 6 bp for 8-mers = 75% complementary)
    cp_threshold = max(6, int(args.length * 0.75))
    high_cp = [(i, j, s1, s2, hd, ld, cp) for i, j, s1, s2, hd, ld, cp in pair_data if cp >= cp_threshold]
    if high_cp:
        print(f"\n  WARNING: {len(high_cp)} pairs with complementarity >= {cp_threshold} bp (dimer risk):")
        print(f"  {'BC1':<5} {'Seq1':<10} {'BC2':<5} {'Seq2':<10} {'CP':>3}  {'RevComp(Seq2)':<10}")
        print(f"  {'-'*52}")
        for i, j, s1, s2, hd, ld, cp in sorted(high_cp, key=lambda x: -x[6])[:20]:
            print(f"  {i:<5} {s1:<10} {j:<5} {s2:<10} {cp:>3}  {reverse_complement(s2):<10}")
        if len(high_cp) > 20:
            print(f"  ... and {len(high_cp)-20} more pairs")
    else:
        print(f"\n  No pairs with complementarity >= {cp_threshold} bp. Low dimer risk.")

    # Flag low Levenshtein pairs
    low_ld = [(i, j, s1, s2, hd, ld, cp) for i, j, s1, s2, hd, ld, cp in pair_data if ld < 3]
    if low_ld:
        print(f"\n  WARNING: {len(low_ld)} pairs with Levenshtein distance < 3 (indel-vulnerable):")
        print(f"  {'BC1':<5} {'Seq1':<10} {'BC2':<5} {'Seq2':<10} {'HD':>3} {'LD':>3}")
        print(f"  {'-'*40}")
        for i, j, s1, s2, hd, ld, cp in sorted(low_ld, key=lambda x: x[5]):
            print(f"  {i:<5} {s1:<10} {j:<5} {s2:<10} {hd:>3} {ld:>3}")

    # --- GC Content ---
    gc_vals = [gc_content(s) * 100 for s in selected]
    print(f"\n  GC CONTENT")
    print(f"  Range:  {min(gc_vals):.1f}% - {max(gc_vals):.1f}%")
    print(f"  Mean:   {sum(gc_vals)/len(gc_vals):.1f}%")

    print(f"\n{'Index':<6} {'Barcode':<10} {'GC%':>5}")
    print("-" * 24)
    for idx, bc in enumerate(selected, 1):
        gc = gc_content(bc) * 100
        print(f"{idx:<6} {bc:<10} {gc:>5.1f}")

    # Step 5: Save to file
    if args.output:
        with open(args.output, 'w') as f:
            f.write("index,barcode\n")
            for idx, bc in enumerate(selected, 1):
                f.write(f"{idx},{bc}\n")
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
