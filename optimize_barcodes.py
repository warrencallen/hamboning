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
  7. Minimize complementarity to user-supplied primer sequences (soft blacklist)

Note on GC content:
  By default, GC count is allowed to be ±1 base beyond strict 40-60%,
  giving a larger candidate pool. Use --strict to enforce exactly 40-60% GC.

Primer blacklist:
  Use --primers to supply primer sequences (comma-separated). The optimizer
  will penalize barcodes that have high complementarity to any primer,
  reducing the risk of non-specific primer-barcode binding.

Usage:
  python optimize_barcodes.py -n 96
  python optimize_barcodes.py -n 96 --length 10
  python optimize_barcodes.py -n 96 --strict
  python optimize_barcodes.py -n 96 --restarts 20
  python optimize_barcodes.py -n 96 --primers ATCGATCG,GCTAGCTA
"""

import argparse
import math
import multiprocessing
import os
import random
import sys
import time
from itertools import product

import numpy as np
from numba import njit, int8, int16, int32

# --- Numba-accelerated core functions ---

# Complement table: A(0)->T(3), C(1)->G(2), G(2)->C(1), T(3)->A(0)
_COMP_TABLE = np.array([3, 2, 1, 0], dtype=np.int8)
_BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
_BASE_UNMAP = ['A', 'C', 'G', 'T']


def seq_to_arr(seq):
    """Convert a DNA string to a numpy int8 array."""
    return np.array([_BASE_MAP[c] for c in seq], dtype=np.int8)


def arr_to_seq(arr):
    """Convert a numpy int8 array back to a DNA string."""
    return ''.join(_BASE_UNMAP[i] for i in arr)


@njit(int32(int8[:], int8[:]), cache=True)
def hamming_nb(s1, s2):
    """Hamming distance between two encoded sequences."""
    d = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            d += 1
    return d


@njit(int32(int8[:], int8[:]), cache=True)
def levenshtein_nb(s1, s2):
    """Levenshtein distance between two encoded sequences."""
    n = len(s1)
    m = len(s2)
    # Use a single row DP
    dp = np.empty(m + 1, dtype=np.int32)
    for j in range(m + 1):
        dp[j] = j
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                val = prev
                if dp[j] < val:
                    val = dp[j]
                if dp[j-1] < val:
                    val = dp[j-1]
                dp[j] = 1 + val
            prev = temp
    return dp[m]


@njit(int8[:](int8[:], int8[:]), cache=True)
def _reverse_complement_nb(seq, comp_table):
    """Return the reverse complement of an encoded sequence."""
    n = len(seq)
    rc = np.empty(n, dtype=np.int8)
    for i in range(n):
        rc[i] = comp_table[seq[n - 1 - i]]
    return rc


@njit(int32(int8[:], int8[:], int8[:]), cache=True)
def max_complementarity_nb(s1, s2, comp_table):
    """Max Watson-Crick base pairs across sliding alignments (min overlap 3)."""
    rc2 = _reverse_complement_nb(s2, comp_table)
    l1 = len(s1)
    l2 = len(rc2)
    max_bp = 0
    for offset in range(-(l1 - 3), l1 - 2):
        bp = 0
        for i in range(l1):
            j = i - offset
            if 0 <= j < l2:
                if s1[i] == rc2[j]:
                    bp += 1
        if bp > max_bp:
            max_bp = bp
    return max_bp


@njit(cache=True)
def _batch_levenshtein_update(encoded_all, new_seq, min_ld_to_set, selected_mask, n_cand):
    """Update min_ld_to_set for all unselected candidates against new_seq."""
    for i in range(n_cand):
        if selected_mask[i]:
            continue
        ld = levenshtein_nb(encoded_all[i], new_seq)
        if ld < min_ld_to_set[i]:
            min_ld_to_set[i] = ld


@njit(cache=True)
def _batch_complementarity_update(encoded_all, new_seq, max_cp_to_set, selected_mask, n_cand, comp_table):
    """Update max_cp_to_set for all unselected candidates against new_seq."""
    for i in range(n_cand):
        if selected_mask[i]:
            continue
        cp = max_complementarity_nb(encoded_all[i], new_seq, comp_table)
        if cp > max_cp_to_set[i]:
            max_cp_to_set[i] = cp


@njit(cache=True)
def _sa_compute_new_pairs(encoded_all, new_seq, selected, pos, n_sel, comp_table):
    """Compute HD, LD, CP for new_seq against all other selected barcodes."""
    new_hd = np.empty(n_sel - 1, dtype=np.int16)
    new_ld = np.empty(n_sel - 1, dtype=np.int16)
    new_cp = np.empty(n_sel - 1, dtype=np.int16)
    k = 0
    for j in range(n_sel):
        if j == pos:
            continue
        other = encoded_all[selected[j]]
        new_hd[k] = hamming_nb(new_seq, other)
        new_ld[k] = levenshtein_nb(new_seq, other)
        new_cp[k] = max_complementarity_nb(new_seq, other, comp_table)
        k += 1
    return new_hd, new_ld, new_cp


# --- Python wrapper functions (for string-based API compatibility) ---

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
    """Return True if seq contains a dinucleotide repeated more than max_repeat times."""
    for i in range(len(seq) - 1):
        dinuc = seq[i:i+2]
        if dinuc[0] == dinuc[1]:
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
    reverse complement of s2, across all sliding alignments with min overlap of 3."""
    rc2 = reverse_complement(s2)
    l = len(s1)
    max_bp = 0
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


def compute_primer_complementarity(seq, primers):
    """Compute maximum complementarity between seq and any primer sequence."""
    max_cp = 0
    for primer in primers:
        cp = max_complementarity(seq, primer)
        if cp > max_cp:
            max_cp = cp
    return max_cp


@njit(cache=True)
def _batch_primer_complementarity(encoded_all, primer_arrs, primer_lens, comp_table, n_cand, n_primers):
    """Compute max primer complementarity for all candidates using numba."""
    result = np.zeros(n_cand, dtype=np.int16)
    for i in range(n_cand):
        max_cp = 0
        for p in range(n_primers):
            primer_p = primer_arrs[p, :primer_lens[p]]
            cp = max_complementarity_nb(encoded_all[i], primer_p, comp_table)
            if cp > max_cp:
                max_cp = cp
        result[i] = max_cp
    return result


@njit(cache=True)
def _generate_candidates_nb(length, gc_count_min, gc_count_max, max_homopolymer, max_dinuc_repeat):
    """Numba-accelerated DFS candidate generation with pruning.

    Uses an iterative DFS with fixed-size state arrays (no Python objects).
    Tracks GC count, homopolymer runs, and even/odd-aligned dinucleotide
    repeat chains at each tree level for O(1) constraint checking.
    """
    # Pre-allocate output buffer (generous upper bound, will trim)
    max_out = 1_000_000
    result = np.empty((max_out, length), dtype=np.int8)
    count = 0

    # State arrays indexed by position (tree level)
    seq = np.zeros(length, dtype=np.int8)
    next_base = np.zeros(length, dtype=np.int8)  # Which base to try next
    gc = np.zeros(length + 1, dtype=np.int32)     # Running GC count
    run = np.ones(length + 1, dtype=np.int32)      # Homopolymer run length
    # Even/odd aligned dinucleotide chain tracking
    ch_even_cnt = np.zeros(length + 1, dtype=np.int32)
    ch_even_d0 = np.full(length + 1, -1, dtype=np.int8)
    ch_even_d1 = np.full(length + 1, -1, dtype=np.int8)
    ch_odd_cnt = np.zeros(length + 1, dtype=np.int32)
    ch_odd_d0 = np.full(length + 1, -1, dtype=np.int8)
    ch_odd_d1 = np.full(length + 1, -1, dtype=np.int8)

    pos = 0
    next_base[0] = 0

    while pos >= 0:
        if pos == length:
            # Valid candidate found
            if count >= max_out:
                new_result = np.empty((max_out * 2, length), dtype=np.int8)
                for i in range(max_out):
                    new_result[i] = result[i]
                result = new_result
                max_out *= 2
            for i in range(length):
                result[count, i] = seq[i]
            count += 1
            pos -= 1
            continue

        b = next_base[pos]
        if b >= 4:
            # Exhausted all bases, backtrack
            next_base[pos] = 0
            pos -= 1
            continue

        next_base[pos] = b + 1
        seq[pos] = b

        # GC bounds pruning
        new_gc = gc[pos] + (1 if (b == 1 or b == 2) else 0)
        remaining = length - pos - 1
        if new_gc + remaining < gc_count_min:
            continue
        if new_gc > gc_count_max:
            continue
        gc[pos + 1] = new_gc

        # Homopolymer pruning
        if pos > 0 and b == seq[pos - 1]:
            new_run = run[pos] + 1
            if new_run > max_homopolymer:
                continue
        else:
            new_run = 1
        run[pos + 1] = new_run

        # Dinucleotide repeat pruning (O(1) per step)
        new_ch_even_cnt = ch_even_cnt[pos]
        new_ch_even_d0 = ch_even_d0[pos]
        new_ch_even_d1 = ch_even_d1[pos]
        new_ch_odd_cnt = ch_odd_cnt[pos]
        new_ch_odd_d0 = ch_odd_d0[pos]
        new_ch_odd_d1 = ch_odd_d1[pos]

        skip = False
        if pos >= 1:
            d0 = seq[pos - 1]
            d1 = b
            if d0 != d1:  # Non-homopolymer dinuc
                if pos % 2 == 1:  # pos-1 is even → even-aligned
                    if d0 == new_ch_even_d0 and d1 == new_ch_even_d1:
                        new_ch_even_cnt = ch_even_cnt[pos] + 1
                    else:
                        new_ch_even_cnt = 1
                        new_ch_even_d0 = d0
                        new_ch_even_d1 = d1
                    if new_ch_even_cnt > max_dinuc_repeat:
                        skip = True
                else:  # pos-1 is odd → odd-aligned
                    if d0 == new_ch_odd_d0 and d1 == new_ch_odd_d1:
                        new_ch_odd_cnt = ch_odd_cnt[pos] + 1
                    else:
                        new_ch_odd_cnt = 1
                        new_ch_odd_d0 = d0
                        new_ch_odd_d1 = d1
                    if new_ch_odd_cnt > max_dinuc_repeat:
                        skip = True
            else:
                # Homopolymer dinuc breaks chain at this alignment
                if pos % 2 == 1:
                    new_ch_even_cnt = 0
                    new_ch_even_d0 = -1
                    new_ch_even_d1 = -1
                else:
                    new_ch_odd_cnt = 0
                    new_ch_odd_d0 = -1
                    new_ch_odd_d1 = -1
        if skip:
            continue

        ch_even_cnt[pos + 1] = new_ch_even_cnt
        ch_even_d0[pos + 1] = new_ch_even_d0
        ch_even_d1[pos + 1] = new_ch_even_d1
        ch_odd_cnt[pos + 1] = new_ch_odd_cnt
        ch_odd_d0[pos + 1] = new_ch_odd_d0
        ch_odd_d1[pos + 1] = new_ch_odd_d1

        pos += 1

    return result[:count]


_CHAR_LOOKUP = np.array([ord('A'), ord('C'), ord('G'), ord('T')], dtype=np.uint8)


def generate_candidates(length=8, gc_min=0.375, gc_max=0.625, max_homopolymer=2, max_dinuc_repeat=2):
    """Generate all valid barcode candidates using numba-accelerated DFS with pruning.

    Returns (candidates_strings, encoded_array) tuple.
    For large candidate pools, string conversion uses vectorized numpy operations.
    """
    # Compute integer GC count bounds
    gc_count_min = 0
    gc_count_max = length
    for gc in range(length + 1):
        if gc / length >= gc_min:
            gc_count_min = gc
            break
    for gc in range(length, -1, -1):
        if gc / length <= gc_max:
            gc_count_max = gc
            break

    encoded = _generate_candidates_nb(length, gc_count_min, gc_count_max, max_homopolymer, max_dinuc_repeat)
    # Fast vectorized string conversion
    char_codes = _CHAR_LOOKUP[encoded]
    fmt = f'S{length}'
    candidates = char_codes.view(fmt).ravel().astype(f'U{length}').tolist()
    return candidates, encoded


def compute_distance_matrix(candidates):
    """Precompute pairwise Hamming distance matrix using numpy for speed."""
    n = len(candidates)
    # Encode sequences as numpy arrays of integers
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.array([[mapping[c] for c in seq] for seq in candidates], dtype=np.int8)
    # Compute pairwise distances in batches to save memory
    # For ~20k candidates, full matrix would be huge; we'll use on-the-fly computation
    return encoded


def greedy_select(candidates, encoded, n_barcodes, seed=None, primer_cp=None, encoded_nb=None):
    """
    Unified multi-objective greedy selection.
    Uses numba-accelerated functions for LD and CP computation.
    """
    rng = random.Random(seed)
    n_cand = len(candidates)

    # Start with a random barcode
    start_idx = rng.randint(0, n_cand - 1)
    selected_indices = [start_idx]
    selected_mask = np.zeros(n_cand, dtype=np.bool_)
    selected_mask[start_idx] = True

    # Initialize per-candidate tracking arrays
    min_hd_to_set = np.sum(encoded != encoded[start_idx], axis=1).astype(np.int16)
    min_ld_to_set = np.full(n_cand, 999, dtype=np.int16)
    max_cp_to_set = np.zeros(n_cand, dtype=np.int16)

    # Use numba batch functions for initial LD/CP computation
    start_seq_arr = encoded_nb[start_idx]
    _batch_levenshtein_update(encoded_nb, start_seq_arr, min_ld_to_set, selected_mask, n_cand)
    _batch_complementarity_update(encoded_nb, start_seq_arr, max_cp_to_set, selected_mask, n_cand, _COMP_TABLE)

    global_min_hd = 999
    global_min_ld = 999
    global_max_cp = 0
    global_max_primer_cp = int(primer_cp[start_idx]) if primer_cp is not None else 0

    for step in range(1, n_barcodes):
        best_idx = -1
        best_would = (-1, -1, 999, 999)

        temp_hd = min_hd_to_set.copy()
        temp_hd[selected_mask] = -1
        max_achievable_hd = int(np.max(temp_hd))

        if max_achievable_hd <= 0:
            print(f"  Warning: Could not find {n_barcodes} barcodes with HD > 0. Stopped at {step}.")
            break

        target_hd = min(global_min_hd, max_achievable_hd) if step > 1 else max_achievable_hd
        viable_indices = np.where(temp_hd >= target_hd)[0]

        for i in viable_indices:
            would_min_hd = min(global_min_hd, int(min_hd_to_set[i])) if step > 1 else int(min_hd_to_set[i])
            would_min_ld = min(global_min_ld, int(min_ld_to_set[i])) if step > 1 else int(min_ld_to_set[i])
            would_max_cp = max(global_max_cp, int(max_cp_to_set[i])) if step > 1 else int(max_cp_to_set[i])
            would_max_pcp = max(global_max_primer_cp, int(primer_cp[i])) if primer_cp is not None else 0

            cand_score = (would_min_hd, would_min_ld, -would_max_cp, -would_max_pcp)
            if cand_score > best_would:
                best_would = cand_score
                best_idx = int(i)

        if best_idx < 0:
            print(f"  Warning: No viable candidate at step {step}. Stopped.")
            break

        selected_indices.append(best_idx)
        selected_mask[best_idx] = True

        if step == 1:
            global_min_hd = int(min_hd_to_set[best_idx])
            global_min_ld = int(min_ld_to_set[best_idx])
            global_max_cp = int(max_cp_to_set[best_idx])
        else:
            global_min_hd = min(global_min_hd, int(min_hd_to_set[best_idx]))
            global_min_ld = min(global_min_ld, int(min_ld_to_set[best_idx]))
            global_max_cp = max(global_max_cp, int(max_cp_to_set[best_idx]))
        if primer_cp is not None:
            global_max_primer_cp = max(global_max_primer_cp, int(primer_cp[best_idx]))

        # Update per-candidate arrays with numba-accelerated batch functions
        new_hd = np.sum(encoded != encoded[best_idx], axis=1).astype(np.int16)
        min_hd_to_set = np.minimum(min_hd_to_set, new_hd)

        new_seq_arr = encoded_nb[best_idx]
        _batch_levenshtein_update(encoded_nb, new_seq_arr, min_ld_to_set, selected_mask, n_cand)
        _batch_complementarity_update(encoded_nb, new_seq_arr, max_cp_to_set, selected_mask, n_cand, _COMP_TABLE)

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


@njit(cache=True)
def _evaluate_set_full_nb(encoded, indices, n_sel, comp_table):
    """Numba-accelerated full set evaluation."""
    min_hd = 999
    min_ld = 999
    max_cp = 0
    sum_hd = 0
    sum_ld = 0
    sum_cp = 0
    count = 0
    for i in range(n_sel):
        si = encoded[indices[i]]
        for j in range(i + 1, n_sel):
            sj = encoded[indices[j]]
            hd = hamming_nb(si, sj)
            ld = levenshtein_nb(si, sj)
            cp = max_complementarity_nb(si, sj, comp_table)
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
    return min_hd, min_ld, max_cp, sum_hd, sum_ld, sum_cp, count


def evaluate_set_full(candidates, indices, primer_cp=None, encoded_nb=None):
    """Evaluate a barcode set with the full objective.
    Returns (min_hd, min_ld, max_cp, max_primer_cp, mean_hd, mean_ld, mean_cp, mean_primer_cp) tuple.
    Uses numba acceleration when encoded_nb is provided."""
    if encoded_nb is not None:
        indices_arr = np.array(indices, dtype=np.int32)
        min_hd, min_ld, max_cp, sum_hd, sum_ld, sum_cp, count = _evaluate_set_full_nb(
            encoded_nb, indices_arr, len(indices), _COMP_TABLE)
        min_hd, min_ld, max_cp = int(min_hd), int(min_ld), int(max_cp)
        sum_hd, sum_ld, sum_cp, count = int(sum_hd), int(sum_ld), int(sum_cp), int(count)
    else:
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
    # Primer complementarity metrics
    if primer_cp is not None:
        pcp_vals = [int(primer_cp[i]) for i in indices]
        max_pcp = max(pcp_vals)
        mean_pcp = sum(pcp_vals) / len(pcp_vals)
    else:
        max_pcp = 0
        mean_pcp = 0.0
    return (min_hd, min_ld, max_cp, max_pcp,
            sum_hd / count, sum_ld / count, sum_cp / count, mean_pcp)


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


@njit(cache=True)
def _compute_all_pairs_nb(encoded, indices, n_sel, comp_table):
    """Compute all pairwise HD, LD, CP for selected barcodes. Returns flat arrays."""
    n_pairs = n_sel * (n_sel - 1) // 2
    all_hd = np.empty(n_pairs, dtype=np.int32)
    all_ld = np.empty(n_pairs, dtype=np.int32)
    all_cp = np.empty(n_pairs, dtype=np.int32)
    k = 0
    for i in range(n_sel):
        si = encoded[indices[i]]
        for j in range(i + 1, n_sel):
            sj = encoded[indices[j]]
            all_hd[k] = hamming_nb(si, sj)
            all_ld[k] = levenshtein_nb(si, sj)
            all_cp[k] = max_complementarity_nb(si, sj, comp_table)
            k += 1
    return all_hd, all_ld, all_cp


def score_to_scalar(score):
    """Convert score tuple to a scalar.
    Score: (min_hd, min_ld, max_cp, max_primer_cp, mean_hd, mean_ld, mean_cp, mean_primer_cp)
    Weights ensure strict lexicographic priority:
      min_HD >> min_LD >> max_CP >> max_primer_CP >> mean_HD >> mean_LD >> mean_CP >> mean_primer_CP
    Higher scalar = better."""
    min_hd, min_ld, max_cp, max_pcp, mean_hd, mean_ld, mean_cp, mean_pcp = score
    return (10000.0 * min_hd
            + 100.0 * min_ld
            - 10.0 * max_cp
            - 5.0 * max_pcp
            + 1.0 * mean_hd
            + 0.5 * mean_ld
            - 0.1 * mean_cp
            - 0.05 * mean_pcp)


def tabu_lns_search(candidates, initial_indices, encoded,
                    iterations=10000, seed=None, primer_cp=None, encoded_nb=None):
    """Tabu Search with Large Neighborhood Search (LNS) optimizer."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed if seed is not None else random.randint(0, 2**31 - 1))
    n_cand = len(candidates)
    n_sel = len(initial_indices)
    n_pairs = n_sel * (n_sel - 1) // 2

    selected = list(initial_indices)
    selected_np = np.array(selected, dtype=np.int32)
    selected_set = set(selected)

    def pair_idx(i, j):
        a, b = (i, j) if i < j else (j, i)
        return a * n_sel - a * (a + 1) // 2 + b - a - 1

    # Precompute reverse index: flat_idx -> (i, j)
    reverse_pair = np.empty((n_pairs, 2), dtype=np.int32)
    for i in range(n_sel):
        for j in range(i + 1, n_sel):
            pidx = pair_idx(i, j)
            reverse_pair[pidx, 0] = i
            reverse_pair[pidx, 1] = j

    hd_arr = np.zeros(n_pairs, dtype=np.int16)
    ld_arr = np.zeros(n_pairs, dtype=np.int16)
    cp_arr = np.zeros(n_pairs, dtype=np.int16)

    # Initialize pairwise metrics using numba
    for i in range(n_sel):
        si = encoded_nb[selected[i]]
        for j in range(i + 1, n_sel):
            sj = encoded_nb[selected[j]]
            pidx = pair_idx(i, j)
            hd_arr[pidx] = hamming_nb(si, sj)
            ld_arr[pidx] = levenshtein_nb(si, sj)
            cp_arr[pidx] = max_complementarity_nb(si, sj, _COMP_TABLE)

    # Build pair index lists per position
    pos_pair_indices = []
    for p in range(n_sel):
        indices = []
        for j in range(n_sel):
            if j != p:
                indices.append(pair_idx(p, j))
        pos_pair_indices.append(np.array(indices, dtype=np.int32))

    # Precompute non-pos boolean masks
    pos_not_pair_mask = []
    for p in range(n_sel):
        mask = np.ones(n_pairs, dtype=np.bool_)
        mask[pos_pair_indices[p]] = False
        pos_not_pair_mask.append(mask)

    sum_hd = int(np.sum(hd_arr))
    sum_ld = int(np.sum(ld_arr))
    sum_cp = int(np.sum(cp_arr))

    if primer_cp is not None:
        sel_primer_cp = np.array([primer_cp[selected[i]] for i in range(n_sel)], dtype=np.int16)
        sum_primer_cp = int(np.sum(sel_primer_cp))
    else:
        sel_primer_cp = np.zeros(n_sel, dtype=np.int16)
        sum_primer_cp = 0

    # --- Approximate min HD tracking ---
    approx_min_hd = np.full(n_cand, 999, dtype=np.int16)
    for idx in selected:
        hd = np.sum(encoded != encoded[idx], axis=1).astype(np.int16)
        np.minimum(approx_min_hd, hd, out=approx_min_hd)
    for idx in selected_set:
        approx_min_hd[idx] = -1

    def refresh_approx_min_hd():
        nonlocal approx_min_hd
        approx_min_hd[:] = 999
        for idx in selected:
            hd = np.sum(encoded != encoded[idx], axis=1).astype(np.int16)
            np.minimum(approx_min_hd, hd, out=approx_min_hd)
        for idx in selected_set:
            approx_min_hd[idx] = -1

    def update_approx_after_swap(old_idx, new_idx):
        nonlocal approx_min_hd
        approx_min_hd[new_idx] = -1
        hd = np.sum(encoded != encoded[new_idx], axis=1).astype(np.int16)
        np.minimum(approx_min_hd, hd, out=approx_min_hd)
        approx_min_hd[new_idx] = -1  # re-mark after minimum
        hd_to_sel = np.sum(encoded[selected] != encoded[old_idx], axis=1)
        approx_min_hd[old_idx] = int(np.min(hd_to_sel))

    # --- Adaptive parameters ---
    tabu_tenure = max(10, n_sel // 4)
    n_eval = 30
    lns_interval = max(100, iterations // 30)
    lns_destroy_k = max(3, min(8, n_sel // 12))
    hd_refresh_interval = max(50, iterations // 100)
    max_no_improve = max(2000, iterations // 5)

    # Tabu list: candidate_idx -> iteration when tabu expires
    tabu = {}

    def get_screened_candidates(n_top):
        top_k = min(n_top, int(np.sum(approx_min_hd > 0)))
        if top_k <= 0:
            return np.array([], dtype=np.int32)
        return np.argpartition(approx_min_hd, -top_k)[-top_k:]

    def get_score():
        max_pcp = int(np.max(sel_primer_cp)) if primer_cp is not None else 0
        mean_pcp = sum_primer_cp / n_sel if primer_cp is not None else 0.0
        return (int(np.min(hd_arr)), int(np.min(ld_arr)), int(np.max(cp_arr)), max_pcp,
                sum_hd / n_pairs, sum_ld / n_pairs, sum_cp / n_pairs, mean_pcp)

    def find_bottleneck_positions():
        """Find positions involved in the worst pairs using precomputed reverse index."""
        positions = set()
        min_hd = int(np.min(hd_arr))
        for pidx in np.where(hd_arr == min_hd)[0][:5]:
            positions.add(int(reverse_pair[pidx, 0]))
            positions.add(int(reverse_pair[pidx, 1]))
        min_ld = int(np.min(ld_arr))
        for pidx in np.where(ld_arr == min_ld)[0][:5]:
            positions.add(int(reverse_pair[pidx, 0]))
            positions.add(int(reverse_pair[pidx, 1]))
        max_cp = int(np.max(cp_arr))
        for pidx in np.where(cp_arr == max_cp)[0][:5]:
            positions.add(int(reverse_pair[pidx, 0]))
            positions.add(int(reverse_pair[pidx, 1]))
        return list(positions)

    def fmt_score(s):
        base = f"HD>={s[0]} LD>={s[1]} CP<={s[2]}"
        if s[3] > 0:
            base += f" PCP<={s[3]}"
        base += f" avgHD={s[4]:.2f} avgLD={s[5]:.2f} avgCP={s[6]:.2f}"
        if s[7] > 0:
            base += f" avgPCP={s[7]:.2f}"
        return base

    def apply_swap(pos, new_cand_idx, new_hd_vals, new_ld_vals, new_cp_vals):
        """Apply a swap at position pos, replacing selected[pos] with new_cand_idx.
        Updates all mutable state. Returns old_cand_idx."""
        nonlocal sum_hd, sum_ld, sum_cp, sum_primer_cp

        old_cand_idx = selected[pos]
        pair_indices = pos_pair_indices[pos]

        old_pos_sum_hd = int(np.sum(hd_arr[pair_indices]))
        old_pos_sum_ld = int(np.sum(ld_arr[pair_indices]))
        old_pos_sum_cp = int(np.sum(cp_arr[pair_indices]))

        sum_hd = sum_hd - old_pos_sum_hd + int(np.sum(new_hd_vals))
        sum_ld = sum_ld - old_pos_sum_ld + int(np.sum(new_ld_vals))
        sum_cp = sum_cp - old_pos_sum_cp + int(np.sum(new_cp_vals))

        selected_set.discard(old_cand_idx)
        selected_set.add(new_cand_idx)
        selected[pos] = new_cand_idx
        selected_np[pos] = new_cand_idx

        if primer_cp is not None:
            new_pcp_val = int(primer_cp[new_cand_idx])
            sum_primer_cp = sum_primer_cp - int(sel_primer_cp[pos]) + new_pcp_val
            sel_primer_cp[pos] = new_pcp_val

        for k, pidx in enumerate(pair_indices):
            hd_arr[pidx] = new_hd_vals[k]
            ld_arr[pidx] = new_ld_vals[k]
            cp_arr[pidx] = new_cp_vals[k]

        return old_cand_idx

    def compute_score_for_swap(pos, new_hd_vals, new_ld_vals, new_cp_vals,
                               non_pos_min_hd, non_pos_min_ld, non_pos_max_cp,
                               old_sum_hd, old_sum_ld, old_sum_cp, new_cand_idx):
        """Compute score for a candidate swap using pre-computed position values."""
        new_min_hd = min(non_pos_min_hd, int(np.min(new_hd_vals)))
        new_min_ld = min(non_pos_min_ld, int(np.min(new_ld_vals)))
        new_max_cp = max(non_pos_max_cp, int(np.max(new_cp_vals)))

        new_pos_sum_hd = int(np.sum(new_hd_vals))
        new_pos_sum_ld = int(np.sum(new_ld_vals))
        new_pos_sum_cp = int(np.sum(new_cp_vals))

        cand_sum_hd = sum_hd - old_sum_hd + new_pos_sum_hd
        cand_sum_ld = sum_ld - old_sum_ld + new_pos_sum_ld
        cand_sum_cp = sum_cp - old_sum_cp + new_pos_sum_cp

        if primer_cp is not None:
            new_pcp_val = int(primer_cp[new_cand_idx])
            old_pcp_val = int(sel_primer_cp[pos])
            cand_sum_pcp = sum_primer_cp - old_pcp_val + new_pcp_val
            cur_max_pcp = int(np.max(sel_primer_cp))
            if old_pcp_val == cur_max_pcp:
                temp_pcp = sel_primer_cp.copy()
                temp_pcp[pos] = new_pcp_val
                cand_max_pcp = int(np.max(temp_pcp))
            else:
                cand_max_pcp = max(cur_max_pcp, new_pcp_val)
            cand_mean_pcp = cand_sum_pcp / n_sel
        else:
            cand_max_pcp = 0
            cand_mean_pcp = 0.0

        cand_score = (new_min_hd, new_min_ld, new_max_cp, cand_max_pcp,
                      cand_sum_hd / n_pairs, cand_sum_ld / n_pairs,
                      cand_sum_cp / n_pairs, cand_mean_pcp)
        return cand_score, score_to_scalar(cand_score)

    current_score = get_score()
    current_scalar = score_to_scalar(current_score)
    best_score = current_score
    best_scalar = current_scalar
    best_selected = list(selected)

    accepted = 0
    improved = 0
    lns_count = 0
    lns_improved = 0
    last_improve_it = 0
    log_interval = max(1, iterations // 20)

    print(f"    Initial: {fmt_score(current_score)} (scalar={current_scalar:.1f})")

    for it in range(iterations):
        # Early termination
        if it - last_improve_it > max_no_improve:
            print(f"    Early stop at iter {it}: no improvement for {max_no_improve} iterations")
            break

        # Periodic refresh of approx_min_hd
        if it > 0 and it % hd_refresh_interval == 0:
            refresh_approx_min_hd()

        # --- LNS Phase ---
        if it > 0 and it % lns_interval == 0:
            lns_count += 1

            # Save state
            saved_selected = list(selected)
            saved_selected_np = selected_np.copy()
            saved_selected_set = set(selected_set)
            saved_hd_arr = hd_arr.copy()
            saved_ld_arr = ld_arr.copy()
            saved_cp_arr = cp_arr.copy()
            saved_sum_hd = sum_hd
            saved_sum_ld = sum_ld
            saved_sum_cp = sum_cp
            saved_sel_primer_cp = sel_primer_cp.copy()
            saved_sum_primer_cp = sum_primer_cp
            saved_approx_min_hd = approx_min_hd.copy()
            saved_scalar = current_scalar

            # Get bottleneck positions for destruction
            bottleneck_pos = find_bottleneck_positions()
            destroy_positions = list(bottleneck_pos[:lns_destroy_k])
            # Pad with random positions if needed
            if len(destroy_positions) < lns_destroy_k:
                remaining = [p for p in range(n_sel) if p not in destroy_positions]
                rng.shuffle(remaining)
                need = lns_destroy_k - len(destroy_positions)
                destroy_positions.extend(remaining[:need])

            # Sequentially replace each destroyed position with best candidate
            for d_pos in destroy_positions:
                screened = get_screened_candidates(n_eval * 3)
                if len(screened) == 0:
                    continue
                if len(screened) > n_eval * 3:
                    screened = np_rng.choice(screened, size=n_eval * 3, replace=False)

                # Pre-compute position-dependent values ONCE
                not_pos_mask = pos_not_pair_mask[d_pos]
                non_pos_min_hd_v = int(np.min(hd_arr[not_pos_mask])) if n_pairs > n_sel - 1 else 999
                non_pos_min_ld_v = int(np.min(ld_arr[not_pos_mask])) if n_pairs > n_sel - 1 else 999
                non_pos_max_cp_v = int(np.max(cp_arr[not_pos_mask])) if n_pairs > n_sel - 1 else 0
                d_pair_indices = pos_pair_indices[d_pos]
                old_sum_hd_v = int(np.sum(hd_arr[d_pair_indices]))
                old_sum_ld_v = int(np.sum(ld_arr[d_pair_indices]))
                old_sum_cp_v = int(np.sum(cp_arr[d_pair_indices]))

                best_ci = -1
                best_ci_scalar = -1e18
                best_ci_hd_vals = None
                best_ci_ld_vals = None
                best_ci_cp_vals = None
                best_ci_score = None

                for ci in screened:
                    ci = int(ci)
                    if ci in selected_set:
                        continue
                    new_seq_arr = encoded_nb[ci]
                    hd_v, ld_v, cp_v = _sa_compute_new_pairs(
                        encoded_nb, new_seq_arr, selected_np, d_pos, n_sel, _COMP_TABLE)
                    cand_score, cand_scalar = compute_score_for_swap(
                        d_pos, hd_v, ld_v, cp_v,
                        non_pos_min_hd_v, non_pos_min_ld_v, non_pos_max_cp_v,
                        old_sum_hd_v, old_sum_ld_v, old_sum_cp_v, ci)
                    if cand_scalar > best_ci_scalar:
                        best_ci_scalar = cand_scalar
                        best_ci = ci
                        best_ci_hd_vals = hd_v
                        best_ci_ld_vals = ld_v
                        best_ci_cp_vals = cp_v
                        best_ci_score = cand_score

                if best_ci >= 0:
                    old_cand = apply_swap(d_pos, best_ci, best_ci_hd_vals, best_ci_ld_vals, best_ci_cp_vals)
                    update_approx_after_swap(old_cand, best_ci)
                    current_score = best_ci_score
                    current_scalar = best_ci_scalar

            # Check if LNS improved
            current_score = get_score()
            current_scalar = score_to_scalar(current_score)

            if current_scalar > saved_scalar:
                # Keep the LNS result
                lns_improved += 1
                accepted += 1
                if current_scalar > best_scalar:
                    best_score = current_score
                    best_scalar = current_scalar
                    best_selected = list(selected)
                    improved += 1
                    last_improve_it = it
            else:
                # Restore state
                selected[:] = saved_selected
                selected_np[:] = saved_selected_np
                selected_set.clear()
                selected_set.update(saved_selected_set)
                hd_arr[:] = saved_hd_arr
                ld_arr[:] = saved_ld_arr
                cp_arr[:] = saved_cp_arr
                sum_hd = saved_sum_hd
                sum_ld = saved_sum_ld
                sum_cp = saved_sum_cp
                sel_primer_cp[:] = saved_sel_primer_cp
                sum_primer_cp = saved_sum_primer_cp
                approx_min_hd[:] = saved_approx_min_hd
                current_score = get_score()
                current_scalar = saved_scalar

            if (it + 1) % log_interval == 0 or it == 0:
                print(f"    iter {it+1:>6}/{iterations}: "
                      f"cur=({fmt_score(current_score)})  "
                      f"best=scalar={best_scalar:.1f}  "
                      f"accepted={accepted} improved={improved} "
                      f"LNS={lns_count}/{lns_improved}")
            continue

        # --- Regular Tabu Swap ---
        bottleneck_positions = find_bottleneck_positions()

        if rng.random() < 0.7 and bottleneck_positions:
            pos = rng.choice(bottleneck_positions)
        else:
            pos = rng.randint(0, n_sel - 1)

        # Pre-compute position-dependent values ONCE per position
        not_pos_mask = pos_not_pair_mask[pos]
        non_pos_min_hd = int(np.min(hd_arr[not_pos_mask])) if n_pairs > n_sel - 1 else 999
        non_pos_min_ld = int(np.min(ld_arr[not_pos_mask])) if n_pairs > n_sel - 1 else 999
        non_pos_max_cp = int(np.max(cp_arr[not_pos_mask])) if n_pairs > n_sel - 1 else 0
        pair_indices = pos_pair_indices[pos]
        old_sum_hd_pos = int(np.sum(hd_arr[pair_indices]))
        old_sum_ld_pos = int(np.sum(ld_arr[pair_indices]))
        old_sum_cp_pos = int(np.sum(cp_arr[pair_indices]))

        # Screen candidates
        screened = get_screened_candidates(n_eval * 2)
        if len(screened) == 0:
            continue
        if len(screened) > n_eval:
            screened = np_rng.choice(screened, size=n_eval, replace=False)

        best_ci = -1
        best_ci_scalar = -1e18
        best_ci_hd_vals = None
        best_ci_ld_vals = None
        best_ci_cp_vals = None
        best_ci_score = None

        for ci in screened:
            ci = int(ci)
            if ci in selected_set:
                continue

            is_tabu = ci in tabu and tabu[ci] > it
            new_seq_arr = encoded_nb[ci]
            hd_v, ld_v, cp_v = _sa_compute_new_pairs(
                encoded_nb, new_seq_arr, selected_np, pos, n_sel, _COMP_TABLE)
            cand_score, cand_scalar = compute_score_for_swap(
                pos, hd_v, ld_v, cp_v,
                non_pos_min_hd, non_pos_min_ld, non_pos_max_cp,
                old_sum_hd_pos, old_sum_ld_pos, old_sum_cp_pos, ci)

            # Accept if non-tabu OR aspiration criterion (better than best known)
            if is_tabu and cand_scalar <= best_scalar:
                continue

            if cand_scalar > best_ci_scalar:
                best_ci_scalar = cand_scalar
                best_ci = ci
                best_ci_hd_vals = hd_v
                best_ci_ld_vals = ld_v
                best_ci_cp_vals = cp_v
                best_ci_score = cand_score

        if best_ci < 0:
            continue

        # Always accept best move (tabu search)
        old_cand_idx = apply_swap(pos, best_ci, best_ci_hd_vals, best_ci_ld_vals, best_ci_cp_vals)

        # Add old barcode to tabu
        tabu[old_cand_idx] = it + tabu_tenure

        # Update approx_min_hd
        update_approx_after_swap(old_cand_idx, best_ci)

        current_score = best_ci_score
        current_scalar = best_ci_scalar
        accepted += 1

        if current_scalar > best_scalar:
            best_score = current_score
            best_scalar = current_scalar
            best_selected = list(selected)
            improved += 1
            last_improve_it = it

        if (it + 1) % log_interval == 0 or it == 0:
            print(f"    iter {it+1:>6}/{iterations}: "
                  f"cur=({fmt_score(current_score)})  "
                  f"best=scalar={best_scalar:.1f}  "
                  f"accepted={accepted} improved={improved} "
                  f"LNS={lns_count}/{lns_improved}")

    print(f"\n    Tabu-LNS complete: {accepted} accepted, {improved} improvements, "
          f"{lns_count} LNS phases ({lns_improved} improved)")
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
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Tabu-LNS search iterations (default: 10000)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--primers', type=str, default=None,
                        help='Primer sequences to avoid (comma-separated DNA sequences). '
                             'Barcodes with high complementarity to these primers will be penalized.')
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

    # Parse primer sequences
    primers = []
    if args.primers:
        for p in args.primers.split(','):
            p = p.strip().upper()
            if p and all(c in 'ACGT' for c in p):
                primers.append(p)
            elif p:
                print(f"WARNING: Ignoring invalid primer sequence '{p}' (must contain only A, C, G, T)")
        if not primers:
            print("WARNING: No valid primer sequences provided, ignoring --primers")

    print(f"=== Barcode Optimizer ===")
    print(f"  Barcode length:    {args.length} nt")
    print(f"  Target count:      {args.num_barcodes}")
    print(f"  GC range:          {gc_min*100:.1f}% - {gc_max*100:.1f}%")
    print(f"  Max homopolymer:   {args.max_homopolymer}")
    print(f"  Max dinuc repeat:  {args.max_dinuc_repeat}")
    if primers:
        print(f"  Primer blacklist:  {len(primers)} sequences")
        for p in primers:
            print(f"    {p} ({len(p)} nt)")
    print(f"  Greedy restarts:   {args.restarts}")
    print(f"  Tabu-LNS iters:    {args.iterations}")
    print()

    # Warm up numba JIT (first call triggers compilation)
    print("JIT compiling numba kernels...", end=" ", flush=True)
    t_jit = time.time()
    _dummy = np.array([0, 1, 2, 3], dtype=np.int8)
    hamming_nb(_dummy, _dummy)
    levenshtein_nb(_dummy, _dummy)
    max_complementarity_nb(_dummy, _dummy, _COMP_TABLE)
    _enc2 = np.array([[0,1,2,3],[1,2,3,0]], dtype=np.int8)
    _batch_levenshtein_update(_enc2, _enc2[0], np.full(2, 999, dtype=np.int16), np.zeros(2, dtype=np.bool_), 2)
    _batch_complementarity_update(_enc2, _enc2[0], np.zeros(2, dtype=np.int16), np.zeros(2, dtype=np.bool_), 2, _COMP_TABLE)
    _sa_compute_new_pairs(_enc2, _enc2[0], np.array([0, 1], dtype=np.int32), 0, 2, _COMP_TABLE)
    _evaluate_set_full_nb(_enc2, np.array([0, 1], dtype=np.int32), 2, _COMP_TABLE)
    _compute_all_pairs_nb(_enc2, np.array([0, 1], dtype=np.int32), 2, _COMP_TABLE)
    _batch_primer_complementarity(_enc2, _enc2[:1], np.array([_enc2.shape[1]], dtype=np.int32), _COMP_TABLE, 2, 1)
    _generate_candidates_nb(4, 1, 3, 2, 2)  # Warm up candidate generator
    print(f"done in {time.time()-t_jit:.1f}s")

    # Step 1: Generate candidate pool (returns both strings and encoded array)
    print("Generating candidate barcodes...", end=" ", flush=True)
    t0 = time.time()
    candidates, encoded_nb = generate_candidates(
        length=args.length,
        gc_min=gc_min,
        gc_max=gc_max,
        max_homopolymer=args.max_homopolymer,
        max_dinuc_repeat=args.max_dinuc_repeat,
    )
    encoded = encoded_nb  # Same array works for both vectorized and numba operations
    t1 = time.time()
    print(f"{len(candidates)} candidates in {t1-t0:.1f}s")

    if len(candidates) < args.num_barcodes:
        print(f"ERROR: Only {len(candidates)} candidates available, cannot select {args.num_barcodes}.")
        sys.exit(1)

    # Precompute primer complementarity for all candidates (using numba)
    primer_cp_arr = None
    if primers:
        print("Computing primer complementarity for all candidates...", end=" ", flush=True)
        t_pcp = time.time()
        primer_raw = [seq_to_arr(p) for p in primers]
        max_plen = max(len(a) for a in primer_raw)
        primer_arrs = np.zeros((len(primer_raw), max_plen), dtype=np.int8)
        primer_lens = np.array([len(a) for a in primer_raw], dtype=np.int32)
        for i, a in enumerate(primer_raw):
            primer_arrs[i, :len(a)] = a
        primer_cp_arr = _batch_primer_complementarity(encoded_nb, primer_arrs, primer_lens, _COMP_TABLE, len(candidates), len(primers))
        print(f"done in {time.time()-t_pcp:.1f}s (max={int(np.max(primer_cp_arr))}, mean={np.mean(primer_cp_arr):.1f})")

    # Step 3: Greedy initialization
    print(f"\nPhase 1: Greedy initialization ({args.restarts} restarts)...")
    base_seed = args.seed if args.seed is not None else random.randint(0, 999999)

    best_indices = None
    best_score = (-1, -1, 999, 999, 0.0, 0.0, 999.0, 999.0)

    def fmt_main_score(s):
        base = f"HD>={s[0]} LD>={s[1]} CP<={s[2]}"
        if s[3] > 0:
            base += f" PCP<={s[3]}"
        base += f" avgHD={s[4]:.2f} avgLD={s[5]:.2f} avgCP={s[6]:.2f}"
        if s[7] > 0:
            base += f" avgPCP={s[7]:.2f}"
        return base

    def _run_greedy_restart(r):
        seed = base_seed + r
        indices, _ = greedy_select(candidates, encoded, args.num_barcodes, seed=seed, primer_cp=primer_cp_arr, encoded_nb=encoded_nb)
        score = evaluate_set_full(candidates, indices, primer_cp=primer_cp_arr, encoded_nb=encoded_nb)
        return r, indices, score

    for r in range(args.restarts):
        r_num, indices, score = _run_greedy_restart(r)
        improved = ""
        if score_is_better(score, best_score):
            best_score = score
            best_indices = indices
            improved = " << NEW BEST"
        print(f"  Restart {r_num+1:>3}/{args.restarts}: {fmt_main_score(score)}{improved}")

    print(f"  Greedy best: {fmt_main_score(best_score)}")

    # Step 4: Tabu-LNS refinement
    print(f"\nPhase 2: Tabu-LNS search ({args.iterations} iterations)...")
    tabu_seed = base_seed + args.restarts
    refined_indices, refined_score = tabu_lns_search(
        candidates, best_indices, encoded,
        iterations=args.iterations,
        seed=tabu_seed,
        primer_cp=primer_cp_arr,
        encoded_nb=encoded_nb,
    )
    print(f"\n  Final: {fmt_main_score(refined_score)}")

    # Step 5: Collect and display results
    selected = [candidates[i] for i in refined_indices]

    # Compute full stats: Hamming, Levenshtein, and Complementarity
    from itertools import combinations
    from collections import Counter

    print("\nComputing pairwise Hamming & Levenshtein distances...", end=" ", flush=True)
    t_stats = time.time()
    refined_indices_arr = np.array(refined_indices, dtype=np.int32)
    hd_flat, ld_flat, cp_flat = _compute_all_pairs_nb(
        encoded_nb, refined_indices_arr, len(refined_indices), _COMP_TABLE)
    all_hd = hd_flat.tolist()
    all_ld = ld_flat.tolist()
    all_cp = cp_flat.tolist()
    # Build pair_data for flagging
    pair_data = []
    k = 0
    n_final = len(selected)
    for i in range(n_final):
        for j in range(i + 1, n_final):
            pair_data.append((i+1, j+1, selected[i], selected[j],
                            all_hd[k], all_ld[k], all_cp[k]))
            k += 1
    print(f"done in {time.time()-t_stats:.1f}s")

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

    # --- Primer Complementarity ---
    if primers:
        print(f"\n  PRIMER COMPLEMENTARITY (non-specific binding risk)")
        pcp_vals = [int(primer_cp_arr[i]) for i in refined_indices]
        print(f"  Max primer complementarity: {max(pcp_vals)} bp")
        print(f"  Mean primer complementarity: {sum(pcp_vals)/len(pcp_vals):.2f} bp")
        pcp_counter = Counter(pcp_vals)
        print(f"  Distribution:")
        for pcp_val in sorted(pcp_counter.keys()):
            print(f"    PCP={pcp_val}: {pcp_counter[pcp_val]:>5} barcodes")
        # Flag high primer complementarity barcodes
        pcp_threshold = max(4, int(min(len(p) for p in primers) * 0.6))
        high_pcp = [(idx+1, s, pcp) for idx, (s, pcp) in enumerate(zip(selected, pcp_vals)) if pcp >= pcp_threshold]
        if high_pcp:
            print(f"\n  WARNING: {len(high_pcp)} barcodes with primer complementarity >= {pcp_threshold} bp:")
            print(f"  {'BC':<5} {'Barcode':<10} {'PCP':>4}")
            print(f"  {'-'*22}")
            for bc_idx, seq, pcp in sorted(high_pcp, key=lambda x: -x[2])[:20]:
                print(f"  {bc_idx:<5} {seq:<10} {pcp:>4}")
        else:
            print(f"\n  No barcodes with primer complementarity >= {pcp_threshold} bp. Low non-specific binding risk.")

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
