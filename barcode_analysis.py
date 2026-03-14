import csv
import sys
from itertools import combinations
from collections import Counter

# Load barcodes from CSV file (output from optimize_barcodes.py)
if len(sys.argv) > 1:
    csv_file = sys.argv[1]
else:
    csv_file = 'optimized_barcodes.csv'

print(f"Reading barcodes from: {csv_file}\n")

barcodes = {}
try:
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = row['index']
            barcode = row['barcode']
            bc_name = f"BC{index}"
            barcodes[bc_name] = {'barcode': barcode}
except FileNotFoundError:
    print(f"ERROR: File '{csv_file}' not found.")
    print(f"Usage: python barcode_analysis.py [csv_file]")
    sys.exit(1)
except KeyError as e:
    print(f"ERROR: CSV file missing expected column: {e}")
    print(f"Expected columns: 'index', 'barcode'")
    sys.exit(1)

print(f"Loaded {len(barcodes)} barcodes\n")

# --- Show all barcodes ---
print(f"{'Name':<8} {'Barcode':<12} {'Len':>3}  {'GC%':>5}")
print("-" * 35)
for name, info in barcodes.items():
    bc = info['barcode']
    gc = (bc.count('G') + bc.count('C')) / len(bc) * 100
    print(f"{name:<8} {bc:<12} {len(bc):>3}  {gc:>5.1f}")

print()
flagged_gc = []
for name, info in barcodes.items():
    bc = info['barcode']
    gc = (bc.count('G') + bc.count('C')) / len(bc) * 100
    status = ""
    if gc < 25 or gc > 75:
        status = "** POOR **"
        flagged_gc.append((name, bc, gc, "POOR"))
    elif gc < 37.5 or gc > 62.5:
        status = "* MARGINAL *"
        flagged_gc.append((name, bc, gc, "MARGINAL"))

if flagged_gc:
    print(f"{'Name':<8} {'Barcode':<12} {'GC%':>5}  {'Status'}")
    print("-" * 40)
    for name, bc, gc, status in flagged_gc:
        print(f"{name:<8} {bc:<12} {gc:>5.1f}  {status}")
else:
    print("All barcodes have acceptable GC content (40-60%).")

gc_values = []
for info in barcodes.values():
    bc = info['barcode']
    gc_values.append((bc.count('G') + bc.count('C')) / len(bc) * 100)
print(f"\nGC content range: {min(gc_values):.1f}% - {max(gc_values):.1f}%")
print(f"Mean GC content: {sum(gc_values)/len(gc_values):.1f}%")

# --- Hamming Distance Analysis ---
print("\n" + "=" * 60)
print("HAMMING DISTANCE ANALYSIS")
print("=" * 60)

def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        return None  # Can't compute for different lengths
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

bc_names = list(barcodes.keys())
bc_seqs = [barcodes[n]['barcode'] for n in bc_names]

# Check if all barcodes are the same length
lengths = set(len(s) for s in bc_seqs)
print(f"Barcode lengths present: {lengths}")

# Compute all pairwise metrics: Hamming, Levenshtein, Complementarity
all_hd = []
all_ld = []
all_cp = []
pair_data = []  # (name1, name2, seq1, seq2, hd, ld, cp)
low_hd_pairs = []
min_hd = float('inf')
min_ld = float('inf')
max_cp = 0

for i, j in combinations(range(len(bc_names)), 2):
    s1, s2 = bc_seqs[i], bc_seqs[j]
    if len(s1) != len(s2):
        continue
    hd = hamming_distance(s1, s2)
    ld = levenshtein(s1, s2)
    cp = max_complementarity(s1, s2)
    
    all_hd.append(hd)
    all_ld.append(ld)
    all_cp.append(cp)
    pair_data.append((bc_names[i], bc_names[j], s1, s2, hd, ld, cp))
    
    if hd < min_hd:
        min_hd = hd
    if ld < min_ld:
        min_ld = ld
    if cp > max_cp:
        max_cp = cp
    
    if hd < 3:  # Flag pairs with HD < 3 (risk of barcode misassignment)
        low_hd_pairs.append((bc_names[i], bc_names[j], s1, s2, hd))

print(f"\nTotal pairwise comparisons: {len(all_hd)}")
if all_hd:
    print(f"Minimum Hamming distance: {min_hd}")
    print(f"Maximum Hamming distance: {max(all_hd)}")
    print(f"Mean Hamming distance: {sum(all_hd)/len(all_hd):.2f}")

    # Distribution
    hd_counter = Counter(all_hd)
    print(f"\nHamming distance distribution:")
    for hd_val in sorted(hd_counter.keys()):
        print(f"  HD={hd_val}: {hd_counter[hd_val]:>5} pairs")

    # Flag problematic pairs
    if low_hd_pairs:
        print(f"  {'HD':<4} {'BC1':<8} {'Seq1':<10} {'BC2':<8} {'Seq2':<10}")
        print(f"  {'-'*46}")
        for n1, n2, s1, s2, hd in sorted(low_hd_pairs, key=lambda x: x[4]):
            diff = ''.join('^' if a != b else ' ' for a, b in zip(s1, s2))
            print(f"  {hd:<4} {n1:<8} {s1:<10} {n2:<8} {s2:<10}")
            print(f"       {'':8} {diff}")
    else:
        print("  None found - all pairs have HD >= 3.")

# --- Levenshtein Distance Analysis ---
print("\n" + "=" * 60)
print("LEVENSHTEIN (EDIT) DISTANCE ANALYSIS")
print("=" * 60)
print(f"Minimum Levenshtein distance: {min_ld}")
print(f"Maximum Levenshtein distance: {max(all_ld)}")
print(f"Mean Levenshtein distance: {sum(all_ld)/len(all_ld):.2f}")

ld_counter = Counter(all_ld)
print(f"\nLevenshtein distance distribution:")
for ld_val in sorted(ld_counter.keys()):
    print(f"  LD={ld_val}: {ld_counter[ld_val]:>5} pairs")

# Flag low LD pairs
low_ld_pairs = [(n1, n2, s1, s2, ld) for n1, n2, s1, s2, hd, ld, cp in pair_data if ld < 3]
if low_ld_pairs:
    print(f"\n--- Pairs with Levenshtein distance < 3 (indel sensitivity) ---")
    print(f"  {'LD':<4} {'BC1':<8} {'Seq1':<10} {'BC2':<8} {'Seq2':<10}")
    print(f"  {'-'*46}")
    for n1, n2, s1, s2, ld in sorted(low_ld_pairs, key=lambda x: x[4])[:20]:
        print(f"  {ld:<4} {n1:<8} {s1:<10} {n2:<8} {s2:<10}")
else:
    print(f"\nAll pairs have Levenshtein distance >= 3.")

# --- Sequence Complementarity Analysis ---
print("\n" + "=" * 60)
print("SEQUENCE COMPLEMENTARITY (DIMER RISK) ANALYSIS")
print("=" * 60)
print(f"Maximum complementarity: {max_cp} bp")
print(f"Mean complementarity: {sum(all_cp)/len(all_cp):.2f} bp")

cp_counter = Counter(all_cp)
print(f"\nComplementarity distribution:")
for cp_val in sorted(cp_counter.keys()):
    print(f"  CP={cp_val}: {cp_counter[cp_val]:>5} pairs")

# Flag high complementarity pairs (>= 6 bp for 8-mers = 75% complementary)
bc_length = len(bc_seqs[0]) if bc_seqs else 8
cp_threshold = max(6, int(bc_length * 0.75))
high_cp_pairs = [(n1, n2, s1, s2, cp) for n1, n2, s1, s2, hd, ld, cp in pair_data if cp >= cp_threshold]
if high_cp_pairs:
    print(f"\n--- Pairs with complementarity >= {cp_threshold} bp (HIGH dimer risk) ---")
    print(f"  {'BC1':<8} {'Seq1':<10} {'BC2':<8} {'Seq2':<10} {'CP':>3}  {'RevComp(Seq2)':<10}")
    print(f"  {'-'*58}")
    for n1, n2, s1, s2, cp in sorted(high_cp_pairs, key=lambda x: -x[4])[:20]:
        rc2 = reverse_complement(s2)
        print(f"  {n1:<8} {s1:<10} {n2:<8} {s2:<10} {cp:>3}  {rc2:<10}")
else:
    print(f"\nNo pairs with complementarity >= 75% of BC length ({cp_threshold} .")


print("\n" + "=" * 60)
