# Hamboning

Generates near-optimal sets of barcodes of arbitrary length and arbitrary number of barcodes.

## Optimization Objectives

Maximizes the minimum pairwise Hamming distance and Levenshtein distance, while minimizing the maximum complementarity between all pairs of barcodes. Also optimizes the mean Hamming distance, Levenshtein distance, and complementarity across all pairs.

Additional constraints (configurable):
- GC content within 40-60%
- No homopolymer runs >= 3
- No dinucleotide repeats >= 3
- Optional primer complementarity minimization

## How It Works

1. **Candidate generation**: Enumerates all valid barcodes meeting GC, homopolymer, and dinucleotide constraints.
2. **Greedy initialization**: Multiple random restarts of a greedy algorithm that selects barcodes maximizing minimum pairwise distance.
3. **Tabu-LNS refinement**: A Tabu Search with Large Neighborhood Search optimizer that refines the greedy solution:
   - Evaluates multiple HD-screened candidates per swap (best-move strategy)
   - Tabu list prevents cycling without temperature tuning
   - Periodic LNS phases destroy bottleneck barcodes and greedily rebuild, escaping local optima
   - Early termination when converged

Core distance functions (Hamming, Levenshtein, complementarity) are JIT-compiled with Numba for near-C performance.

## Requirements

- Python 3.7+
- NumPy
- Numba

## Usage

```bash
python optimize_barcodes.py --length 8 --num-barcodes 96 --output optimized_barcodes.csv --restarts 3 --iterations 10000
```

### Key Arguments

- `--length` or `-l`: Length of each barcode (default: 8)
- `--num-barcodes` or `-n`: Number of barcodes to generate (default: 96)
- `--output` or `-o`: Output file for barcode list (CSV)
- `--restarts`: Number of random restarts for greedy initialization (default: 3)
- `--iterations`: Number of Tabu-LNS search iterations (default: 10000)
- `--strict`: Strict GC filter - only barcodes with GC count in [ceil(L*0.4), floor(L*0.6)]
- `--max-homopolymer`: Maximum allowed homopolymer run length (default: 2)
- `--max-dinuc-repeat`: Maximum allowed dinucleotide repeat count (default: 2)
- `--primers`: Primer sequences to avoid (comma-separated). Penalizes barcodes with high complementarity to these primers.
- `--seed`: Random seed for reproducibility

## Output

CSV file with two columns:
- `index`: Barcode number (1-indexed)
- `barcode`: Optimized barcode sequence
