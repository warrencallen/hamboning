# Hamboning

Generates near-optimal sets of barcodes of arbitrary length and arbitrary number of barcodes.

## Optimization Objectives

Maximizes the minimum pairwise Hamming distance and Levenshtein distance, while minimizing the maximum complementarity between all pairs of barcodes. Also optimizes the mean Hamming distance, Levenshtein distance, and complementarity across all pairs.

Additional constraints (configurable):
- GC content within 40-60%
- No homopolymer runs ≥ 3
- No dinucleotide repeats ≥ 3

## Requirements

- Python 3.7+
- NumPy

## Usage

```bash
python optimize_barcodes.py --length 8 --num-barcodes 50 --output optimized_barcodes.csv --restarts 20 --strict --sa-iterations 50000
```

### Key Arguments

- `--length` or `-l`: Length of each barcode (default: 8)
- `--num-barcodes` or `-n`: Number of barcodes to generate (default: 96)
- `--output` or `-o`: Output file for barcode list (CSV)
- `--restarts`: Number of random restarts for greedy initialization (default: 3)
- `--strict`: Strict GC filter - only barcodes with GC count in [ceil(L×0.4), floor(L×0.6)]
- `--max-homopolymer`: Maximum allowed homopolymer run length (default: 2)
- `--max-dinuc-repeat`: Maximum allowed dinucleotide repeat count (default: 2)
- `--sa-iterations`: Number of iterations for simulated annealing (default: 50000)
- `--sa-temp`: SA starting temperature (default: 5.0)
- `--sa-cool`: SA ending temperature (default: 0.01)
- `--seed`: Random seed for reproducibility

## Output

CSV file with two columns:
- `index`: Barcode number (1-indexed)
- `barcode`: Optimized barcode sequence

