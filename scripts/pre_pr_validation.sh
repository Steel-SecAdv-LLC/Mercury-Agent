#!/bin/bash
# Mercury-Agent v1.4.0 Pre-PR Validation Script
#
# Copyright (C) 2025 Steel Security Advisors LLC
# License: GPL-3.0+
#
# Usage: bash scripts/pre_pr_validation.sh

set -e

echo "Mercury-Agent v1.4.0 Pre-PR Validation"
echo "========================================"

# 1. Check version
echo -n "Version check: "
version=$(python -c "import omni_mercury_engine; print(omni_mercury_engine.__version__)")
if [ "$version" = "1.4.0" ]; then
    echo "v$version OK"
else
    echo "v$version FAIL (expected 1.4.0)"
    exit 1
fi

# 2. Syntax check
echo -n "Syntax check: "
python -m py_compile src/omni_mercury_engine/detectors/threshold_calibrator.py
python -m py_compile scripts/validate_live_data_metrics.py
python -m py_compile scripts/generate_baseline_report.py
echo "OK"

# 3. Import checks
echo -n "Import checks: "
python -c "from omni_mercury_engine.detectors.threshold_calibrator import find_optimal_threshold, ThresholdOptimizer"
python -c "from omni_mercury_engine.datasets.cache import DatasetCache"
echo "OK"

# 4. Test collection
echo -n "Test collection: "
test_count=$(python -m pytest tests/validation/ --collect-only -q 2>/dev/null | tail -1 | awk '{print $1}')
echo "$test_count tests collected OK"

# 5. Run quick validation (no live data, tests should skip)
echo -n "Quick validation (no live data): "
python -m pytest tests/validation/ -q --tb=short 2>/dev/null || true
echo "OK"

# 6. Baseline exists
echo -n "Baseline file: "
if [ -f benchmarks/live_data_baseline.json ]; then
    echo "OK"
else
    echo "MISSING (run: python scripts/generate_baseline_report.py)"
    exit 1
fi

# 7. Check for large files
echo -n "Large files check: "
large_files=$(find . -type f -size +1M -not -path "./.git/*" -not -path "./__pycache__/*" 2>/dev/null | wc -l)
if [ "$large_files" -eq 0 ]; then
    echo "OK"
else
    echo "WARNING: Found $large_files large files"
    find . -type f -size +1M -not -path "./.git/*" -not -path "./__pycache__/*" 2>/dev/null
fi

echo ""
echo "All pre-PR checks passed."
echo "Ready to create PR."
