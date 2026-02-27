#!/usr/bin/env bash
# Opens the three sklearn-removal tracking issues on GitHub.
# Requires: gh CLI authenticated with repo write scope.
#
# Usage:  bash .github/pending_issues/open_issues.sh
set -euo pipefail

REPO="Steel-SecAdv-LLC/Mercury-Agent"

echo "Creating sklearn-removal tracking issues for $REPO ..."

gh issue create --repo "$REPO" \
  --title "Replace sklearn dependencies in ml/online_learning.py" \
  --label "tech-debt" --label "sklearn-removal" --label "non-blocking" \
  --body-file .github/pending_issues/sklearn_online_learning.md

gh issue create --repo "$REPO" \
  --title "Replace sklearn dependencies in ml/cross_domain_transfer.py" \
  --label "tech-debt" --label "sklearn-removal" --label "non-blocking" \
  --body-file .github/pending_issues/sklearn_cross_domain_transfer.md

gh issue create --repo "$REPO" \
  --title "Replace sklearn dependencies in ml/concept_drift_evaluation.py" \
  --label "tech-debt" --label "sklearn-removal" --label "non-blocking" \
  --body-file .github/pending_issues/sklearn_concept_drift_evaluation.md

echo "Done. All three issues created."
