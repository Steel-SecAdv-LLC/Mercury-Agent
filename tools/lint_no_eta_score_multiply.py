#!/usr/bin/env python3
"""X12a gate-hardening lint: forbid multiplying a fusion/score value by any power
of the ethical multiplier eta.

This makes the calibration-corrupting bug class (Brief V6 / V12d --
``fusion_score = weighted_sum * eta ** exponent``) STATICALLY UNREPRESENTABLE:
the ethical signal must be a hard fail-closed indicator beside sigma_Immutable,
never a soft multiplicative dial folded into the score path (Brief R4 -- the
ethics gate may be HARDENED, never weighted into any loss/score).

Rule (AST): flag any ``a * b`` where one operand is, directly or via a local
assignment, a power ``<eta-like> ** _`` and (optionally, --strict off) the other
operand is a score-like name.  With ``--any-eta-power`` ANY multiplication by an
eta power is flagged regardless of the other operand.

Usage:
  python tools/lint_no_eta_score_multiply.py <file.py> [<file.py> ...]
  python tools/lint_no_eta_score_multiply.py --selftest
Exit code 1 if any violation is found.
"""

from __future__ import annotations

import ast
import sys

ETA_NAMES = {"eta", "eta_ethical", "ethical_scaling", "ethical_multiplier", "eta_value"}
ETA_HINTS = ("eta", "ethical")
SCORE_HINTS = ("score", "fusion", "weighted_sum", "weighted", "blend", "anomaly")


def _is_eta_name(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        n = node.id.lower()
        return node.id in ETA_NAMES or n == "eta" or n.startswith("eta_")
    if isinstance(node, ast.Attribute):
        return any(h in node.attr.lower() for h in ETA_HINTS)
    return False


def _is_eta_power(node: ast.AST, tainted: set[str]) -> bool:
    """True if node is `<eta-like> ** _` or a local name assigned such a power."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return _is_eta_name(node.left)
    if isinstance(node, ast.Name) and node.id in tainted:
        return True
    return False


def _is_score_like(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return any(h in node.id.lower() for h in SCORE_HINTS)
    if isinstance(node, ast.Attribute):
        return any(h in node.attr.lower() for h in SCORE_HINTS)
    return False


class EtaMultVisitor(ast.NodeVisitor):
    def __init__(self, any_eta_power: bool):
        self.any_eta_power = any_eta_power
        self.tainted: set[str] = set()
        self.violations: list[tuple[int, str]] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        # taint: name = <eta-like> ** _
        if (isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Pow)
                and _is_eta_name(node.value.left)):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    self.tainted.add(tgt.id)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Mult):
            l, r = node.left, node.right
            for a, b in ((l, r), (r, l)):
                if _is_eta_power(a, self.tainted) and (self.any_eta_power or _is_score_like(b)):
                    self.violations.append(
                        (node.lineno, ast.dump(node)[:90]))
                    break
        self.generic_visit(node)


def lint_source(src: str, any_eta_power: bool = False) -> list[tuple[int, str]]:
    v = EtaMultVisitor(any_eta_power)
    v.visit(ast.parse(src))
    return v.violations


def lint_file(path: str, any_eta_power: bool = False) -> list[tuple[int, str]]:
    with open(path, encoding="utf-8") as f:
        return lint_source(f.read(), any_eta_power)


GOOD = "fusion_score = w_R * r + w_H * h + w_O * o\nveto = (eta >= eta_star) and sigma_ok\n"
BAD1 = "ethical_scaling = eta ** self.ethical_exponent\nfusion_score = weighted_sum * ethical_scaling\n"
BAD2 = "fusion_score = weighted_sum * (eta ** 1.618)\n"


def _selftest() -> int:
    ok = True
    if lint_source(GOOD):
        print("SELFTEST FAIL: false positive on GOOD")
        ok = False
    if not lint_source(BAD1):
        print("SELFTEST FAIL: missed BAD1 (taint via assignment)")
        ok = False
    if not lint_source(BAD2):
        print("SELFTEST FAIL: missed BAD2 (inline eta power)")
        ok = False
    print("SELFTEST PASS" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    any_eta = "--any-eta-power" in argv
    files = [a for a in argv if not a.startswith("-")]
    total = 0
    for path in files:
        for lineno, snippet in lint_file(path, any_eta):
            print(f"{path}:{lineno}: VIOLATION score*(eta**_) -> {snippet}")
            total += 1
    if total:
        print(f"\n{total} violation(s): the ethical multiplier must be a hard veto, "
              f"not a score multiplier (Brief R4/V6).")
        return 1
    print("OK: no score*(eta**power) multiplications found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
