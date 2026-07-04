# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""A tiny propositional formula language routed through the real DPLL oracle.

The verifier loop needs to adjudicate natural-ish propositional claims -- "P or
not P is a tautology", "(P implies Q) and P implies Q is a tautology" -- without
inventing a second SAT engine. This module supplies exactly the missing glue:

* :func:`parse` turns a formula string into an AST over ``{~, and, or, ->, <->,
  xor}`` (words *or* symbols), with the usual precedence.
* :func:`to_cnf` Tseitin-transforms a normalized AST into CNF over
  :class:`omni_mercury_engine.verifiers.propositional.Literal`, and
  :func:`is_tautology` / :func:`is_satisfiable_formula` decide it with the
  package's own DPLL solver (``verifiers.propositional.is_satisfiable``) -- so the
  oracle behind a blocked claim is the shipped, tested decision procedure, not a
  bespoke evaluator.

Everything is decidable and bounded: a formula over more than
:data:`_MAX_VARS` distinct variables raises :class:`PropositionalParseError`
(fail-closed to "unavailable" upstream) rather than risking a pathological solve.
"""

from __future__ import annotations

from dataclasses import dataclass

from omni_mercury_engine.verifiers.propositional import Clause, Literal, is_satisfiable

#: Hard cap on distinct variables; a larger formula is refused (fail-closed).
_MAX_VARS = 20

# Word/symbol spellings of each connective (checked longest-first when scanning).
_NOT = ("~", "!", "¬", "not")
_AND = ("&&", "&", "∧", "/\\", "and")
_OR = ("||", "|", "∨", "\\/", "or")
_IFF = ("<->", "<=>", "↔", "≡", "iff")
_IMP = ("->", "=>", "→", "implies", "imp")
_XOR = ("^", "⊕", "xor")


class PropositionalParseError(ValueError):
    """A formula string was malformed or exceeded the bounded variable budget."""


# --------------------------------------------------------------------------- #
# AST.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Node:
    """An AST node: a variable (``op='var'``) or an operator over children."""

    op: str  # 'var' | 'not' | 'and' | 'or' | 'imp' | 'iff' | 'xor'
    name: str = ""
    children: tuple[Node, ...] = ()


# --------------------------------------------------------------------------- #
# Tokenizer.
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> list[str]:
    """Split ``text`` into variable, connective, and paren tokens."""
    tokens: list[str] = []
    i = 0
    n = len(text)
    # Multi-char symbol spellings, longest first so '<->' beats '<' etc.
    symbolic = sorted(
        [s for group in (_NOT, _AND, _OR, _IFF, _IMP, _XOR) for s in group if not s.isalpha()],
        key=len,
        reverse=True,
    )
    word_ops = {w for group in (_NOT, _AND, _OR, _IFF, _IMP, _XOR) for w in group if w.isalpha()}
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            tokens.append(ch)
            i += 1
            continue
        matched = next((s for s in symbolic if text.startswith(s, i)), None)
        if matched is not None:
            tokens.append(matched)
            i += len(matched)
            continue
        if ch.isalnum() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            tokens.append(word.lower() if word.lower() in word_ops else word)
            i = j
            continue
        raise PropositionalParseError(f"unexpected character {ch!r} at position {i} in {text!r}")
    return tokens


def _canon(tok: str) -> str:
    """Map a token to a canonical connective name, or ``''`` if it is not one."""
    low = tok.lower()
    for name, group in (
        ("not", _NOT),
        ("and", _AND),
        ("or", _OR),
        ("iff", _IFF),
        ("imp", _IMP),
        ("xor", _XOR),
    ):
        if tok in group or low in group:
            return name
    return ""


# --------------------------------------------------------------------------- #
# Recursive-descent parser (precedence: iff < imp < or < xor < and < not < atom).
# --------------------------------------------------------------------------- #
class _Parser:
    """Recursive-descent parser over the token list."""

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> Node:
        node = self._iff()
        if self.pos != len(self.tokens):
            raise PropositionalParseError(f"trailing tokens after parse: {self.tokens[self.pos:]}")
        return node

    def _binary(self, sub: str, ops: tuple[str, ...], node_op: str) -> Node:
        left: Node = getattr(self, sub)()
        while (tok := self._peek()) is not None and _canon(tok) in ops:
            self._next()
            right: Node = getattr(self, sub)()
            left = Node(node_op, children=(left, right))
        return left

    def _iff(self) -> Node:
        return self._binary("_imp", ("iff",), "iff")

    def _imp(self) -> Node:
        return self._binary("_or", ("imp",), "imp")

    def _or(self) -> Node:
        return self._binary("_xor", ("or",), "or")

    def _xor(self) -> Node:
        return self._binary("_and", ("xor",), "xor")

    def _and(self) -> Node:
        return self._binary("_not", ("and",), "and")

    def _not(self) -> Node:
        tok = self._peek()
        if tok is not None and _canon(tok) == "not":
            self._next()
            return Node("not", children=(self._not(),))
        return self._atom()

    def _atom(self) -> Node:
        tok = self._peek()
        if tok is None:
            raise PropositionalParseError("unexpected end of formula")
        if tok == "(":
            self._next()
            node = self._iff()
            if self._peek() != ")":
                raise PropositionalParseError("missing closing parenthesis")
            self._next()
            return node
        if tok == ")" or _canon(tok):
            raise PropositionalParseError(f"expected a variable, found {tok!r}")
        self._next()
        return Node("var", name=tok)


def parse(formula: str) -> Node:
    """Parse ``formula`` into an AST (raises :class:`PropositionalParseError`)."""
    tokens = _tokenize(formula)
    if not tokens:
        raise PropositionalParseError("empty formula")
    return _Parser(tokens).parse()


def parse_trailing(text: str) -> Node | None:
    """Parse the *longest parseable suffix* of ``text`` as a formula, or ``None``.

    Generative text embeds a formula after prose ("Note that P and not P is a
    tautology" captures "Note that P and not P"). The formula the author means is
    the longest token-suffix that parses cleanly, so this drops leading prose
    without a fragile prose/formula regex: it tokenizes once and returns the AST
    of the earliest start index whose remaining tokens form a complete formula.
    ``None`` when no suffix parses (the capture was not a formula at all).
    """
    tokens = _tokenize(text)
    for start in range(len(tokens)):
        # A formula starts with a variable, '(' or a negation -- never a binary
        # connective or ')'.
        head = tokens[start]
        if head == ")" or _canon(head) in ("and", "or", "imp", "iff", "xor"):
            continue
        try:
            return _Parser(tokens[start:]).parse()
        except PropositionalParseError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Normalization to {var, not, and, or} then Tseitin CNF.
# --------------------------------------------------------------------------- #
def _normalize(node: Node) -> Node:
    """Rewrite imp/iff/xor into ``{var, not, and, or}`` (semantics preserved)."""
    if node.op == "var":
        return node
    if node.op == "not":
        return Node("not", children=(_normalize(node.children[0]),))
    a = _normalize(node.children[0])
    b = _normalize(node.children[1])
    if node.op in ("and", "or"):
        return Node(node.op, children=(a, b))
    if node.op == "imp":  # a -> b == ~a | b
        return Node("or", children=(Node("not", children=(a,)), b))
    if node.op == "iff":  # a <-> b == (~a | b) & (~b | a)
        return Node(
            "and",
            children=(
                Node("or", children=(Node("not", children=(a,)), b)),
                Node("or", children=(Node("not", children=(b,)), a)),
            ),
        )
    if node.op == "xor":  # a ^ b == (a | b) & (~a | ~b)
        return Node(
            "and",
            children=(
                Node("or", children=(a, b)),
                Node("or", children=(Node("not", children=(a,)), Node("not", children=(b,)))),
            ),
        )
    raise PropositionalParseError(f"unknown operator {node.op!r}")  # pragma: no cover


class _Tseitin:
    """Tseitin transform of a normalized AST to CNF over propositional Literals."""

    def __init__(self) -> None:
        self.clauses: list[Clause] = []
        self._counter = 0

    def _fresh(self) -> Literal:
        self._counter += 1
        return Literal(f"_t{self._counter}", True)

    def encode(self, node: Node) -> Literal:
        """Return the literal equivalent to ``node``, appending defining clauses."""
        if node.op == "var":
            return Literal(node.name, True)
        if node.op == "not":
            child = self.encode(node.children[0])
            return ~child  # negating a literal needs no aux var / clauses
        a = self.encode(node.children[0])
        b = self.encode(node.children[1])
        t = self._fresh()
        if node.op == "and":  # t <-> (a & b)
            self.clauses += [frozenset({~t, a}), frozenset({~t, b}), frozenset({t, ~a, ~b})]
        elif node.op == "or":  # t <-> (a | b)
            self.clauses += [frozenset({t, ~a}), frozenset({t, ~b}), frozenset({~t, a, b})]
        else:  # pragma: no cover - normalization removes imp/iff/xor
            raise PropositionalParseError(f"non-normalized operator {node.op!r}")
        return t


def to_cnf(node: Node) -> tuple[Literal, tuple[Clause, ...]]:
    """Tseitin-transform ``node`` to ``(root_literal, defining_clauses)``.

    Raises:
        PropositionalParseError: if the formula spans more than :data:`_MAX_VARS`
            distinct variables (fail-closed on an unbounded solve).
    """
    normalized = _normalize(node)
    variables = _collect_vars(normalized)
    if len(variables) > _MAX_VARS:
        raise PropositionalParseError(
            f"formula uses {len(variables)} variables; the bounded oracle caps at {_MAX_VARS}"
        )
    tseitin = _Tseitin()
    root = tseitin.encode(normalized)
    return root, tuple(tseitin.clauses)


def _collect_vars(node: Node) -> set[str]:
    """Return the set of variable names appearing in ``node``."""
    if node.op == "var":
        return {node.name}
    out: set[str] = set()
    for child in node.children:
        out |= _collect_vars(child)
    return out


def node_is_satisfiable(node: Node) -> bool:
    """Whether the AST ``node`` has a satisfying assignment (via the DPLL oracle)."""
    root, clauses = to_cnf(node)
    return is_satisfiable((*clauses, frozenset({root})))


def node_is_tautology(node: Node) -> bool:
    """Whether the AST ``node`` is a tautology: its negation is unsatisfiable (DPLL)."""
    root, clauses = to_cnf(node)
    # tautology(phi) iff (CNF(phi) & ~root) is UNSAT.
    return not is_satisfiable((*clauses, frozenset({~root})))


def is_satisfiable_formula(formula: str) -> bool:
    """Whether ``formula`` has a satisfying assignment (via the DPLL oracle)."""
    return node_is_satisfiable(parse(formula))


def is_tautology(formula: str) -> bool:
    """Whether ``formula`` is a tautology: its negation is unsatisfiable (DPLL)."""
    return node_is_tautology(parse(formula))


__all__ = [
    "Node",
    "PropositionalParseError",
    "is_satisfiable_formula",
    "is_tautology",
    "node_is_satisfiable",
    "node_is_tautology",
    "parse",
    "parse_trailing",
    "to_cnf",
]
