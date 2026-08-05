# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Obfuscation-resistant text normalization and multilingual hazard lexicons.

The weapons/mass-casualty uplift gate (:mod:`.ethical_bounding`) historically
matched an English-only hazard vocabulary against the raw lowercased query. Two
classes of evasion sail straight past that:

* **Obfuscation** -- homoglyph substitution (Cyrillic ``а`` for Latin ``a``),
  leetspeak (``n3rv3 ag3nt``), zero-width joiners, and per-character spacing
  (``s a r i n`` / ``n-e-r-v-e``) all break a naive ``substring in text`` scan.
* **Language** -- an English keyword list cannot route a query written in
  Spanish, Russian, Arabic, or Chinese.

This module closes both, deterministically and with no model dependency, in a
way that fits Axis A's *high-recall / no-precision-required* router contract
(see ``docs/HARM_POLICY.md`` §3): a spurious normalization collapse only routes
a query into the Axis-B intent assessment -- it is never, by itself, a reason to
refuse -- so aggressive folding is safe here.

Two public surfaces:

* :func:`normalized_haystack` returns a newline-joined bundle of normalization
  variants (base, obfuscation-folded, separator-collapsed). Existing substring
  and regex matching runs unchanged over the bundle and therefore sees every
  variant at once.
* :data:`MULTILINGUAL_HAZARD_TERMS` / :data:`MULTILINGUAL_OFFENSIVE_CUES`
  extend the English Axis-A / Axis-B lexicons with taxonomy-level category
  terms in the world's most widely spoken languages (native script **and**
  Latin transliteration). These are dictionary-level category nouns/verbs --
  never operational specifics -- consistent with the public CBRNE/
  non-proliferation vocabulary the English lexicon already uses.

Pure stdlib -- no third-party dependency -- so it loads anywhere the ethics
gate does.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Character-level de-obfuscation.
# ---------------------------------------------------------------------------

# Zero-width and format characters an adversary interleaves to break substring
# matching ("sa​rin"). Mapped to None => deleted.
_ZERO_WIDTH_CODEPOINTS = (
    0x00AD,  # soft hyphen
    0x180E,  # Mongolian vowel separator
    0x200B,  # zero-width space
    0x200C,  # zero-width non-joiner
    0x200D,  # zero-width joiner
    0x200E,  # left-to-right mark
    0x200F,  # right-to-left mark
    0x202A,  # left-to-right embedding
    0x202B,  # right-to-left embedding
    0x202C,  # pop directional formatting
    0x202D,  # left-to-right override
    0x202E,  # right-to-left override
    0x2060,  # word joiner
    0x2061,  # function application
    0x2062,  # invisible times
    0x2063,  # invisible separator
    0x2064,  # invisible plus
    0xFEFF,  # zero-width no-break space / BOM
)
_ZERO_WIDTH = dict.fromkeys(_ZERO_WIDTH_CODEPOINTS, None)

# Confusable -> ASCII fold. Covers the Cyrillic/Greek/full-width letters whose
# glyphs are visually identical to Latin, plus a few common look-alikes. The
# fold is applied only to build the *obfuscation-folded* variant, never to the
# base variant, so native-script multilingual terms still match in the base.
_HOMOGLYPHS: dict[str, str] = {
    # Cyrillic
    "а": "a",
    "в": "b",
    "е": "e",
    "к": "k",
    "м": "m",
    "н": "h",
    "о": "o",
    "р": "p",
    "с": "c",
    "т": "t",
    "у": "y",
    "х": "x",
    "ѕ": "s",
    "і": "i",
    "ј": "j",
    "ԁ": "d",
    "ԛ": "q",
    "ԝ": "w",
    "г": "r",
    # Greek
    "α": "a",
    "β": "b",
    "ε": "e",
    "ι": "i",
    "κ": "k",
    "μ": "m",
    "ν": "v",
    "ο": "o",
    "ρ": "p",
    "τ": "t",
    "υ": "u",
    "χ": "x",
    "ζ": "z",
    "η": "n",
    # Full-width Latin
    "ａ": "a",
    "ｂ": "b",
    "ｃ": "c",
    "ｄ": "d",
    "ｅ": "e",
    "ｆ": "f",
    "ｇ": "g",
    "ｈ": "h",
    "ｉ": "i",
    "ｊ": "j",
    "ｋ": "k",
    "ｌ": "l",
    "ｍ": "m",
    "ｎ": "n",
    "ｏ": "o",
    "ｐ": "p",
    "ｑ": "q",
    "ｒ": "r",
    "ｓ": "s",
    "ｔ": "t",
    "ｕ": "u",
    "ｖ": "v",
    "ｗ": "w",
    "ｘ": "x",
    "ｙ": "y",
    "ｚ": "z",
}

# Leetspeak digit/symbol -> letter. Single-valued (high-recall router tolerates
# the ambiguity of 1->i vs 1->l); applied only to the folded variant.
_LEET: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
    "!": "i",
    "|": "l",
    "£": "l",
    "€": "e",
    "+": "t",
    "¡": "i",
}

_COLLAPSE_RE = re.compile(r"[^a-z0-9-￿]+")
_WHITESPACE_RE = re.compile(r"\s+")
# Intra-word punctuation only (keeps whitespace + word chars incl. non-Latin
# scripts), so a per-character-DOTTED multi-word term ("n.e.r.v.e a.g.e.n.t")
# rejoins to "nerve agent" without merging the two words.
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _strip_diacritics(text: str) -> str:
    """Fold Latin diacritics (``é``->``e``, ``ñ``->``n``) via NFKD mark removal.

    Non-Latin scripts (Cyrillic, Arabic, CJK) carry no combining marks here and
    pass through unchanged, so native-script multilingual terms are preserved.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def base_normalize(text: str) -> str:
    """Lowercase, NFKC-normalize, strip zero-width/format chars and diacritics.

    Script-preserving: this is the variant native-script multilingual terms are
    matched against.
    """
    t = unicodedata.normalize("NFKC", text or "").translate(_ZERO_WIDTH).lower()
    t = _strip_diacritics(t)
    return _WHITESPACE_RE.sub(" ", t).strip()


def _fold_obfuscation(base: str) -> str:
    """Apply homoglyph + leet folds to a base-normalized string."""
    out: list[str] = []
    for ch in base:
        ch = _HOMOGLYPHS.get(ch, ch)
        out.append(_LEET.get(ch, ch))
    return "".join(out)


def _collapse(folded: str) -> str:
    """Remove separators so per-character spacing (``s a r i n``) rejoins."""
    return _COLLAPSE_RE.sub("", folded)


def _punct_strip(folded: str) -> str:
    """Drop intra-word punctuation but keep word boundaries (dotted multi-words)."""
    return _WHITESPACE_RE.sub(" ", _PUNCT_RE.sub("", folded)).strip()


# A run of 2+ whitespace is the word gap in a per-character-spaced phrase: a
# human writing "n e r v e   a g e n t" separates the letters with one space and
# the words with more. Single spaces inside such a run are letter separators.
_WORD_GAP_RE = re.compile(r"\s{2,}")


def _rejoin_spaced_words(text: str) -> str:
    r"""Rebuild words from per-character spacing; ``""`` when there is nothing to rebuild.

    ``"h o w   t o   m a k e   a   n e r v e   a g e n t"`` becomes
    ``"how to make a nerve agent"``. Runs of two or more consecutive
    single-character tokens are joined; every other token is passed through, so
    ordinary prose and the legitimate one-letter words ``a``/``i`` survive
    unchanged.

    This complements -- it does not replace -- the *collapsed* variant. Collapsing
    removes every separator, which recovers a spaced single-token term
    (``s a r i n``) but destroys the word boundaries that the Axis-B intent
    regexes need in order to match at all. Rejoining keeps those boundaries, so a
    spaced-out query is assessed for *intent*, not merely routed by Axis A.

    When the attacker marks word gaps (2+ spaces) the segmentation is read
    straight off the spacing. Under **uniform single spacing** there is no such
    marker -- ``"m a k e a b o m b"`` glues to ``"makeabomb"`` -- so each glued
    run is handed to :func:`segment_glued_run`, which recovers the word
    boundaries from Mercury's own gate vocabulary instead of guessing. That
    matters because gluing alone defeats Axis B: the intent regexes are written
    with ``\b`` boundaries, so a boundary-free run matches none of them.

    Returns:
        The rejoined text, or ``""`` when no run was found (the caller then adds
        no variant, so ordinary queries pay nothing).
    """
    # When the text carries explicit word gaps the attacker has already told us
    # where the words end; re-deriving boundaries from a vocabulary could only
    # contradict that, and did -- an early revision that segmented every run
    # regressed the marked-gap corpus from 182/182 blocked to 162/182. Recovery
    # therefore runs only on uniformly-spaced text, where nothing else can.
    recover = _WORD_GAP_RE.search(text) is None
    joined_any = False
    groups: list[str] = []
    for group in _WORD_GAP_RE.split(text):
        out: list[str] = []
        run: list[str] = []

        def flush(out: list[str] = out, run: list[str] = run) -> None:
            nonlocal joined_any
            if len(run) >= 2:
                glued = "".join(run)
                out.append(segment_glued_run(glued) if recover else glued)
                joined_any = True
            else:
                out.extend(run)
            run.clear()

        for token in group.split():
            if len(token) == 1 and token.isalnum():
                run.append(token)
            else:
                flush()
                out.append(token)
        flush()
        if out:
            groups.append(" ".join(out))
    return " ".join(groups) if joined_any else ""


# ---------------------------------------------------------------------------
# Word-boundary recovery for uniformly-spaced text.
#
# Per-character spacing with a *uniform* single separator ("m a k e a b o m b")
# carries no word-gap marker, so rejoining yields one glued run. Axis A survives
# that -- ``term_match_forms`` probes each lexicon term's own collapsed form --
# but Axis B does not: every offensive-intent pattern is written with ``\b``
# boundaries and a boundary-free run matches none of them. Measured on the
# fit-on offensive corpus, spacing every character with a single space dropped
# the gate from 182/182 blocked to 8/182.
#
# The recovery is a dynamic program over Mercury's own gate vocabulary: prefer
# the segmentation that leaves the fewest characters unexplained, tie-broken
# toward fewer words. Spans no vocabulary explains stay glued rather than being
# split arbitrarily, so the function never invents a boundary it cannot justify.
# It runs only on text that already contained a per-character-spaced run, which
# ordinary prose never does.
# ---------------------------------------------------------------------------

#: Closed-class English and the request/production/targeting/evasion vocabulary
#: the Axis-B intent patterns are written over. Sourced from those patterns and
#: from ordinary function words -- never from an evaluation corpus, so recovery
#: cannot become memorisation of the benchmark. Hazard nouns are *not* listed
#: here: :func:`register_segmentation_terms` contributes them from the Axis-A
#: lexicon at import, keeping one source of truth.
#:
#: .. important::
#:    The **allow** vocabulary below is not optional politeness -- it is what
#:    keeps recovery symmetric. Segmentation decides which words the Axis-B
#:    regexes can see, so a vocabulary carrying only offensive verbs renders a
#:    defensive query unreadable while leaving an offensive one legible, and the
#:    gate then refuses work it exists to support. Measured: with the offensive
#:    half alone, spacing the benign corpus produced **14/180 false positives**
#:    ("how to detect a anthrax and diagnose exposure" among them) because
#:    ``detect`` and ``diagnose`` had no entries to segment on. With the allow
#:    half present it is 0/180. Any future growth of the offensive half must
#:    carry the corresponding defensive terms in the same change.
_SEGMENTATION_WORD_GROUPS: tuple[str, ...] = (
    # --- shared: function words, determiners, pronouns, prepositions ---
    "a an the and or of for to in into on at by as with without from that this "
    "it is are be can could would should will do does did i we you one me my "
    "some any all more most best other another each every no not",
    # --- shared: request frames ---
    "how what when where which who why need want trying help teach give tell "
    "show walk through step steps guide instructions instruction recipe recipes "
    "tutorial method methods process procedure route routes way ways plan plans "
    "list parts order detail details",
    # --- offensive: production / acquisition verbs and their inflections ---
    "make makes making made produce produces producing production synthesize "
    "synthesise synthesizing synthesis manufacture manufacturing fabricate "
    "fabrication build building construct construction assemble assembly "
    "formulate create creating generate cook cooking brew brewing prepare "
    "preparation obtain acquire acquiring source buy purchase get precursor "
    "precursors ingredient ingredients material materials equipment",
    # --- offensive: quantity / potency ---
    "amount amounts dose dosage quantity batch gram grams kilogram kilograms "
    "purity yield concentration potency lethal deadly grade weapons",
    # --- offensive: targeting / mass-harm framing ---
    "attack attacking target targeting maximize maximise maximizing casualty "
    "casualties kill killing harm harming damage destroy deploy deployment "
    "disperse dispersal release spread poison poisoning contaminate mass crowd "
    "stadium water supply air food people population civilian civilians",
    # --- offensive: weaponisation / evasion ---
    "weapon weaponize weaponise weaponizing evade evading avoid avoiding "
    "undetected untraceable bypass conceal concealing hide smuggle "
    "use using against",
    # --- allow: detection / identification ---
    "detect detects detected detecting detector detectors detection identify "
    "identifies identifying identification diagnose diagnosis diagnostic "
    "diagnostics screen screening test testing kit assay sensor sensors "
    "biosensor alarm alarms monitor monitors monitoring recognize recognise "
    "symptom symptoms sequencing surveillance epidemiology dosimeter",
    # --- allow: treatment / protection ---
    "treat treats treatment treating antidote antidotes decontaminate "
    "decontamination protective equipment gear suit mask clothing ppe vaccine "
    "vaccines vaccination therapy therapeutic dosing oxime atropine antitoxin "
    "casualty care countermeasure countermeasures prophylaxis respirator "
    "filter filtering medical",
    # --- allow: response / planning ---
    "triage consequence modeling modelling dispersion plume evacuation "
    "emergency response management exposure",
    # --- allow: policy / licensed practice / mechanism ---
    "policy regulation regulations compliance treaty history safety standard "
    "standards proliferation export control law licensed certified permit "
    "regulatory approval commercial blasting demolition controlled licensing "
    "requirement mechanism action chemistry physics work works research "
    "defend defense defence protect protection safely",
)

_BASE_SEGMENTATION_WORDS: frozenset[str] = frozenset(
    word for group in _SEGMENTATION_WORD_GROUPS for word in group.split()
)

#: Vocabulary contributed by the gate's Axis-A lexicon via
#: :func:`register_segmentation_terms`. Kept separate from the base set so the
#: registration is idempotent and inspectable.
_REGISTERED_SEGMENTATION_WORDS: set[str] = set()

#: Longest run handed to the dynamic program. The gate already bounds its
#: subject string; this is a second, local ceiling so segmentation can never
#: become a cost centre on pathological input.
_MAX_SEGMENT_RUN = 512

#: Shortest vocabulary entry admitted. Two-character entries match almost
#: anywhere, and because the dynamic program's first objective is to leave few
#: characters unexplained, admitting them makes *shredding an unknown word*
#: cheaper than leaving it whole: with a floor of 2, ``dissemination`` came back
#: as ``d is sem in at i on``, which matches no pattern at all. A floor of 3
#: costs nothing, because short function words do not need to be in the
#: vocabulary to be recovered -- an unexplained character sitting between two
#: known words already becomes its own token, so ``howtomakeabomb`` still yields
#: ``how to make a bomb`` with neither ``to`` nor ``a`` listed.
_MIN_VOCAB_LEN = 3

#: Strip regex metacharacters when harvesting literal words from a compiled
#: pattern. ``\b``/``\w``/``\s`` and friends become separators so the harvest
#: does not weld an escape onto the word beside it.
_REGEX_ESCAPE_RE = re.compile(r"\\[A-Za-z]")
_REGEX_LITERAL_WORD_RE = re.compile(r"[a-z]{4,}")


def register_segmentation_terms(terms: Iterable[str]) -> None:
    """Contribute lexicon terms to the word-boundary recovery vocabulary.

    Called by the gate at import time with its Axis-A hazard lexicon, so the
    segmenter and the router share one vocabulary and cannot drift apart. Terms
    are base-normalised and split on whitespace, so a multi-word entry
    (``"nerve agent"``) contributes both of its words. Idempotent.

    Args:
        terms: Lexicon entries, e.g. the Axis-A hazard keywords.
    """
    for term in terms:
        for word in base_normalize(term).split():
            folded = _fold_obfuscation(word)
            if len(folded) >= _MIN_VOCAB_LEN and folded.isalnum():
                _REGISTERED_SEGMENTATION_WORDS.add(folded)


def register_pattern_literals(patterns: Iterable[str]) -> None:
    """Contribute the literal words of Axis-B regex sources to the vocabulary.

    Segmentation decides which words the intent patterns are *able* to see, so
    any word a pattern can match must be recoverable or that pattern is dead on
    uniformly-spaced input. Harvesting the literals from the pattern sources
    themselves makes that automatic and symmetric: an offensive pattern and an
    allow pattern contribute by the same rule, in the same change that adds
    them, with no second list to keep in step.

    Only runs of four or more lowercase letters are taken, which skips the
    optional-suffix fragments regex alternations are full of (``(?:ing)?``,
    ``(?:ors)?``) -- those are too short to be worth segmenting on and admitting
    them would reintroduce shredding.

    Args:
        patterns: Regex source strings, e.g. ``pattern.pattern`` for each
            compiled Axis-B pattern.
    """
    for source in patterns:
        cleaned = _REGEX_ESCAPE_RE.sub(" ", source.lower())
        _REGISTERED_SEGMENTATION_WORDS.update(_REGEX_LITERAL_WORD_RE.findall(cleaned))


def _vocabulary() -> frozenset[str]:
    """The current segmentation vocabulary (base + everything registered)."""
    return frozenset(_BASE_SEGMENTATION_WORDS | _REGISTERED_SEGMENTATION_WORDS)


def segment_glued_run(run: str) -> str:
    """Recover word boundaries in a glued run using the gate's own vocabulary.

    ``"attackplantomaximizecasualties"`` becomes
    ``"attack plan to maximize casualties"``. Characters no vocabulary word
    explains are kept glued together rather than split at a guessed boundary, so
    the result never asserts a segmentation the vocabulary cannot support:
    ``"makeaqqqbomb"`` yields ``"make a qqq bomb"``, not a fabricated reading of
    ``qqq``.

    The dynamic program minimises unexplained characters first and word count
    second. Preferring fewer words on a tie is what stops a long word being
    shredded into a chain of short ones ("casualties" surviving instead of
    becoming "casual ties").

    Args:
        run: A glued run of alphanumerics, already obfuscation-folded.

    Returns:
        The run with recovered boundaries, or the run unchanged when it is
        empty, over :data:`_MAX_SEGMENT_RUN`, or explains nothing.
    """
    n = len(run)
    if not run or n > _MAX_SEGMENT_RUN:
        return run
    vocab = _vocabulary()
    longest = max((len(w) for w in vocab), default=0)
    if not longest:
        return run

    # dp[i] = (unexplained chars, words) for the best segmentation of run[:i];
    # back[i] = length of the final piece and whether it was a vocabulary hit.
    inf = (n + 1, n + 1)
    dp: list[tuple[int, int]] = [inf] * (n + 1)
    back: list[tuple[int, bool]] = [(0, False)] * (n + 1)
    dp[0] = (0, 0)
    for i in range(1, n + 1):
        # Unexplained single character: costs one, and does not open a new word
        # when the previous piece was also unexplained (they stay glued).
        prev_unknown = back[i - 1][1] is False and i - 1 > 0
        cand = (dp[i - 1][0] + 1, dp[i - 1][1] + (0 if prev_unknown else 1))
        best, best_back = cand, (1, False)
        for length in range(min(longest, i), _MIN_VOCAB_LEN - 1, -1):
            if run[i - length : i] in vocab:
                prior = dp[i - length]
                cand = (prior[0], prior[1] + 1)
                if cand < best:
                    best, best_back = cand, (length, True)
        dp[i], back[i] = best, best_back

    if dp[n][0] >= n:  # nothing was explained -- do not assert a segmentation
        return run

    # Walk the back-pointers, then merge neighbouring unexplained pieces so an
    # unrecognised span stays one glued token instead of becoming loose letters.
    pieces: list[tuple[str, bool]] = []
    i = n
    while i > 0:
        length, known = back[i]
        pieces.append((run[i - length : i], known))
        i -= length
    pieces.reverse()

    merged: list[str] = []
    last_known = True
    for piece, known in pieces:
        if not known and merged and not last_known:
            merged[-1] += piece
        else:
            merged.append(piece)
        last_known = known
    return " ".join(merged)


# Cyrillic -> Latin PHONETIC transliteration (distinct from the visual homoglyph
# fold above: e.g. 'в' is visually 'B' but phonetically 'v'). Complements the
# homoglyph variant so a *phonetic* Cyrillic obfuscation ("н3рв3" mixing Cyrillic
# н/р/в with leetspeak) also normalizes toward its Latin form. Applied only to
# build an extra variant; the base and visual-fold variants are unchanged.
_CYRILLIC_TRANSLIT: dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "ґ": "g",
    "д": "d",
    "е": "e",
    "є": "ie",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "і": "i",
    "ї": "i",
    "й": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def _cyrillic_translit(base: str) -> str:
    """Phonetically transliterate any Cyrillic in ``base`` to Latin."""
    if not any(ch in _CYRILLIC_TRANSLIT for ch in base):
        return base
    return "".join(_CYRILLIC_TRANSLIT.get(ch, ch) for ch in base)


def normalized_haystack(text: str) -> str:
    r"""Return a newline-joined bundle of normalization variants for matching.

    The bundle is ``base \n folded \n despaced \n collapsed``:

    * **base** -- lowercased, zero-width-stripped, diacritic-folded, whitespace-
      collapsed. Script-preserving; multi-word and native-script terms match here.
    * **folded** -- base with homoglyph + leetspeak folded to ASCII. Catches
      ``n3rv3 ag3nt`` and Cyrillic-glyph spoofing.
    * **despaced** -- folded with intra-word punctuation dropped but word
      boundaries kept. Catches dotted/hyphenated multi-word terms
      (``n.e.r.v.e a.g.e.n.t``).
    * **rejoined** -- per-character-spaced words rebuilt using 2+ spaces as the
      word gap (``h o w   t o   m a k e`` -> ``how to make``). Present only when
      the query actually contains such a run. Unlike *collapsed* this keeps word
      boundaries, so the Axis-B intent regexes can still match a spaced query.
    * **collapsed** -- folded with *all* separators removed. Catches
      per-character spacing (``s a r i n``).

    .. important::
       The **collapsed** variant has every separator removed, so a lexicon term
       that itself contains a separator (``"nerve agent"``, ``"pipe bomb"`` --
       70% of the Axis-A lexicon) can *never* be found in it by a plain
       ``term in haystack`` test. Match lexicon terms with
       :func:`term_match_forms` + :func:`term_in_haystack`, which probe the
       term's own collapsed form as well; a raw substring test silently
       re-opens the per-character-spacing bypass those two functions exist to
       close.

    Regex matching (``pattern.search(haystack)``) runs unchanged and sees every
    variant. Fail-open on error: returns the plain lowercased text so a
    normalization bug never blinds the gate.
    """
    try:
        base = base_normalize(text)
        folded = _fold_obfuscation(base)
        collapsed = _collapse(folded)
        despaced = _punct_strip(folded)
        # Word-boundary-preserving de-spacing. Derived from the text *before*
        # base_normalize squeezes runs of whitespace, because the run length is
        # exactly the signal that distinguishes a letter gap from a word gap.
        # Empty (and therefore skipped) for any query without per-character
        # spacing, which is virtually all of them.
        rejoined = _fold_obfuscation(base_normalize(_rejoin_spaced_words(text or "")))
        # Phonetic-Cyrillic variant: transliterate then apply the same folds, so a
        # phonetic Cyrillic obfuscation (Cyrillic н/р/в mixed with leetspeak) also
        # normalizes toward Latin. Skipped when there is no Cyrillic (no cost).
        translit = _cyrillic_translit(base)
        translit_variants: tuple[str, ...] = ()
        if translit != base:
            t_folded = _fold_obfuscation(translit)
            translit_variants = (t_folded, _collapse(t_folded), _punct_strip(t_folded))
        # De-dup identical variants to keep the haystack compact.
        seen: set[str] = set()
        variants: list[str] = []
        for v in (base, folded, despaced, rejoined, collapsed, *translit_variants):
            if v and v not in seen:
                seen.add(v)
                variants.append(v)
        return "\n".join(variants)
    except Exception:
        return (text or "").lower()


# ---------------------------------------------------------------------------
# Lexicon-term matching against a normalized haystack.
#
# ``normalized_haystack`` emits a *collapsed* variant with every separator
# removed so per-character spacing ("s a r i n") still matches. That works for a
# single-token term, whose collapsed form equals itself -- but a term containing
# a separator ("nerve agent") keeps its space and therefore CANNOT occur in the
# collapsed variant at all. Because ~70% of the Axis-A lexicon is multi-word,
# a plain ``term in haystack`` test let per-character spacing evade almost the
# entire router: measured, spacing every character of the fit-on offensive
# corpus dropped the gate from 182/182 blocked to 1/182 (the single survivor was
# ``s a r i n``, the one *single-token* term). These two functions are the fix
# and the single source of truth for it -- see tests/ethical/
# test_weapons_gate_properties.py for the lexicon-wide property that pins it.
# ---------------------------------------------------------------------------


def canonical_normalize(text: str) -> str:
    """Return a single de-obfuscated form of ``text`` for tokenizing.

    :func:`normalized_haystack` returns *several* variants because substring and
    regex matching wants to see all of them at once. A tokenizer wants exactly
    one canonical string instead -- feeding it the bundle would double-count
    every token once per variant.

    This is that one string: lowercased, zero-width-stripped, diacritic- and
    homoglyph/leet-folded, with per-character spacing rebuilt into words where
    the word gaps make that recoverable. Obfuscation is therefore normalized
    away *before* tokenization, so a downstream model does not have to learn
    leetspeak.

    Fail-open: returns the plain lowercased text if normalization raises.
    """
    try:
        rejoined = _rejoin_spaced_words(text or "")
        base = base_normalize(rejoined or text)
        return _fold_obfuscation(base)
    except Exception:
        return (text or "").lower()


def term_match_forms(term: str) -> tuple[str, ...]:
    """Return every surface form of ``term`` that must be probed against a haystack.

    A lexicon term has to be matched in the same normalization space as the
    query. :func:`normalized_haystack` bundles several variants of the query, so
    a term needs the corresponding variants of itself:

    * its **base**-normalized form (matches the base/despaced query variants),
    * its **obfuscation-folded** form (matches the folded query variant -- this
      is what lets a native-script multilingual term match a homoglyph-folded
      query), and
    * its **collapsed**, separator-free form (matches the collapsed query
      variant -- this is what closes the per-character-spacing bypass for
      multi-word terms).

    Forms are de-duplicated, so a single-token ASCII term collapses to one
    string and costs no extra work. Intended to be called **once per term at
    import time** and the result cached; :func:`term_in_haystack` consumes it.

    Args:
        term: A lexicon entry, e.g. ``"nerve agent"`` or ``"sarin"``.

    Returns:
        A de-duplicated tuple of non-empty surface forms, sorted for
        determinism.
    """
    base = base_normalize(term)
    if not base:
        return ()
    folded = _fold_obfuscation(base)
    forms = {base, folded, _collapse(folded)}
    return tuple(sorted(f for f in forms if f))


def term_in_haystack(haystack: str, forms: tuple[str, ...]) -> bool:
    """True when any surface form of a lexicon term occurs in ``haystack``.

    Args:
        haystack: A bundle from :func:`normalized_haystack`.
        forms: The term's pre-compiled forms from :func:`term_match_forms`.

    Returns:
        ``True`` if the term is present in *any* normalization variant.
    """
    return any(form in haystack for form in forms)


def compile_terms(terms: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    """Pre-compile an iterable of lexicon terms into per-term match forms.

    Order is preserved so a caller can zip the result back against parallel
    metadata (intent tier, audit label) without re-deriving it.
    """
    return tuple(term_match_forms(t) for t in terms)


def any_term_in_haystack(haystack: str, compiled: tuple[tuple[str, ...], ...]) -> bool:
    """True when any pre-compiled term in ``compiled`` occurs in ``haystack``."""
    return any(term_in_haystack(haystack, forms) for forms in compiled)


# ---------------------------------------------------------------------------
# Multilingual hazard lexicon (Axis A extension).
#
# Taxonomy-level category terms only -- the same class of public
# CBRNE/non-proliferation vocabulary the English lexicon uses -- in native
# script and Latin transliteration across widely spoken languages. High recall
# is the goal: a hit only routes a query into the Axis-B intent gate; it never
# refuses on its own, so broad coverage here is safe and desired.
#
# Keyed by the HazardDomain *value* string (chemical/biological/... ) so
# ethical_bounding can merge them into _HAZARD_DOMAIN_KEYWORDS without importing
# the enum here (keeps this module dependency-light).
# ---------------------------------------------------------------------------

MULTILINGUAL_HAZARD_TERMS: dict[str, tuple[str, ...]] = {
    "chemical": (
        # Spanish / Portuguese / Italian / French / German / Dutch
        "arma quimica",
        "agente nervioso",
        "agente nervoso",
        "agente quimico",
        "gas nervioso",
        "gas venenoso",
        "arma quimica",
        "agente neurotoxico",
        "gas mostaza",
        "arme chimique",
        "agent neurotoxique",
        "gaz moutarde",
        "gaz innervant",
        "chemiewaffe",
        "nervengift",
        "nervenkampfstoff",
        "senfgas",
        "giftgas",
        "chemisch wapen",
        "zenuwgas",
        "arma chimica",
        "gas nervino",
        # Russian / Ukrainian (Cyrillic + translit)
        "химическое оружие",
        "нервно-паралитический газ",
        "зарин",
        "зоман",
        "иприт",
        "химичне зброя",
        "himicheskoe oruzhie",
        "nervno",
        # Arabic + translit
        "سلاح كيميائي",
        "غاز الأعصاب",
        "غاز السارين",
        "silah kimiyai",
        "ghaz al aasab",
        # Persian / Turkish / Hebrew
        "سلاح شیمیایی",
        "kimyasal silah",
        "sinir gazı",
        "נשק כימי",
        # Chinese (Simplified/Traditional) / Japanese / Korean
        "化学武器",
        "神经毒气",
        "沙林毒气",
        "芥子气",
        "神經毒氣",
        "神経ガス",
        "化学兵器",
        "サリン",
        "화학무기",
        "신경작용제",
        # Hindi / Bengali / Urdu / Indonesian / Vietnamese / Swahili / Polish
        "रासायनिक हथियार",
        "तंत्रिका गैस",
        "রাসায়নিক অস্ত্র",
        "کیمیائی ہتھیار",
        "senjata kimia",
        "gas saraf",
        "vũ khí hóa học",
        "silaha ya kemikali",
        "broń chemiczna",
        "gaz musztardowy",
    ),
    "biological": (
        "arma biologica",
        "agente patogeno",
        "arma biológica",
        "arme biologique",
        "agent pathogene",
        "arme bacteriologique",
        "biowaffe",
        "biologische waffe",
        "krankheitserreger",
        "milzbrand",
        "biologisch wapen",
        "ziekteverwekker",
        "arma biologica",
        "agente biologico",
        "биологическое оружие",
        "биологичне зброя",
        "сибирская язва",
        "biologicheskoe oruzhie",
        "vozbuditel",
        "سلاح بيولوجي",
        "سلاح جرثومي",
        "silah biولوji",
        "silah biyuluji",
        "biyolojik silah",
        "נשק ביולוגי",
        "سلاح بیولوژیک",
        "生物武器",
        "生化武器",
        "炭疽",
        "生物兵器",
        "생물무기",
        "탄저균",
        "जैविक हथियार",
        "রোগজীবাণু অস্ত্র",
        "حیاتیاتی ہتھیار",
        "senjata biologi",
        "vũ khí sinh học",
        "silaha ya kibayolojia",
        "broń biologiczna",
    ),
    "radiological": (
        "bomba sucia",
        "dispositivo de dispersion radiologica",
        "bombe sale",
        "arme radiologique",
        "schmutzige bombe",
        "radiologische waffe",
        "vuile bom",
        "bomba sporca",
        "грязная бомба",
        "радиологическое оружие",
        "gryaznaya bomba",
        "قنبلة قذرة",
        "قنبلة إشعاعية",
        "qunbula qadhira",
        "kirli bomba",
        "פצצה מלוכלכת",
        "بمب کثیف",
        "脏弹",
        "放射性武器",
        "汚い爆弾",
        "더러운 폭탄",
        "방사능 폭탄",
        "गंदा बम",
        "নোংরা বোমা",
        "bom kotor",
        "bom radiologi",
        "brudna bomba",
    ),
    "nuclear": (
        "arma nuclear",
        "bomba atomica",
        "material fisible",
        "uranio enriquecido",
        "arme nucleaire",
        "bombe atomique",
        "matiere fissile",
        "kernwaffe",
        "atomwaffe",
        "atombombe",
        "spaltbares material",
        "kernwapen",
        "arma nucleare",
        "bomba atomica",
        "ядерное оружие",
        "атомная бомба",
        "оружейный плутоний",
        "yadernoe oruzhie",
        "atomnaya bomba",
        "збройовий плутоній",
        "سلاح نووي",
        "قنبلة نووية",
        "يورانيوم مخصب",
        "silah nawawi",
        "nükleer silah",
        "atom bombası",
        "נשק גרעיני",
        "سلاح هسته‌ای",
        "核武器",
        "原子弹",
        "浓缩铀",
        "核兵器",
        "原子爆弾",
        "핵무기",
        "핵폭탄",
        "परमाणु हथियार",
        "परमाणु बम",
        "পারমাণবিক অস্ত্র",
        "جوہری ہتھیار",
        "senjata nuklir",
        "vũ khí hạt nhân",
        "silaha ya nyuklia",
        "broń jądrowa",
        "broń nuklearna",
    ),
    "high_yield_explosive": (
        "artefacto explosivo improvisado",
        "explosivo de alto poder",
        "bomba de tuberia",
        "coche bomba",
        "detonador",
        "engin explosif improvise",
        "explosif brisant",
        "voiture piegee",
        "unkonventionelle spreng",
        "sprengfalle",
        "rohrbombe",
        "autobombe",
        "bermbom",
        "ordigno esplosivo",
        "autobomba",
        "самодельное взрывное устройство",
        "взрывчатка",
        "детонатор",
        "самодельна вибухівка",
        "vzryvchatka",
        "самодельное",
        "عبوة ناسفة",
        "متفجرات",
        "سيارة مفخخة",
        "abwa nasifa",
        "el yapımı patlayıcı",
        "patlayıcı",
        "מטען חבלה",
        "מכונית תופת",
        "简易爆炸装置",
        "爆炸物",
        "汽车炸弹",
        "爆発物",
        "パイプ爆弾",
        "급조폭발물",
        "폭발물",
        "차량 폭탄",
        "इम्प्रोवाइज्ड विस्फोटक",
        "विस्फोटक",
        "গাড়ি বোমা",
        "دیسی ساختہ بم",
        "bom rakitan",
        "bahan peledak",
        "vật liệu nổ",
        "kilipuzi",
        "ładunek wybuchowy",
        "samochód pułapka",
    ),
    "conventional_weapon": (
        "arma fantasma",
        "arma no rastreable",
        "silenciador casero",
        "arme fantome",
        "arme non tracable",
        "geisterwaffe",
        "unregistrierte waffe",
        "arma fantasma",
        "неотслеживаемое оружие",
        "самодельный ствол",
        "سلاح شبح",
        "hayalet silah",
        "נשק רפאים",
        "幽灵枪",
        "ゴーストガン",
        "유령총",
        "senjata hantu",
        "súng ma",
    ),
    "cyber_offensive": (
        "programa de rescate",
        "ataque a infraestructura critica",
        "rançongiciel",
        "logiciel malveillant destructeur",
        "erpressungstrojaner",
        "schadsoftware",
        "программа-вымогатель",
        "вредоносное по",
        "destruktivnoe",
        "برامج الفدية",
        "fidye yazılımı",
        "zararlı yazılım",
        "勒索软件",
        "破坏性恶意软件",
        "ランサムウェア",
        "랜섬웨어",
        "perangkat pemeras",
        "mã độc tống tiền",
    ),
    "other_mass_harm": (
        "arma de destruccion masiva",
        "ataque con victimas masivas",
        "arme de destruction massive",
        "massenvernichtungswaffe",
        "оружие массового поражения",
        "oruzhie massovogo",
        "سلاح دمار شامل",
        "kitle imha silahı",
        "נשק להשמדה המונית",
        "大规模杀伤性武器",
        "大量破壊兵器",
        "대량살상무기",
        "सामूहिक विनाश का हथियार",
        "senjata pemusnah massal",
        "vũ khí hủy diệt hàng loạt",
        "broń masowego rażenia",
    ),
}


# ---------------------------------------------------------------------------
# Multilingual offensive-intent cues (Axis B extension).
#
# The core actionable verbs -- "how to make/produce/synthesize", "weaponize" --
# in widely spoken languages, as (substring, intent-tier-name, audit-label)
# triples. Unlike Axis A, these ARE weapons-actionability evidence, so the set
# is kept to the clearest, least ambiguous phrasings. The tier name maps to
# OperationalIntent members in ethical_bounding; production cues still require
# an Axis-A hazard-domain hit to gate (same rule as the English generic
# production verbs), so a benign non-English "how to cook" cannot trip them.
# ---------------------------------------------------------------------------

MULTILINGUAL_OFFENSIVE_CUES: tuple[tuple[str, str, str], ...] = (
    # --- production / synthesis "how to make X" (tier PRODUCTION) ---
    ("como fabricar", "PRODUCTION", "production_howto"),
    ("como producir", "PRODUCTION", "production_howto"),
    ("como sintetizar", "PRODUCTION", "production_howto"),
    ("comment fabriquer", "PRODUCTION", "production_howto"),
    ("comment produire", "PRODUCTION", "production_howto"),
    ("comment synthetiser", "PRODUCTION", "production_howto"),
    ("wie herstellen", "PRODUCTION", "production_howto"),
    ("herstellung von", "PRODUCTION", "production_howto"),
    ("come produrre", "PRODUCTION", "production_howto"),
    ("como fabricar", "PRODUCTION", "production_howto"),
    ("как изготовить", "PRODUCTION", "production_howto"),
    ("как сделать", "PRODUCTION", "production_howto"),
    ("как синтезировать", "PRODUCTION", "production_howto"),
    ("kak izgotovit", "PRODUCTION", "production_howto"),
    ("kak sdelat", "PRODUCTION", "production_howto"),
    ("كيفية صنع", "PRODUCTION", "production_howto"),
    ("كيفية تصنيع", "PRODUCTION", "production_howto"),
    ("nasıl yapılır", "PRODUCTION", "production_howto"),
    ("nasıl üretilir", "PRODUCTION", "production_howto"),
    ("如何制造", "PRODUCTION", "production_howto"),
    ("如何制作", "PRODUCTION", "production_howto"),
    ("合成方法", "PRODUCTION", "production_howto"),
    ("製造方法", "PRODUCTION", "production_howto"),
    ("作り方", "PRODUCTION", "production_howto"),
    ("만드는 방법", "PRODUCTION", "production_howto"),
    ("제조법", "PRODUCTION", "production_howto"),
    ("cara membuat", "PRODUCTION", "production_howto"),
    ("cách chế tạo", "PRODUCTION", "production_howto"),
    ("kaise banaye", "PRODUCTION", "production_howto"),
    ("बनाने का तरीका", "PRODUCTION", "production_howto"),
    ("jak wyprodukować", "PRODUCTION", "production_howto"),
    # --- weaponization (tier WEAPONIZATION, inherently offensive) ---
    ("convertir en arma", "WEAPONIZATION", "weaponization_design"),
    ("transformer en arme", "WEAPONIZATION", "weaponization_design"),
    ("als waffe", "WEAPONIZATION", "weaponization_design"),
    ("verwaffnung", "WEAPONIZATION", "weaponization_design"),
    ("превратить в оружие", "WEAPONIZATION", "weaponization_design"),
    ("боевое применение", "WEAPONIZATION", "weaponization_design"),
    ("تحويله إلى سلاح", "WEAPONIZATION", "weaponization_design"),
    ("silah haline getir", "WEAPONIZATION", "weaponization_design"),
    ("武器化", "WEAPONIZATION", "weaponization_design"),
    ("兵器化", "WEAPONIZATION", "weaponization_design"),
    ("무기화", "WEAPONIZATION", "weaponization_design"),
    ("senjatakan", "WEAPONIZATION", "weaponization_design"),
    # --- targeting / maximize casualties (tier TARGETING, inherently offensive) ---
    ("maximizar victimas", "TARGETING", "targeting_planning"),
    ("maximiser les victimes", "TARGETING", "targeting_planning"),
    ("möglichst viele opfer", "TARGETING", "targeting_planning"),
    ("максимум жертв", "TARGETING", "targeting_planning"),
    ("أكبر عدد من الضحايا", "TARGETING", "targeting_planning"),
    ("en fazla can kaybı", "TARGETING", "targeting_planning"),
    ("最大化伤亡", "TARGETING", "targeting_planning"),
    ("大量殺傷", "TARGETING", "targeting_planning"),
    ("최대 사상자", "TARGETING", "targeting_planning"),
)


__all__ = [
    "MULTILINGUAL_HAZARD_TERMS",
    "MULTILINGUAL_OFFENSIVE_CUES",
    "any_term_in_haystack",
    "base_normalize",
    "canonical_normalize",
    "compile_terms",
    "normalized_haystack",
    "register_segmentation_terms",
    "segment_glued_run",
    "term_in_haystack",
    "term_match_forms",
]
