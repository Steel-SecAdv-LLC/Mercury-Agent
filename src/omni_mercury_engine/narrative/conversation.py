# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Grounded answers for the questions a console user asks the agent.

Every answer here is built from one of three sources and says which: the
detection record (what ran, when, what was flagged, against which
threshold), the code (the detector manifest, the CLI's own command table, the
loader domains, the ethical gates, the API's severity bands), or the
narrative engine's synthesis of a recorded run. Nothing is estimated or
invented: a field a run did not report is left out, a number that lives in
code is read from code, and a question none of the three can answer gets a
plain statement of what can be asked instead of an apology with options.

:class:`MercuryVoice.speak` routes a question to one of these builders with
:func:`classify`; the builders return text and follow-up prompts, and the
voice wraps them in a ``VoiceResponse`` with the matching sources.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "SEVERITY_BANDS",
    "classify",
    "describe_detection",
    "detection_result_from_record",
    "fallback_menu",
    "follow_up",
    "glossary_answer",
    "parse_window",
    "relative_time",
]

# --------------------------------------------------------------------------- #
# Facts read from the code, with the modules they come from
# --------------------------------------------------------------------------- #

#: Severity bands of ``api.server._classify_severity``: the peak score divided
#: by the threshold, read left to right. Pinned to the server's values by
#: ``tests/narrative/test_voice_conversation.py``.
SEVERITY_BANDS: tuple[tuple[float, str], ...] = ((1.5, "low"), (2.5, "medium"), (4.0, "high"))

#: ``api.server.detect_univariate``'s z-score threshold for a sensitivity ``s``.
THRESHOLD_FORMULA = "2.0 + (1 - sensitivity) x 3.0"

#: Commands the agent may recommend, each verified against the live click
#: group before it is spoken (a command that does not exist is never named).
_NEXT_STEP_COMMANDS: tuple[tuple[str, str], ...] = (
    ("detect", "score a CSV or JSON file with the trained statistical ensemble"),
    ("tier-detect", "run the streaming detector tier on a 1-D series, torch-free"),
    ("rca", "attribute a multivariate anomaly to its root-cause nodes"),
    ("explain", "explain a detection decision"),
    ("hazard-viz", "render a hazard detector's diagnostics to PNG, GeoJSON or HTML"),
    ("build-features", "build a reusable feature archive from raw data"),
    ("train", "train the fusion model on labelled features"),
)

# --------------------------------------------------------------------------- #
# Question classification
# --------------------------------------------------------------------------- #

_GREETING = (
    "hello",
    "hi ",
    "hi!",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "greetings",
)
_THANKS = ("thank", "thx", "cheers", "appreciate")
_IDENTITY = re.compile(
    r"\b(?:who are you|what are you(?!r)|your name|introduce yourself|tell me about yourself)\b"
)
_GATE = re.compile(r"\bgates?\b")
_RECOMMEND = (
    "recommend",
    "what should i do",
    "what do i do",
    "course of action",
    "suggest an action",
)
_NEXT_RUN = (
    "what should i run",
    "run next",
    "what next",
    "next step",
    "where do i start",
    "how do i start",
)
_COLLECT = (
    "collect data",
    "load data",
    "get data",
    "data loader",
    "loaders",
    "which domains",
    "what domains",
    "data sources",
    "ingest",
    "how do i collect",
)
_POSTURE = ("posture", "threat level", "overall picture", "big picture", "how bad", "risk summary")
_COMPARE = ("compare", "versus", " vs ", "against yesterday", "than yesterday")
_DETAIL: dict[str, tuple[str, ...]] = {
    "samples": (
        "which sample",
        "what sample",
        "flagged sample",
        "which rows",
        "which points",
        "which indices",
    ),
    "threshold": (
        "what threshold",
        "which threshold",
        "threshold did",
        "threshold used",
        "the threshold",
    ),
    "method": (
        "which method",
        "what method",
        "which detector ran",
        "what detector ran",
        "which route",
    ),
    "score": ("what score", "peak score", "how high", "highest score"),
    "when": ("when was", "when did", "what time", "how long ago"),
}
_LAST_N = re.compile(
    r"\b(?:last|latest|recent|previous)\s+(\d+)\s+(?:runs?|detections?|events?|anomal\w*)"
)
_LIST_RUNS = (
    "show the runs",
    "list runs",
    "list the runs",
    "recent runs",
    "event timeline",
    "show runs",
    "show me the runs",
    "show me recent",
    "what happened",
    "recently",
    "lately",
    "historical events",
    "similar events",
    "timeline",
)
_STATUS = ("status", "health", "running", "operational")
_HELP = ("help", "how do you work", "guide me", "what can you do", "what can i ask")
_EXPLAIN_WORDS = ("explain", "narrate", "why", "walk me through", "interpret", "reasoning")
_EXPLAIN_VARIANTS: dict[str, tuple[str, ...]] = {
    "reasoning": ("reasoning chain", "full reasoning", "reasoning steps", "show reasoning"),
    "uncertainty": ("uncertainty", "unsure", "how sure"),
    "dossier": (
        "detailed analysis",
        "full analysis",
        "all the details",
        "full dossier",
        "everything about",
    ),
}
_CONFIDENCE = (
    "how confident",
    "confidence level",
    "your confidence",
    "how certain",
    "how reliable",
)
_STATS = ("detailed statistics", "statistics", "stats")
_HEALTH_DETAIL = ("component health", "components", "subsystems", "what is connected")
_CAPABILITY = ("capabilit", "what can you do", "are you able", "can you do")
_DETECTOR = ("detector", "what can you detect", "do you support", "do you handle", " cover ")
_CAN_DETECT = re.compile(r"\bcan you (?:detect|find|spot|catch)\b")
_COUNT = ("how many", "count", "number of", "summar", "overview", "tally", "total")
_DETECTION_STEMS = ("anomal", "detect", "flag", "alert", "event", "incident", "run")
_LATEST = ("latest", "last ", "most recent", "newest", "last?")
_RUN_COMMAND = re.compile(r"^(?:please\s+)?(?:run|execute|start|launch|kick off)\s+([a-z][a-z-]*)")
_GLOSSARY_LEAD = re.compile(
    r"^(?:what(?:'s| is| are| does| do)|define|meaning of|explain what|tell me about|what do you mean by)\b"
)


def classify(text: str) -> tuple[str, Any]:
    """Name the grounded answer a question should get, with any argument it needs.

    Returns ``(kind, argument)``; ``("search", None)`` means none of the
    grounded builders applies and the retriever's free-text search runs.
    The order matters: social openers and definitions are recognised before
    anything that would read "explain" or "run" as a request about the record.
    """
    t = " " + text.strip().lower() + " "
    bare = t.strip()
    if bare in {"hi", "hello", "hey", "yo", "hiya"} or any(
        t.startswith(" " + g) for g in _GREETING
    ):
        return "greeting", None
    if any(w in t for w in _THANKS) and len(bare) <= 40:
        return "thanks", None
    if _IDENTITY.search(t):
        return "identity", None
    if any(w in t for w in _HEALTH_DETAIL) or any(w in t for w in _STATS):
        return "status", "detailed"
    if any(w in t for w in _STATUS):
        return "status", None
    if any(w in t for w in _HELP):
        return "help", None
    term = _glossary_term(bare)
    if term is not None:
        return "glossary", term
    if _GATE.search(t) and any(w in t for w in ("clear", "why", "pass", "ethic", "what", "how")):
        return "gate", None
    if any(w in t for w in _RECOMMEND):
        return "recommend", None
    if any(w in t for w in _NEXT_RUN):
        return "next_run", None
    if any(w in t for w in _COLLECT):
        return "collect", None
    m = _RUN_COMMAND.match(bare)
    if m:
        return "run_command", m.group(1)
    if any(w in t for w in _POSTURE):
        return "posture", None
    if any(w in t for w in _COMPARE) and any(
        w in t for w in ("today", "yesterday", "hour", "week", "month")
    ):
        return "compare", None
    for aspect, words in _DETAIL.items():
        if any(w in t for w in words):
            return "detail", aspect
    m = _LAST_N.search(t)
    if m:
        return "last_n", max(1, min(int(m.group(1)), 20))
    if any(w in t for w in _LIST_RUNS):
        return "last_n", 5
    for variant, words in _EXPLAIN_VARIANTS.items():
        if any(w in t for w in words):
            return "explain", variant
    about_run = any(d in t for d in ("detection", "anomal", "run", "flag", "alert", "result"))
    if any(w in t for w in _EXPLAIN_WORDS) and about_run:
        return "explain", "narrative"
    if any(w in t for w in _CONFIDENCE):
        return "confidence", None
    if any(w in t for w in _COUNT) and any(d in t for d in _DETECTION_STEMS):
        return "summary", None
    if any(w in t for w in _LATEST) and any(d in t for d in _DETECTION_STEMS):
        return "latest", None
    if _CAN_DETECT.search(t) or any(w in t for w in _CAPABILITY) or any(w in t for w in _DETECTOR):
        return "capability", None
    return "search", None


# --------------------------------------------------------------------------- #
# Follow-ups that lean on the previous turn
# --------------------------------------------------------------------------- #

_ELABORATE = (
    "why",
    "explain",
    "tell me more",
    "more",
    "details",
    "go on",
    "elaborate",
    "what does that mean",
    "how so",
    "meaning",
    "and",
)
_WINDOW_WORDS = ("today", "yesterday", "hour", "week", "month")
_ORDINAL_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
_ORDINAL_RE = re.compile(r"\b(?:run|number|no\.?|entry|item)\s*#?(\d{1,2})\b|^#?(\d{1,2})$")


def follow_up(text: str) -> tuple[str, Any] | None:
    """Recognise a short reply that only makes sense against the previous turn.

    ``("ordinal", n)`` picks run ``n`` from the last listing (``-1`` the last
    one shown), ``("window", text)`` asks the previous question about another
    period ("and yesterday?"), ``("elaborate", None)`` asks for more ("why?",
    "tell me more"). ``None`` means the reply stands on its own. Only
    consulted when :func:`classify` found nothing, so a full question is
    never mistaken for a follow-up.
    """
    bare = text.strip().lower().rstrip("?!. ")
    if not bare or len(bare) > 40:
        return None
    m = _ORDINAL_RE.search(bare)
    if m:
        return "ordinal", int(m.group(1) or m.group(2))
    for word, n in _ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", bare):
            return "ordinal", n
    if "last one" in bare or "latest one" in bare:
        return "ordinal", -1
    if any(w in bare for w in _WINDOW_WORDS):
        return "window", bare
    if bare in _ELABORATE or any(bare.startswith(w + " ") for w in _ELABORATE):
        return "elaborate", None
    return None


# --------------------------------------------------------------------------- #
# Glossary
# --------------------------------------------------------------------------- #

_GLOSSARY_KEYS: tuple[tuple[str, str], ...] = (
    ("severity", r"\bseverity\b|\b(?:low|medium|high|critical) mean"),
    ("threshold", r"\bthreshold"),
    ("z-score", r"\bz[- ]?scores?\b"),
    ("anomaly score", r"\banomaly score|\bthe score\b"),
    ("confidence", r"\bconfidence\b"),
    ("ama", r"\bama\b"),
    ("findoyou", r"find\s?[oω]\s?you|findωyou"),
    ("decoupling", r"\bdecoupl|\bswitches\b"),
    ("detector", r"\ba detector\b|\bdetector mean|\bmanifest\b"),
    ("neurosymbolic", r"neuro-?symbolic"),
    ("gate", r"\bsigma\b|σ_immutable|harm[- ]uplift|ethical gates?|\bthe gates?\b"),
    ("fusion", r"\bfusion\b"),
    ("contamination", r"\bcontamination\b"),
    ("conformal", r"\bconformal\b"),
)


def _glossary_term(bare: str) -> str | None:
    if not _GLOSSARY_LEAD.match(bare):
        return None
    for term, pattern in _GLOSSARY_KEYS:
        if re.search(pattern, bare):
            return term
    return None


def glossary_answer(term: str, facts: dict[str, Any]) -> tuple[str, list[str]]:
    """One paragraph on ``term``, with numbers read from the code paths named.

    ``facts`` carries the live values the caller resolved (AMA version,
    decoupling switch states, manifest size) so the paragraph is about this
    installation, not a generic one.
    """
    bands = ", ".join(f"below {edge:g}x {name}" for edge, name in SEVERITY_BANDS)
    top = SEVERITY_BANDS[-1][0]
    answers: dict[str, tuple[str, list[str]]] = {
        "severity": (
            f"Severity is the peak anomaly score divided by the run's threshold, read against "
            f"the API's bands ({bands}, {top:g}x and above critical; api/server.py "
            f"_classify_severity). A medium run therefore scored between 1.5 and 2.5 times its "
            f"threshold. In my memory a run's severity fraction is that ratio over {top:g}, so "
            f"1.0 is the critical boundary and the percentage I quote is a share of it.",
            ["What was the latest anomaly?", "What is the threshold?", "Recommend an action"],
        ),
        "threshold": (
            f"The threshold is the score a sample must exceed to be flagged. The core "
            f"/detect/univariate route derives it from the request's sensitivity as "
            f"{THRESHOLD_FORMULA} (sensitivity 0.5 gives 3.5; api/server.py), and the batch "
            f"route uses the same formula. Each run's own threshold is stored with the run, "
            f"so 'What threshold did you use?' answers from the record.",
            ["What threshold did you use?", "What is a z-score?", "What is severity?"],
        ),
        "z-score": (
            "A z-score is how many standard deviations a value sits from the mean of its "
            "series: (x - mean) / std. The statistical detector (manifest entry 'statistical', "
            "detectors/statistical.py) scores samples with z-score, percentile and MAD "
            "methods, so a score of 6.2 means roughly six standard deviations from the mean.",
            ["What is the threshold?", "What is severity?", "Which detectors do you have?"],
        ),
        "anomaly score": (
            "The anomaly score is the detector's per-sample output before any threshold is "
            "applied; for the statistical detector it is in standard-deviation units and is "
            "not bounded to [0, 1]. The peak score of a run is what I compare with the "
            "threshold to give a severity band.",
            ["What is severity?", "What was the latest anomaly?"],
        ),
        "confidence": (
            "Confidence is a detector's own statement of how sure it is of a verdict. The "
            "core detection routes report none, so I say 'not reported' rather than filling a "
            "number in; the narrative engine then marks the finding unverified and recommends a "
            "second detector before acting. The confidence on my own replies is how directly "
            "the answer came from the record or the code: 1.0 for those, lower for a free-text "
            "match.",
            ["How confident are you?", "Explain uncertainty sources"],
        ),
        "ama": (
            "AMA Cryptography is Mercury's mandatory cryptographic core (omni_mercury_engine/"
            "_pqc_gate.py): the engine refuses to import unless its native library loads with "
            "ML-DSA-65 signatures, Kyber-1024 key encapsulation and SPHINCS+. It also supplies "
            "the HMAC, HKDF, SHA-2/SHA-3 and constant-time primitives Mercury's operational "
            "paths use instead of OpenSSL."
            + (f" Installed: version {facts['ama_version']}." if facts.get("ama_version") else "")
            + " Operators can switch the optional AMA-backed surfaces off with "
            "`mercury-agent decouple ama-optional off`; the core gate is not a switch.",
            ["What are your capabilities?", "What is decoupling?"],
        ),
        "findoyou": (
            "FINDΩYOU™ is a kin application for missing-person work. Mercury's bridge "
            "(integrations/findoyou_bridge.py) produces signed three-state image evidence for "
            "it: a verdict of supported, refuted or undecided, signed with AMA."
            + (
                f" On this machine the bridge is {'enabled' if facts.get('findoyou_enabled') else 'switched off by the operator'}."
                if "findoyou_enabled" in facts
                else ""
            ),
            ["What are your capabilities?", "What is decoupling?"],
        ),
        "decoupling": (
            "Decoupling switches (omni_mercury_engine/decouple.py) let an operator pull the "
            "kin systems apart: 'findoyou' turns the FINDΩYOU™ evidence bridge off and "
            "'ama-optional' turns off the AMA-backed surfaces Mercury can run without. State "
            "lives in ~/.config/omni_mercury/decouple.json or a MERCURY_DECOUPLE_<NAME> "
            "variable; `mercury-agent decouple status` shows it."
            + (f" Current state: {facts['switches']}." if facts.get("switches") else ""),
            ["What is AMA?", "What is FINDOYOU?"],
        ),
        "detector": (
            f"A detector is one registered scoring method in the detector manifest "
            f"(core/detector_registry.py), with a family, an input type and the data shape its "
            f"seams were proven on. The manifest holds {facts.get('manifest_size', '?')} of "
            f"them; `mercury-agent detect --detector` chooses among the engine's base set.",
            ["Which detectors do you have?", "Which detectors handle images?"],
        ),
        "neurosymbolic": (
            "Neurosymbolic detection pairs the neural detectors with a symbolic reasoning layer "
            "that applies rules to their outputs and records a reasoning chain. The API route "
            "is POST /api/v1/detect/neurosymbolic; the batch route does not offer it.",
            ["Which detectors do you have?", "Explain the latest detection"],
        ),
        "gate": (
            "Two hard ethical gates run before I communicate anything: the harm-uplift gate "
            "(cognitive/ethical_bounding.py), which refuses requests that would provide "
            "uplift toward harm, and the σ_Immutable gate (security/sigma_immutable_gate.py), "
            "a trained network over a signed corpus that must score above its domain "
            "threshold. Both fail closed: if either refuses or cannot be evaluated, the "
            "operation halts instead of degrading.",
            ["Why did it clear the gate?", "What are your capabilities?"],
        ),
        "fusion": (
            "Fusion is the neural path that combines the base detectors' outputs into one "
            "verdict (`mercury-agent detect --detector fusion`, POST /api/v1/detect/fusion), "
            "trained by `mercury-agent train` on labelled features from `build-features`.",
            ["Which detectors do you have?", "What should I run next?"],
        ),
        "contamination": (
            "Contamination is the share of samples assumed anomalous when a detector sets its "
            "operating point without labels; `mercury-agent tier-detect --contamination` and "
            "the tier route take it as a fraction.",
            ["What is the threshold?", "What should I run next?"],
        ),
        "conformal": (
            "Conformal control calibrates a detector's alarm so the false-alarm rate on held-out "
            "data stays at or below a chosen level (`tier-detect --conformal-alpha`); the "
            "visual analyzer applies it to image anomalies.",
            ["What is contamination?", "Which detectors handle images?"],
        ),
    }
    return answers[term]


# --------------------------------------------------------------------------- #
# Record wording
# --------------------------------------------------------------------------- #


def relative_time(timestamp: Any) -> str:
    """Relative time in plain words, from an epoch-seconds timestamp."""
    if not isinstance(timestamp, (int, float)):
        return "at an unrecorded time"
    delta = max(0.0, time.time() - float(timestamp))
    if delta < 60:
        return "just now"
    if delta < 3600:
        minutes = int(delta // 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    if delta < 86400:
        hours = int(delta // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(delta // 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


def parse_window(text: str) -> tuple[str, float | None, float | None]:
    """Map the time words in a question to ``(label, since, until)`` in epoch seconds.

    ``until`` is ``None`` for windows that run up to now; "yesterday" ends at
    the start of today, so today's runs are not counted twice.
    """
    now = time.time()
    day_start = now - (now % 86400.0)
    if "hour" in text:
        return "the last hour", now - 3600.0, None
    if "today" in text:
        return "today (UTC)", day_start, None
    if "yesterday" in text:
        return "yesterday (UTC)", day_start - 86400.0, day_start
    if "week" in text:
        return "the last 7 days", now - 7 * 86400.0, None
    if "month" in text:
        return "the last 30 days", now - 30 * 86400.0, None
    return "all recorded runs", None, None


def within(entries: Sequence[dict[str, Any]], until: float | None) -> list[dict[str, Any]]:
    """Drop entries recorded at or after ``until`` (``None`` keeps all)."""
    if until is None:
        return list(entries)
    return [e for e in entries if float(e.get("timestamp") or 0.0) < until]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def describe_detection(record: dict[str, Any], timestamp: Any) -> str:
    """One sentence about a recorded run, built only from the fields it carries."""
    method = str(record.get("method") or "detection")
    count = record.get("anomaly_count")
    total = record.get("n_samples")
    score = _number(record.get("anomaly_score"))
    threshold = _number(record.get("threshold"))
    indices = record.get("indices") or []
    label = record.get("severity_label")

    parts = [f"{method} run {relative_time(timestamp)}:"]
    if record.get("scope") == "series":
        verdict = "anomalous" if record.get("anomaly_detected") else "clean"
        over = f" over {total} sample{'s' if total != 1 else ''}" if isinstance(total, int) else ""
        parts.append(f"whole-series verdict {verdict}{over} (no sample positions);")
    elif isinstance(count, int) and isinstance(total, int) and total > 0:
        flagged = f"{count} of {total} sample{'s' if total != 1 else ''} flagged"
        if indices:
            shown = ", ".join(str(i) for i in indices[:5])
            more = (
                f" and {count - min(len(indices), 5)} more" if count > min(len(indices), 5) else ""
            )
            flagged += f" (sample{'s' if len(indices) > 1 else ''} {shown}{more})"
        parts.append(flagged + ";")
    elif isinstance(count, int):
        parts.append(f"{count} sample{'s' if count != 1 else ''} flagged;")
    if score is not None:
        peak = f"peak score {score:.2f}"
        if threshold is not None and threshold > 0:
            peak += f", {score / threshold:.1f}x the {threshold:.2f} threshold"
        if label:
            peak += f" ({label})"
        parts.append(peak + ".")
    sentence = " ".join(parts)
    return sentence[:-1] + "." if sentence.endswith(";") else sentence


def detection_result_from_record(record: dict[str, Any], timestamp: Any) -> dict[str, Any]:
    """The detection result the narrative engine reads, built from a memory record."""
    count = record.get("anomaly_count")
    total = record.get("n_samples")
    threshold = _number(record.get("threshold"))
    score = _number(record.get("anomaly_score"))
    steps: list[str] = [
        f"method: {record.get('method') or 'detection'}",
        f"recorded {relative_time(timestamp)}",
    ]
    if record.get("scope") == "series":
        steps.append(
            f"whole-series verdict: {'anomalous' if record.get('anomaly_detected') else 'clean'}"
            + (f" over {total} samples" if isinstance(total, int) else "")
            + "; no sample positions are identified"
        )
    elif isinstance(count, int) and isinstance(total, int):
        steps.append(f"{count} of {total} samples flagged")
    indices = record.get("indices") or []
    if indices:
        steps.append("flagged samples: " + ", ".join(str(i) for i in indices[:8]))
    if threshold is not None and threshold > 0:
        steps.append(f"threshold: {threshold:.3f}")
        if score is not None:
            steps.append(f"peak score {score:.3f} is {score / threshold:.1f}x the threshold")
    result: dict[str, Any] = {
        "anomaly_detected": bool(record.get("anomaly_detected", False)),
        "anomaly_score": score if score is not None else 0.0,
        "reasoning_chain": steps,
        "summary": describe_detection(record, timestamp),
    }
    severity = _number(record.get("severity"))
    if severity is not None:
        result["severity"] = severity
    return result


# --------------------------------------------------------------------------- #
# Answers built from the record
# --------------------------------------------------------------------------- #


def run_detail(aspect: str, entry: dict[str, Any] | None) -> tuple[str, list[str]]:
    """Answer one aspect (samples, threshold, method, score, when) of the newest run."""
    if entry is None:
        return (
            "There is no recorded run to read that from yet; run detect and ask again.",
            ["What should I run next?", "How do I collect data?"],
        )
    rec, ts = entry["detection"], entry["timestamp"]
    method = str(rec.get("method") or "detection")
    follow = ["Explain the latest detection", "What was the latest anomaly?", "Recommend an action"]
    if aspect == "samples":
        indices = rec.get("indices") or []
        count = rec.get("anomaly_count")
        total = rec.get("n_samples")
        if rec.get("scope") == "series":
            verdict = "anomalous" if rec.get("anomaly_detected") else "clean"
            return (
                f"The newest run ({method}, {relative_time(ts)}) returned one verdict for the "
                f"whole series{f' of {total} samples' if isinstance(total, int) else ''}: "
                f"{verdict}. It does not identify sample positions; a per-sample run "
                "(detect, tier-detect, or the multivariate route) does.",
                follow,
            )
        if not indices:
            return (
                f"The newest run ({method}, {relative_time(ts)}) flagged no sample"
                + (f" of {total}" if isinstance(total, int) else "")
                + ".",
                follow,
            )
        more = (
            f"; {count - len(indices)} more were flagged beyond the first {len(indices)} kept"
            if isinstance(count, int) and count > len(indices)
            else ""
        )
        return (
            f"The newest run ({method}, {relative_time(ts)}) flagged sample"
            f"{'s' if len(indices) > 1 else ''} {', '.join(str(i) for i in indices)}"
            + (f" of {total}" if isinstance(total, int) else "")
            + f"{more}. Positions are zero-based row indices of the data you sent.",
            follow,
        )
    if aspect == "threshold":
        threshold = _number(rec.get("threshold"))
        if threshold is None:
            return (f"The newest run ({method}) recorded no threshold.", follow)
        return (
            f"The newest run ({method}, {relative_time(ts)}) flagged samples whose score "
            f"exceeded {threshold:.3f}. The core route derives that from the request's "
            f"sensitivity as {THRESHOLD_FORMULA}; a sensitivity of 0.5 gives 3.5.",
            follow,
        )
    if aspect == "method":
        return (
            f"The newest run used the {method} method, {relative_time(ts)}"
            + (f" (detection {rec['detection_id']})" if rec.get("detection_id") else "")
            + ".",
            follow,
        )
    if aspect == "score":
        score = _number(rec.get("anomaly_score"))
        threshold = _number(rec.get("threshold"))
        if score is None:
            return (f"The newest run ({method}) recorded no score.", follow)
        text = f"The newest run's peak score was {score:.3f}"
        if threshold:
            text += f", {score / threshold:.2f}x its {threshold:.3f} threshold"
        label = rec.get("severity_label")
        return (text + (f" ({label})." if label else "."), follow)
    return (f"The newest run ({method}) was recorded {relative_time(ts)}.", follow)


def last_runs(entries: Sequence[dict[str, Any]], n: int) -> tuple[str, list[str]]:
    """Number the newest ``n`` runs, newest first."""
    if not entries:
        return (
            "No detection runs are recorded in memory yet. Memory holds the runs made through "
            "this server since it started (and across restarts when MERCURY_AGENT_MEMORY_PATH is set).",
            ["What should I run next?", "How do I collect data?"],
        )
    shown = list(entries[:n])
    lines = [
        f"{i + 1}. {describe_detection(e['detection'], e['timestamp'])}"
        for i, e in enumerate(shown)
    ]
    head = (
        f"Recent detection events, newest first ({len(shown)} of {len(entries)} recorded "
        f"run{'s' if len(entries) != 1 else ''}):"
    )
    return (
        head + " " + " ".join(lines),
        ["Explain the latest detection", "Summarize threat posture", "How many anomalies today?"],
    )


def posture(entries: Sequence[dict[str, Any]]) -> tuple[str, list[str]]:
    """The whole record at a glance: counts, band distribution, newest flagged run."""
    if not entries:
        return (
            "No detection runs are recorded, so there is no posture to summarise: nothing has "
            "been scored through this server yet. Run detect on your data and ask again.",
            ["What should I run next?", "How do I collect data?"],
        )
    flagged = [e for e in entries if e["detection"].get("anomaly_detected", False)]
    bands: dict[str, int] = {}
    for e in flagged:
        label = str(e["detection"].get("severity_label") or "unbanded")
        bands[label] = bands.get(label, 0) + 1
    order = ["critical", "high", "medium", "low", "unbanded"]
    band_text = ", ".join(f"{bands[b]} {b}" for b in order if b in bands) or "none"
    newest_flagged = flagged[0] if flagged else None
    worst = (
        max(flagged, key=lambda e: _number(e["detection"].get("severity")) or 0.0)
        if flagged
        else None
    )
    parts = [
        f"Across {len(entries)} recorded run{'s' if len(entries) != 1 else ''}, {len(flagged)} flagged at "
        f"least one sample and {len(entries) - len(flagged)} were clean. Severity bands of the flagged "
        f"runs: {band_text}."
    ]
    if newest_flagged is not None:
        parts.append(
            "Newest flagged: "
            + describe_detection(newest_flagged["detection"], newest_flagged["timestamp"])
        )
    if worst is not None and worst is not newest_flagged:
        parts.append(
            "Highest severity: " + describe_detection(worst["detection"], worst["timestamp"])
        )
    if not any(b in bands for b in ("critical", "high")):
        parts.append("No run reached the high or critical band.")
    else:
        parts.append(
            "At least one run reached the high or critical band; it is listed above and "
            "'Recommend an action' gives the narrative engine's recommendations for it."
        )
    return (
        " ".join(parts),
        ["Recommend an action", "Explain the latest detection", "How many anomalies today?"],
    )


def compare_windows(
    text: str, fetch: Callable[[float | None], Sequence[dict[str, Any]]]
) -> tuple[str, list[str]]:
    """Counts for the two windows a question names (default today versus yesterday)."""
    now = time.time()
    day_start = now - (now % 86400.0)
    first = "today (UTC)"
    second = "yesterday (UTC)"
    a = [e for e in fetch(day_start)]
    b = [e for e in fetch(day_start - 86400.0) if float(e["timestamp"]) < day_start]
    if "week" in text:
        first, second = "the last 7 days", "the 7 days before"
        a = list(fetch(now - 7 * 86400.0))
        b = [e for e in fetch(now - 14 * 86400.0) if float(e["timestamp"]) < now - 7 * 86400.0]

    def tally(rows: Sequence[dict[str, Any]]) -> str:
        flagged = sum(1 for e in rows if e["detection"].get("anomaly_detected", False))
        return f"{len(rows)} run{'s' if len(rows) != 1 else ''}, {flagged} flagged"

    return (
        f"{first[0].upper()}{first[1:]}: {tally(a)}. {second[0].upper()}{second[1:]}: {tally(b)}.",
        ["Summarize threat posture", "How many anomalies today?", "What was the latest anomaly?"],
    )


# --------------------------------------------------------------------------- #
# Answers built from the code
# --------------------------------------------------------------------------- #


def next_steps(
    entries: Sequence[dict[str, Any]], commands: Mapping[str, Sequence[str]]
) -> tuple[str, list[str]]:
    """Recommend commands for the current state, naming only ones the CLI really has.

    ``commands`` maps each CLI command name to its option names, read from the
    click group at call time.
    """

    def cmd(name: str, *opts: str) -> str | None:
        if name not in commands:
            return None
        avail = set(commands[name])
        return " ".join(["mercury-agent", name, *[o for o in opts if o.split(" ")[0] in avail]])

    flagged = [e for e in entries if e["detection"].get("anomaly_detected", False)]
    steps: list[str] = []
    if not entries:
        first = cmd("detect", "--input <file.csv>")
        if first:
            steps.append(
                f"Nothing has been scored yet, so start with `{first}` (or drop a CSV/JSON file on "
                "the console and press RUN); a 1-D series or a numeric matrix both work."
            )
    elif flagged:
        tier = cmd("tier-detect", "--input <file.csv>", "--contamination 0.1")
        rca = cmd("rca", "--input <file.csv>", "--top-k 3")
        explain = cmd("explain", "--input <file.csv>")
        steps.append(
            "You have flagged runs. To confirm them with a second method, "
            + (
                f"run `{tier}` (the torch-free detector tier with conformal false-alarm control)"
                if tier
                else "run the detector tier"
            )
            + "."
        )
        if rca:
            steps.append(
                f"If the data is multivariate, `{rca}` attributes the anomaly to its root-cause columns."
            )
        if explain:
            steps.append(
                f"`{explain}` explains the decision; in the console, 'Explain the latest detection' does the same from memory."
            )
    else:
        detect = cmd("detect", "--input <file.csv>", "--threshold <value>")
        steps.append(
            "Every recorded run was clean. Either the data is normal or the threshold is high "
            + (f"for it; `{detect}` lowers the bar, " if detect else "; a lower threshold ")
            + "and `tier-detect` with a contamination fraction sets the operating point from the data instead."
        )
    feats = cmd("build-features", "--data <dir>", "--labels <file>")
    train = cmd("train", "--data <features.npz>")
    if feats and train:
        steps.append(f"With labelled data, `{feats}` then `{train}` trains the fusion model on it.")
    if not steps:
        steps.append(
            "Score your data first: drop a CSV or JSON file on the console's COLLECT DATA zone and "
            "press RUN, then ask me about the result."
        )
    return (
        " ".join(steps),
        ["How do I collect data?", "Which detectors do you have?", "What was the latest anomaly?"],
    )


def collect_data(
    domains: Sequence[str], commands: Mapping[str, Sequence[str]]
) -> tuple[str, list[str]]:
    """How data reaches Mercury: the console drop zone, the CLI inputs and the loader domains."""
    build = "build-features" in commands
    text = (
        "Three ways. In the console, drop a CSV or JSON file (up to 10 MB) on COLLECT DATA and "
        "press RUN; the rows are sent to the connected server. From the shell, `mercury-agent "
        "detect --input <file>` scores a file directly"
        + (
            ", and `mercury-agent build-features --data <dir> --labels <file>` turns raw data into a training archive"
            if build
            else ""
        )
        + f". For live sources, the package ships {len(domains)} domain loaders "
        f"(omni_mercury_engine/loaders/): {', '.join(domains)}. Each fetches from its public "
        "source through the egress-policed HTTP client, engineers features, and declares which "
        "columns its labels derive from so benchmarks stay honest."
    )
    return (
        text,
        ["What should I run next?", "Which detectors do you have?", "What are your capabilities?"],
    )


def run_command(
    name: str, commands: Mapping[str, Sequence[str]], text: str
) -> tuple[str, list[str]]:
    """The agent does not execute commands; it hands back the exact one to run."""
    if name not in commands:
        known = ", ".join(sorted(commands))
        return (
            f"`{name}` is not a mercury-agent command. The CLI has: {known}. In the console, type "
            "the command in the terminal and press RUN, or ask me what to run next.",
            ["What should I run next?", "Which detectors do you have?"],
        )
    file_match = re.search(r"\b([\w./-]+\.(?:csv|json|npy))\b", text)
    opts = set(commands[name])
    line = f"mercury-agent {name}"
    if file_match and "--input" in opts:
        line += f" --input {file_match.group(1)}"
    elif file_match and "--data" in opts:
        line += f" --data {file_match.group(1)}"
    return (
        f"I do not execute commands, but this is the one: `{line}`. In the console, paste it in the "
        "terminal and press RUN (detect, tier-detect, rca, hazard-viz and physics run on the connected "
        "server; the others are prepared for your shell). I will see the result in memory afterwards.",
        ["What was the latest anomaly?", "What should I run next?"],
    )


def gate_explanation(latest: dict[str, Any] | None) -> tuple[str, list[str]]:
    """What the two hard gates are and that this reply itself passed them."""
    text = (
        "Every reply and every narration passes two hard ethical gates before it is spoken, and "
        "this answer did too. The harm-uplift gate (cognitive/ethical_bounding.py) scores the "
        "request itself for uplift toward harm and refuses when it finds it; the σ_Immutable gate "
        "(security/sigma_immutable_gate.py) runs a trained network over a signed corpus and must "
        "score above its domain-calibrated threshold. Both fail closed: a refusal, or a gate that "
        "cannot be evaluated, halts the operation rather than degrading it. A detection clears the "
        "gate when its request carries no harm uplift and the σ_Immutable score is above threshold "
        "for its domain; nothing about the anomaly score itself decides that."
    )
    if latest is not None:
        text += " The newest recorded run was " + describe_detection(
            latest["detection"], latest["timestamp"]
        )
    return (
        text,
        ["What is the gate?", "Explain the latest detection", "What are your capabilities?"],
    )


def confidence_answer(latest: dict[str, Any] | None) -> tuple[str, list[str]]:
    """What confidence means here and what the newest run reported."""
    text = (
        "My answers about the record and the code carry confidence 1.0 because they are read, not "
        "inferred; a free-text match carries the match's relevance instead. The detectors behind "
        "the core routes report no confidence of their own, so I never state one for a run: the "
        "severity band (peak score against threshold) is the strength signal, and the narrative "
        "engine marks such findings unverified until a second detector agrees."
    )
    if latest is not None:
        rec = latest["detection"]
        label = rec.get("severity_label")
        text += " The newest run: " + describe_detection(rec, latest["timestamp"])
        if label:
            text += f" Its band is {label}; its confidence is not reported."
    return (text, ["Explain uncertainty sources", "What is severity?", "Recommend an action"])


def identity(version: str, facts: dict[str, Any]) -> tuple[str, list[str]]:
    """Who the agent is, from the package and this installation."""
    text = (
        f"I am Mercury Agent, version {version}: the conversational surface of the Omni Mercury "
        f"Engine, a multi-domain anomaly and hazard detection system. I answer from this server's "
        f"detection record and from its code (a manifest of {facts.get('manifest_size', '?')} "
        f"detectors, {facts.get('loader_count', '?')} domain loaders, the self-model of what is "
        f"installed), and I narrate detections through the narrative engine behind two hard "
        f"ethical gates. I do not execute commands or invent numbers; ask 'what can you do?' for "
        f"the questions I answer."
    )
    return (
        text,
        ["What can you do?", "What was the latest anomaly?", "Which detectors do you have?"],
    )


def fallback_menu(query: str) -> tuple[str, list[str]]:
    """Said when neither the record, the code nor the free-text search has an answer."""
    text = (
        f"I have nothing in the detection record or the code that answers '{query.strip()}'. I can "
        "tell you what ran and when (latest anomaly, flagged samples, threshold, how many today), "
        "explain a run through the narrative engine, describe the detectors and capabilities this "
        "server has, define Mercury's terms (severity, threshold, z-score, AMA, FINDΩYOU, the gate), "
        "and say what to run next or how to collect data."
    )
    return (
        text,
        ["What can you do?", "What was the latest anomaly?", "Which detectors do you have?"],
    )
