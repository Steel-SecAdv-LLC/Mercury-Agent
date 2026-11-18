# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The single fail-closed choke point every public decision surface passes through.

Mercury used to enforce a *benevolence float* at its decision boundaries: a
keyword/context score that had to clear ``0.99`` before a detection was allowed
to run.  Two things were wrong with that control, and both are fixed here.

1. **It measured the wrong thing.**  The score was computed over a *fixed
   string* the engine synthesised for itself
   (``"anomaly_detection:{domain}:audit verify protect research evidence fair
   oversight monitor data care help support"``) — a keyword salad chosen so the
   gate would pass.  The caller's actual request never reached the scorer, so
   the number could not discriminate anything.  The control was theatre.
2. **A high pass-bar on a benign-data score is not a harm control.**  It
   rejected benign inputs whose vocabulary happened to be plain, and it let any
   input through whose vocabulary happened to be positive.

The enforced control is now the two-axis (hazard-domain × operational-intent)
**harm-uplift gate** documented in ``docs/HARM_POLICY.md`` and implemented by
:func:`~omni_mercury_engine.cognitive.ethical_bounding.assess_weapons_uplift`.
It is scored over the **real decision** — the surface being called, the caller's
domain hint, and the caller's actual request/payload — never over a synthetic
string.  Its polarity is *block on harm*: a benign decision is permitted because
no harm evidence was found, not because it scored highly on a positivity
lexicon.

Fail-closed, in three places
----------------------------

* :func:`assess_weapons_uplift` already fails closed internally: any internal
  error yields ``HARD_REFUSE`` rather than an exception or a silent ALLOW.
* :func:`enforce_decision_boundary` fails closed *around* it: if building the
  subject, running the gate, or auditing the verdict raises anything at all,
  the boundary raises
  :class:`~omni_mercury_engine.cognitive.ethical_bounding.EthicalConstraintViolationError`
  with ``check="harm_uplift"``.  There is no path through this function that
  returns normally on an error.
* There is **no flag, environment variable, or keyword argument that disables
  the gate.**  Removing the call is the only way to bypass it, and the
  capability-contract registry (:data:`~omni_mercury_engine.agentic.capabilities.contract.CONTRACT_MARKER`)
  makes that removal a CI failure — see ``tests/pillars/test_non_maleficence.py``.

Benevolence is retained as an **advisory** signal only: when a scorer is
supplied the boundary computes it, attaches it to the verdict, and logs it.  It
never decides.

Determinism
-----------

:func:`enforce_decision_boundary` is a pure function of the subject: the same
decision produces the same :class:`BoundaryVerdict` on every surface.  That is
what lets ``tests/pillars/test_non_maleficence.py`` assert "identical decision →
identical verdict" across ``detect`` / ``detect_batch`` / ``detect_biometric`` /
``detect_security_threat`` / fleet dispatch.
"""

from __future__ import annotations

import collections
import dataclasses
import io
import logging
import re
import socket
import threading
import types
import unicodedata
from array import array
from collections import UserString, deque
from collections.abc import (
    Mapping as _Mapping,
    Sequence as _Sequence,
)
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, NoReturn

from omni_mercury_engine.cognitive.ethical_bounding import (
    USE_SHIPPED_CLASSIFIER,
    EthicalConstraintViolationError,
    HazardDomain,
    OperationalIntent,
    WeaponsDisposition,
    WeaponsRiskAssessment,
    assess_weapons_uplift,
    sanitize_domain,
)
from omni_mercury_engine.cognitive.gate_audit import record_gate_decision

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

#: Size of one passage handed to the harm gate, and of an audited record.
#:
#: It is NOT a bound on what the gate reads. It used to be: the gate scored one
#: string cut at this length, on the stated ground that "a real uplift request
#: cannot hide past it". It could. Each of the pillar suite's own red-team
#: requests was permitted behind 4,640 characters of weather small-talk. The
#: gate now reads all of the caller's text: once whole, and, where that
#: reading finds something, as overlapping passages of this size
#: (:func:`enforce_decision_boundary`).
MAX_SUBJECT_CHARS = 4096

#: How much caller text one decision may carry. Past it the gate does not
#: guess: the decision is refused, fail-closed, like any other it cannot
#: evaluate. Reading it whole is about 1.1 ms per thousand characters, so
#: this bounds a decision with nothing in it at roughly one second; one with
#: hazard vocabulary in it is read again as passages, about 2.5 ms per
#: thousand more.
MAX_ASSESSED_CHARS = 1_048_576

#: How many values of a structured payload the gate will walk looking for
#: text before it refuses to decide. A numeric array is not walked at all
#: (its shape is its summary), so this is reached only by very large plain
#: Python containers.
MAX_ASSESSED_NODES = 5_000_000

#: Nesting the walk follows before it refuses to decide.
_MAX_TEXT_DEPTH = 64

#: The machinery of the process: what these hold is not the caller's, and one
#: of them leads to all the others of its kind (a logger to every logger).
_PROCESS_STATE: tuple[type, ...] = (
    logging.Logger, logging.Handler, logging.Manager, logging.PlaceHolder, logging.Formatter,
    threading.Thread, type(threading.Lock()), type(threading.RLock()), threading.Event,
    threading.Condition, io.IOBase, socket.socket,
)  # fmt: skip

#: How often one thing is read over. A label of a letter or two is read
#: this many times (a word kept a letter to a cell repeats its letters), and
#: of a run of equal elements in a list this many are read: the rest of a
#: column of 300,000 "ok" says nothing the first of them did not.
_REPEATS = 16

#: A label this short is never passed over for holding no word.
_SHORT_STRING = 3


class _Labels(list[Any]):
    """The cells of a table: an array, a column of a frame, its row labels.

    Read as labels, each distinct one once: a column of two million names of
    forty machines is forty names. What a caller hands over in lists and
    records is not read this way.
    """


#: Upper bound on the rendered summary of a single structured payload value.
_MAX_VALUE_CHARS = 512

#: Number of mapping keys rendered into a payload summary before eliding.
_MAX_KEYS = 24

#: Upper bound on the whole payload summary. The gate scores ``describe()``,
#: whose caller-controlled ``request`` must always survive the
#: :data:`MAX_SUBJECT_CHARS` cut. Without a total cap a structured payload can
#: render to tens of kilobytes and, placed before ``request``, evict it from the
#: scored string entirely -- a harm-uplift bypass (large benign payload +
#: trailing uplift ``request`` -> only the benign padding is scored). The cap
#: leaves at least ``MAX_SUBJECT_CHARS - _MAX_PAYLOAD_SUMMARY_CHARS`` for the
#: request, and ``describe()`` additionally scores the request FIRST.
_MAX_PAYLOAD_SUMMARY_CHARS = 2048

#: Recursion ceiling for :func:`summarize_payload`. Deep enough that a string
#: nested a few containers down (``{"a": {"b": {"c": "<uplift text>"}}}``) still
#: reaches the gate's lexicons instead of collapsing to the token ``dict`` -- the
#: old depth-2 ceiling hid exactly that. The total-character cap above, not the
#: depth, is what bounds cost, so this can be generous.
_MAX_PAYLOAD_DEPTH = 6


#: Types that are numbers and nothing else (a subclass could carry text).
_NUMBERS: frozenset[type] = frozenset({int, float, bool, complex, type(None)})

#: Plain containers of values, a dict's keys, values and items as views among them.
_SEQUENCES: tuple[type, ...] = (
    list, tuple, set, frozenset, deque,
    type({}.keys()), type({}.values()), type({}.items()),
)  # fmt: skip

#: A letter that a date, a time or a number is not written with. A string
#: without one (``2026-11-18T10:00:00Z``, an id of digits, ``1.5e-3``) has no
#: word in it for the gate to read.
#:
#: This was "two letters together", which also passed over text written one
#: letter at a time: ``s a r i n`` in a payload went unread, though the
#: normalizer exists to put exactly that back together.
_WORD = re.compile(r"[^\W\d_TtZzEe]")

#: A run of printable ASCII inside bytes that are not text: what ``strings``
#: would show of them, at a length a phrase has and chance seldom does. In a
#: buffer of small numbers two bytes in three are printable ones; at six in
#: a row a quarter of such a buffer was "text".
_PRINTABLE_RUN = re.compile(rb"[\x20-\x7e]{16,}")


class UnassessableSubjectError(ValueError):
    """The decision carries more than the gate can read; it must be refused."""


def _object_text(payload: Any) -> list[Any] | None:
    """What an object that is not a plain container holds, or ``None`` if nothing is read.

    A float matrix has no words in it and is summarised by its shape. An array
    of strings or objects, the text columns of a data frame, the fields of a
    dataclass or a pydantic model, the attributes of a plain object, the
    arguments of an exception, a path and the name and value of an enum
    member are the caller's words in another container. Summarised by shape
    or by type name they went unread: the reasoning backend hands the gate a
    ``ReasoningContext``, which it saw as the single word ``ReasoningContext``
    while the evidence inside went on to a language model.

    What is left opaque, and stays its type name as it always was: a class, a
    module, a function (their attributes are code, not what a caller said);
    the machinery of the process itself (:data:`_PROCESS_STATE`: a logger, a
    lock, an open file), which holds nothing a caller wrote and leads to
    everything else of its kind; and a generator or any other one-shot
    iterator, which cannot be read without using it up for the surface it
    was passed to. One of the kinds above that cannot be read the way its
    kind is read is opaque too: this never raises, so that an odd object is
    not a way to have a decision refused.

    An object's attributes are followed as deep as containers are
    (:data:`_MAX_TEXT_DEPTH`) and no deeper. Followed four deep and then
    passed over, a request five objects down went unread; followed eight
    deep and then refused, a linked list of ten was a refusal.

    Returns:
        What it holds; ``None`` when nothing is read.
    """
    try:
        if isinstance(payload, (type, types.ModuleType, *_PROCESS_STATE)):
            return None
        if isinstance(payload, range):
            return []  # numbers
        if isinstance(payload, PurePath):
            return [str(payload)]
        if isinstance(payload, Enum):
            return [payload.name, payload.value]
        if isinstance(payload, array):
            return [payload.tounicode()] if payload.typecode in "uw" else []
        if isinstance(payload, UserString):
            return [str(payload)]
        if hasattr(type(payload), "dtypes") and hasattr(type(payload), "columns"):  # a frame
            return _frame_text(payload)
        dtype: Any = getattr(payload, "dtype", None)
        kind = getattr(dtype, "kind", None)
        if isinstance(kind, str):
            names = getattr(dtype, "names", None)
            if kind == "V" and names:
                # A structured array: only the fields that can hold text, each
                # as the array it is. Converted whole, six million records of
                # numbers were six million tuples to walk, and a refusal.
                held = [payload[name] for name in names if dtype[name].kind in _TEXT_KINDS]
            else:
                held = [_Labels(_as_list(payload))] if kind in _TEXT_KINDS else []
            return held + _index_text(payload)
        if isinstance(payload, _Sequence):
            # As a plain list, which is walked like any other: numbers in one
            # pass, and within the same budget.
            return [list(payload)]
        values: list[Any] = []
        if isinstance(payload, BaseException):
            values.extend(payload.args)
        record = dataclasses.is_dataclass(payload) or any(
            hasattr(type(payload), mark) for mark in ("__pydantic_fields__", "__fields__")
        )
        if callable(payload) and not record:
            # A function, a partial, a mock: code. (A record that can also be
            # called is still a record.)
            return values or None
        # The instance's own attributes, by value (their names are the code's
        # words, not the caller's). Each value once: a dataclass's fields and
        # a model's are in its ``__dict__`` or its slots already, and read
        # from there and again from the fields or a dump they were walked,
        # and counted against the budget, twice.
        attributes = getattr(payload, "__dict__", None)
        if isinstance(attributes, dict):
            values.extend(attributes.values())
        for owner in type(payload).__mro__:
            slots = owner.__dict__.get("__slots__", ())
            for name in (slots,) if isinstance(slots, str) else slots:
                # A double-underscore slot is the class's own bookkeeping (a
                # model's set of field names), not something it was given.
                if not (name.startswith("__") and name.endswith("__")):
                    values.append(getattr(payload, name, None))
        # What a pydantic model holds beside its stored fields.
        for beside in ("__pydantic_extra__", "__pydantic_private__"):
            held_beside = getattr(payload, beside, None)
            if isinstance(held_beside, dict):
                values.append(held_beside)
        computed = getattr(type(payload), "model_computed_fields", None)
        if isinstance(computed, dict):
            values.extend(getattr(payload, name, None) for name in computed)
        return values or None
    except Exception:
        return None


def _frame_text(frame: Any) -> list[Any]:
    """The column names, row labels and text columns of a data frame.

    Columns are taken by position. Taken by name, a frame with two columns of
    one name gave back a frame where a column was expected, the read failed,
    and the whole frame went through as its type name.
    """
    values: list[Any] = [str(name) for name in frame.columns]
    values.extend(_index_text(frame))
    if callable(getattr(type(frame), "get_columns", None)):  # polars
        columns = frame.get_columns()
    else:  # pandas
        columns = [column for _, column in frame.items()]
    for column in columns:
        if getattr(getattr(column, "dtype", None), "kind", "O") in _TEXT_KINDS:
            values.append(_Labels(_as_list(column)))
    return values


#: dtype kinds that can hold text: object, unicode, bytes, the string dtype, and
#: structured records.
_TEXT_KINDS = "OUSTV"


def _index_text(labelled: Any) -> list[Any]:
    """The row labels of a frame or a series, when they can hold text."""
    index = getattr(labelled, "index", None)
    if index is None or callable(index):
        return []
    if getattr(getattr(index, "dtype", None), "kind", "O") not in _TEXT_KINDS:
        return []
    return [_Labels(_as_list(index))]


def _as_list(values: Any) -> list[Any]:
    """An array or a column as a Python list (numpy and pandas ``tolist``, polars ``to_list``)."""
    convert = getattr(values, "tolist", None) or values.to_list
    listed: list[Any] = convert()
    return listed


def _bytes_text(raw: bytes) -> str:
    """The text in ``raw``: every reading that makes text of it, else its printable runs.

    Bytes that are not text used to be decoded as though they were, every
    undecodable byte becoming a replacement character: a megabyte of
    compressed or random data counted as a megabyte of text, was refused for
    its size, and below that size was scored as noise. Text is told from
    data by what it decodes to:

    * UTF-8, when it is that and at most one byte in ten is a control code.
    * Otherwise nothing in the bytes says what they are, and each of these
      that makes text of them is read: Latin-1, when it is mostly ASCII;
      UTF-16 (:func:`_utf16_text`), in the order a byte-order mark declares
      or else in both; UTF-8 with control codes between its words
      (:func:`_words_between_controls`).
    * Where they are not Latin-1, their runs of printable ASCII as well, as
      ``strings`` would give.

    A byte-order mark does not end the reading: two bytes before an ASCII
    request made UTF-16 of it, in which it says nothing.
    """
    declared = _BYTE_ORDER_MARKS.get(raw[:2])
    if declared:
        raw = raw[2:]  # the mark is not text in any reading
    even = raw[: len(raw) - len(raw) % 2]  # a UTF-16 text cut in the middle of a unit
    plain = _is_text(raw)
    if plain and not declared:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            pass
    latin = (
        plain
        and len(raw.translate(None, _LATIN1_TEXT)) * 20 <= len(raw)
        and len(raw.translate(None, _ASCII)) * 10 <= len(raw) * 3
    )
    readings = [
        (
            raw.decode("latin-1")
            if latin
            else " ".join(run.decode("ascii") for run in _PRINTABLE_RUN.findall(raw))
        )
    ]
    if declared:
        readings.append(_utf16_text(even, declared, declared=True))
    elif not latin:
        wide = [_utf16_text(even, order) for order in _BYTE_ORDER_MARKS.values()]
        # Every reading is read, so their order decides nothing but where the
        # passages are cut; of the two byte orders the one with more ASCII is first.
        readings += sorted(wide, key=lambda text: -sum(map(str.isascii, text[:_MAX_VALUE_CHARS])))
        readings.append(_words_between_controls(raw))
    return "\n".join(dict.fromkeys(filter(None, readings)))  # each reading once


_BYTE_ORDER_MARKS = {b"\xff\xfe": "utf-16-le", b"\xfe\xff": "utf-16-be"}
#: Bytes that are text in Latin-1: tab, line ends, and the printable ranges.
_LATIN1_TEXT = bytes([9, 10, 13, *range(0x20, 0x7F), *range(0xA0, 0x100)])
_ASCII = bytes(range(0x80))
#: Bytes no text is made of: the control codes other than tab and the line ends.
_CONTROL = bytes([*range(0, 9), 11, 12, *range(14, 0x20), 0x7F])
_CONTROL_TO_SPACE = bytes.maketrans(_CONTROL, b" " * len(_CONTROL))
_WHITE_SPACE = b"\t\n\r "
#: What words are made of, in UTF-8: ASCII letters and the bytes of any other letter.
_LETTER_BYTES = bytes([*range(0x41, 0x5B), *range(0x61, 0x7B), *range(0x80, 0x100)])
#: Characters no text is made of: control codes, and code points that are
#: unassigned or for private use (``unicodedata`` general categories).
_NOT_TEXT = frozenset({"Cc", "Cn", "Co", "Cs"})
#: A UTF-16 text shorter than this many units is not recognised without a
#: byte-order mark: any few bytes are a character or two of some alphabet.
_MIN_UTF16_UNITS = 4
#: A text uses few characters and uses them unevenly; numbers read as
#: UTF-16 do neither. Half of a text is made of its this many commonest
#: characters (Chinese, which uses most, has about three fifths of a text
#: in its 256 commonest; 16-bit values up to 1,000 have a quarter)...
_COMMONEST = 256
#: ...and in a text of :data:`_LONG_ENOUGH` characters or more, the
#: commonest is used at least this many times as often as the average one
#: (a space in English: eleven times; counts from 0 to 9, which are a
#: hundred characters used alike: under twice). A shorter text is too
#: short to tell, and is read.
_UNEVEN = 3
_LONG_ENOUGH = 512


def _is_text(raw: bytes) -> bool:
    """Whether ``raw`` could be text: at most one byte in ten is a control code.

    A buffer of small numbers decodes without error in more than one
    encoding (as UTF-8, or as UTF-16 when it holds 16-bit values below 256)
    and was then a megabyte of "text" to count and score.
    """
    return (len(raw) - len(raw.translate(None, _CONTROL))) * 10 <= len(raw)


def _words_between_controls(raw: bytes) -> str:
    """``raw`` as UTF-8 without its control codes, when what is left is words.

    A field padded with NULs to a fixed width, values with a unit separator
    between them, a request with a control code between its letters: more
    than a tenth control codes, so not text by that count, and all of it
    words. A NUL is padding and is dropped; any other control code stood
    between two things and is read as a space. Small numbers are valid UTF-8
    as well, and between their control codes are digits and punctuation as
    often as letters: three in four of what is left must be letters.
    """
    marks = raw.translate(None, _CONTROL + _WHITE_SPACE)
    if not marks or len(marks.translate(None, _LETTER_BYTES)) * 4 > len(marks):
        return ""
    try:
        return raw.replace(b"\x00", b"").translate(_CONTROL_TO_SPACE).decode("utf-8").strip()
    except UnicodeDecodeError:
        return ""


def _utf16_text(even: bytes, order: str, *, declared: bool = False) -> str:
    """``even`` read as UTF-16 in ``order``, or ``""`` when that does not make text of it.

    It is text when at most one character in ten is a control code, half of
    a surrogate pair (a buffer of numbers soon holds one) or a code point
    nothing is assigned to; and, where no byte-order mark ``declared`` it,
    when it is written as text is: in few characters (:data:`_COMMONEST`),
    unevenly used (:data:`_UNEVEN`). A text in one byte order is as a rule
    also text in the other, in an alphabet it was not written in, and
    nothing in the bytes says which was meant: both are read and both are
    counted.

    What this does not read is a text padded with thousands of different
    characters, which is what numbers look like. Nothing tells the two apart.
    """
    if len(even) < 2 * _MIN_UTF16_UNITS:
        return ""
    text = even.decode(order, errors="replace")
    counts = collections.Counter(text)
    spoiled = sum(
        count
        for ch, count in counts.items()
        if ch == "\ufffd" or (ch not in "\t\n\r" and unicodedata.category(ch) in _NOT_TEXT)
    )
    if spoiled * 10 > len(text):
        return ""
    if declared:
        return text
    commonest = counts.most_common(_COMMONEST)
    if sum(count for _, count in commonest) * 2 < len(text):
        return ""
    if len(text) >= _LONG_ENOUGH and commonest[0][1] * len(counts) < _UNEVEN * len(text):
        return ""
    return text


def payload_text(payload: Any) -> list[str]:
    """Every piece of text in ``payload``, however deep and however far along.

    :func:`summarize_payload` is a bounded description for the audit record:
    the first 24 keys, the first 512 characters of a string, six levels. The
    gate cannot decide on a description that leaves text out, so it reads this
    instead: every string value, every string key, bytes that are text (and
    the printable runs of bytes that are not), the string cells of arrays and
    data frames, and what :func:`_object_text` reads of an object. Numbers
    carry no words and are skipped.

    Raises:
        UnassessableSubjectError: When the payload holds more text than
            :data:`MAX_ASSESSED_CHARS`, more values than
            :data:`MAX_ASSESSED_NODES`, or nests deeper, in containers or in
            objects, than :data:`_MAX_TEXT_DEPTH`.
    """
    found: list[str] = []
    chars = 0
    nodes = 0
    # Containers already read, by identity -- and held, not just their ids. A
    # set of bare ids was wrong: the dict a model dumps to and the list an
    # array converts to are made here and freed as soon as they are read, the
    # next one is given the same address, and it was skipped as "already
    # seen". A payload of two models or two string arrays was read as the
    # first alone, and whatever the second said went through unread.
    seen: dict[int, Any] = {}
    read: dict[str, int] = {}
    # Each value, how deep it lies, whether it is a label (read once,
    # however often it occurs), and whether it is inside a table.
    stack: list[tuple[Any, int, bool, bool]] = [(payload, 0, False, False)]
    while stack:
        value, depth, label, table = stack.pop()
        nodes += 1
        if nodes > MAX_ASSESSED_NODES:
            raise UnassessableSubjectError(
                f"the payload holds more than {MAX_ASSESSED_NODES:,} values; "
                "pass numeric data as an array"
            )
        if value is None or isinstance(value, (bool, int, float, complex)):
            continue
        if isinstance(value, (bytes, bytearray, memoryview)):
            value = _bytes_text(bytes(value))
        if isinstance(value, str):
            wanted = _holds_a_word(value)
            if wanted and label:
                times = read.get(value, 0)
                read[value] = times + 1
                wanted = times < (_REPEATS if len(value) <= _SHORT_STRING else 1)
            if wanted:
                chars += len(value)
                if chars > MAX_ASSESSED_CHARS:
                    raise UnassessableSubjectError(
                        f"the payload holds more than {MAX_ASSESSED_CHARS:,} characters of text"
                    )
                found.append(value)
            continue
        if depth >= _MAX_TEXT_DEPTH:
            raise UnassessableSubjectError(
                f"the payload nests deeper than {_MAX_TEXT_DEPTH} levels"
            )
        if id(value) in seen:  # a container reached twice is read, and counted, once
            continue
        seen[id(value)] = value
        children: list[tuple[Any, bool]]
        if isinstance(value, _Mapping):
            # A key is read beside what it names, each time: the field "how
            # to make" was read once for 400 records, and the record that
            # named an agent under it was permitted. Beside a number it names
            # nothing new, and is a label: the keys of 45,000 records of
            # readings are three strings, and counted each time they filled
            # the reading budget and a table of numbers was refused.
            children = [
                part
                for key, held in value.items()
                for part in ((key, table or _says_nothing(held)), (held, table))
            ]
        elif isinstance(value, _SEQUENCES):
            # A sequence of nothing but numbers has no words in it. Its types
            # are checked in one pass at C speed instead of visiting every
            # element here (a 1,000 x 1,000 list of floats cost 170 ms a
            # decision); eight of them count as one value walked.
            items: Any = value
            if set(map(type, items)) <= _NUMBERS:
                nodes += len(items) // 8
                continue
            table = table or isinstance(value, _Labels)
            # What a caller hands over in a list is read in the order given,
            # every element: "how", "to" and "make" came round again further
            # down a list of words, were passed over as read already, and the
            # request they opened was permitted.
            children = [(item, table) for item in (items if table else _runs_cut(items))]
        else:
            held_by_object = _object_text(value)
            if held_by_object is None:
                continue
            children = [(item, table) for item in held_by_object]
        stack.extend((child, depth + 1, as_label, table) for child, as_label in reversed(children))
    return found


def _holds_a_word(text: str) -> bool:
    """Whether ``text`` could hold a word (see ``_WORD``).

    60,000 timestamps are no text at all. A string of a letter or two is not
    passed over for holding no word: "n", "e", "r", "v", "e" is one.
    """
    if len(text) <= _SHORT_STRING:
        return any(char.isalpha() for char in text)
    return _WORD.search(text) is not None


def _says_nothing(value: Any) -> bool:
    """Whether ``value`` is a number, or a string no word could be in."""
    if isinstance(value, str):
        return not _holds_a_word(value)
    return value is None or isinstance(value, (bool, int, float, complex))


def _runs_cut(items: Any) -> list[Any]:
    """``items`` as a caller's list is read: letters joined, long runs of one thing cut.

    Elements of one character each are a string handed over a letter at a
    time, spaces and all, and are joined as ``"".join`` would join them.

    A run of equal elements is cut to its first :data:`_REPEATS`. What is
    left out stood next to its like, which is read: nothing that was beside
    something else is beside it no longer.
    """
    kept: list[Any] = []
    letters: list[str] = []
    before: Any = _runs_cut  # equal to no element
    run = 0
    for item in items:
        if type(item) is str and len(item) == 1:
            letters.append(item)
            continue
        if letters:
            kept.append("".join(letters))
            letters.clear()
            before = _runs_cut
        alike = type(item) is type(before) and isinstance(item, (str, dict, list, tuple))
        run = run + 1 if alike and item == before else 0
        if run < _REPEATS:
            kept.append(item)
        before = item
    if letters:
        kept.append("".join(letters))
    return kept


def _spans(length: int, size: int) -> list[tuple[int, int]]:
    """``[start, end)`` of the passages of ``size`` that cover ``length``, overlapping by half.

    Any stretch up to half a passage long lies whole inside at least one of
    them, so a request is not cut in two at every boundary it meets.
    """
    stride = max(size // 2, 1)
    last_start = max(length - (size - stride), 1)
    return [(start, min(start + size, length)) for start in range(0, last_start, stride)]


def summarize_payload(payload: Any, *, _depth: int = 0) -> str:
    """Render ``payload`` as the factual description of what is being decided on.

    The point is to surface the caller's *real* input to the gate without
    materialising megabytes of telemetry:

    * arrays/tensors become ``ndarray(shape=..., dtype=...)`` — structure only,
      because a float matrix carries no lexical harm evidence;
    * strings are passed through verbatim (truncated), because a string payload
      *is* the request;
    * mappings and sequences are walked one level so a domain payload that
      carries text (``{"query": "...", "notes": "..."}``) reaches the gate
      instead of being reduced to ``dict``.

    Args:
        payload: Any caller-supplied input.
        _depth: Internal recursion guard.

    Returns:
        A bounded, human-readable description.  Never raises: an object whose
        ``repr`` explodes degrades to its type name, since a summariser that
        could raise would be a way to break the control it feeds.
    """
    try:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload[:_MAX_VALUE_CHARS]
        if isinstance(payload, (bool, int, float)):
            return str(payload)
        shape = getattr(payload, "shape", None)
        if shape is not None:
            dtype = getattr(payload, "dtype", None)
            return f"{type(payload).__name__}(shape={tuple(shape)}, dtype={dtype})"
        if _depth >= _MAX_PAYLOAD_DEPTH:
            return type(payload).__name__
        if isinstance(payload, dict):
            keys = list(payload)[:_MAX_KEYS]
            rendered = ", ".join(
                f"{k}={summarize_payload(payload[k], _depth=_depth + 1)}" for k in keys
            )
            elided = "" if len(payload) <= _MAX_KEYS else f", +{len(payload) - _MAX_KEYS} more"
            return f"{{{rendered}{elided}}}"
        if isinstance(payload, (list, tuple, set, frozenset)):
            items = list(payload)[:_MAX_KEYS]
            rendered = ", ".join(summarize_payload(v, _depth=_depth + 1) for v in items)
            return f"[{rendered}]"
        return type(payload).__name__
    except Exception:  # pragma: no cover - defensive; a summariser must not raise
        return type(payload).__name__


@dataclass(frozen=True)
class DecisionSubject:
    """The real decision a public surface is about to make.

    Every field is sourced from the actual call — there is deliberately no way
    to construct a subject out of canned positive keywords, which is what the
    superseded ``_enforce_ethics_at_boundary`` action string did.

    Attributes:
        surface: Fully-qualified name of the public surface (e.g.
            ``"OmniMercuryEngine.detect"``).  Audit provenance.
        operation: A short factual statement of what the surface does with the
            input (e.g. ``"analyse request payload for injection/XSS threats"``).
            This is a *description of the code path*, not a claim about the
            caller's intent, and it is fixed per surface so it cannot be tuned
            per call to move a verdict.
        domain: Caller-supplied domain hint, collapsed to the whitelisted
            alphabet by
            :func:`~omni_mercury_engine.cognitive.ethical_bounding.sanitize_domain`
            so a hostile hint cannot inject either harm or allow keywords.
        request: The caller's own free text, verbatim (a query, a payload, a
            file path).  Empty, or ``None``, when the surface takes no text.
        payload: The caller's structured input; summarised by
            :func:`summarize_payload`.
        context: Extra structured provenance passed to the gate (e.g.
            ``licensed_context``) and to the audit record.
    """

    surface: str
    operation: str
    domain: str = "general"
    request: str = ""
    payload: Any = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        """Render the decision as the text the harm gate scores.

        Returns:
            ``"<surface>: <operation> | domain=... | request=... | input=..."``,
            truncated to :data:`MAX_SUBJECT_CHARS`.

        The ``request`` -- the caller's own free text, and the highest-signal
        harm evidence -- is placed **before** the structured payload summary, and
        the summary is capped at :data:`_MAX_PAYLOAD_SUMMARY_CHARS`. Both matter:
        the caller controls both fields, and a payload that renders to tens of
        kilobytes would otherwise fill the whole :data:`MAX_SUBJECT_CHARS` budget
        and push a trailing uplift ``request`` off the end -- so the gate would
        score only the benign padding and permit the decision. Ordering the
        request first, and bounding the summary, guarantees the request is always
        scored.
        """
        parts = [f"{self.surface}: {self.operation}", f"domain={sanitize_domain(self.domain)}"]
        request = self._request()
        if request:
            parts.append(f"request={request}")
        summary = summarize_payload(self.payload)
        if summary:
            parts.append(f"input={summary[:_MAX_PAYLOAD_SUMMARY_CHARS]}")
        return " | ".join(parts)[:MAX_SUBJECT_CHARS]

    def _request(self) -> str:
        """The request as text. ``None`` is no request, as it was before the gate measured it."""
        if isinstance(self.request, str):
            return self.request
        return "" if self.request is None else str(self.request)

    def _reading(self) -> _Reading:
        """Everything the caller said, laid out for :func:`enforce_decision_boundary`.

        Raises:
            UnassessableSubjectError: When the caller's text exceeds what the
                gate will read (:data:`MAX_ASSESSED_CHARS`).
        """
        first = self.describe()
        request = self._request()
        texts = payload_text(self.payload)
        if len(request) + sum(map(len, texts)) > MAX_ASSESSED_CHARS:
            raise UnassessableSubjectError(
                f"the decision carries more than {MAX_ASSESSED_CHARS:,} characters of text"
            )
        heading = f"{self.surface}: {self.operation} | domain={sanitize_domain(self.domain)}"
        if (not request or f"request={request}" in first) and (
            len(texts) <= _MAX_KEYS and all(text in first for text in texts)
        ):
            return _Reading(first=first, lead=heading, body=None)
        said = ", ".join(texts)
        if len(request) > _MAX_VALUE_CHARS:
            # A request this long is read as the payload is: in passages.
            body = f"request={request} | input={said}" if said else f"request={request}"
            return _Reading(first=first, lead=f"{heading} | ", body=body)
        # A short request is what the caller asks of the payload, so it opens
        # every passage of it: "how do I make it" is put to the agent named in
        # the last row as it is to the one named in the first.
        asked = f"request={request} | " if request else ""
        return _Reading(first=first, lead=f"{heading} | {asked}input=", body=said)

    def passages(self) -> list[str]:
        """Everything the caller said, as passages of :data:`MAX_SUBJECT_CHARS`.

        When :meth:`describe` holds all of the caller's text it is the only
        one. When it does not -- the request runs past
        :data:`MAX_SUBJECT_CHARS`, or the payload holds text its bounded
        summary leaves out -- the text is cut into overlapping passages under
        the same ``surface: operation | domain=`` heading, each opening with
        the request when the request is short.

        Raises:
            UnassessableSubjectError: When the caller's text exceeds what the
                gate will read (:data:`MAX_ASSESSED_CHARS`).
        """
        reading = self._reading()
        if reading.body is None:
            return [reading.first]
        return [reading.passage(start, end) for start, end in reading.spans()]


@dataclass(frozen=True)
class _Reading:
    """What the gate scores of one decision.

    Attributes:
        first: :meth:`DecisionSubject.describe`: the bounded description that
            is audited.
        lead: What every passage opens with: the heading, and a short request.
        body: The text that is cut into passages, when ``first`` does not hold
            all the caller said; ``None`` when it does.
    """

    first: str
    lead: str
    body: str | None

    def whole(self) -> str:
        """The decision in one string, everything the caller said included."""
        return self.first if self.body is None else self.lead + self.body

    def spans(self) -> list[tuple[int, int]]:
        """Where in ``body`` each passage lies."""
        return _spans(len(self.body or ""), max(MAX_SUBJECT_CHARS - len(self.lead), 1024))

    def passage(self, start: int, end: int) -> str:
        """``body[start:end]`` under the lead."""
        return self.lead + (self.body or "")[start:end]


#: The response ladder, least restrictive first (``WeaponsDisposition``'s own order).
_RESTRICTION: dict[WeaponsDisposition, int] = {
    disposition: rank for rank, disposition in enumerate(WeaponsDisposition)
}


@dataclass(frozen=True)
class BoundaryVerdict:
    """The permitted outcome of a boundary check.

    A blocking assessment never produces a verdict — it raises — so every
    instance of this class describes a *permitted* decision.

    Attributes:
        subject: The decision that was gated.
        assessment: The harm-uplift assessment that permitted it.
        benevolence: The advisory benevolence score, when a scorer was
            supplied; ``None`` otherwise.  Advisory means advisory: it is
            logged and attached, and it decides nothing.
        requires_provenance: ``True`` when the gate permitted the decision but
            requires any emitted answer to be source-attributed
            (``ALLOW_PROVENANCE``).  Non-blocking, enforced at the output
            boundary.
    """

    subject: DecisionSubject
    assessment: WeaponsRiskAssessment
    benevolence: float | None = None

    @property
    def requires_provenance(self) -> bool:
        """True when the permit is conditional on source attribution."""
        return self.assessment.disposition is WeaponsDisposition.ALLOW_PROVENANCE


def _assess(
    reading: _Reading, context: dict[str, Any], harm_classifier: Any
) -> tuple[str, WeaponsRiskAssessment]:
    """Score one decision: the text to quote for it, and the assessment that stands.

    A decision whose bounded description holds all the caller said is scored
    as that description, as it always was.

    A longer one is read whole first. When that finds no hazard vocabulary
    and no offensive intent anywhere, no part of it can, and the decision is
    permitted there: one reading, where passages take two and a half.

    When it finds something, the verdict is the passages'. The gate pairs a
    hazard with an intent inside the text it is handed, and what a passage
    pairs was said together; what the whole text pairs may be a production
    phrase on one page and a hazard word forty pages on. Measured on 80
    encyclopedia articles (``research/harm_gate_long_text``): pairing at any
    distance refused 7 of 40 on general subjects, each on such a pair, where
    passages refuse none. So cues further apart than a passage are not paired --
    the locality of every sliding-window control in ``docs/HARM_POLICY.md`` --
    with one exception that is not a matter of distance: a short request is
    put to every passage of the payload (:meth:`DecisionSubject._reading`).

    A passage is cut where its length runs out, which can be in the middle of
    a phrase: "make a nerve agent | detector" reads as production of the
    agent. So a passage that blocks is read again with a few words more at
    each end (:data:`_CUT_MARGIN`); if that does not block, the cut did it,
    and the longer reading stands. It is a few words, not the neighbouring
    passages: those were read at first, and anything defensive in eight
    thousand characters around a request then talked its passage down. The
    most restrictive passage decides, and is what a refusal quotes.
    """

    def score(text: str) -> WeaponsRiskAssessment:
        return assess_weapons_uplift(text, context, harm_classifier=harm_classifier)

    if reading.body is None:
        return reading.first, score(reading.first)
    whole = score(reading.whole())
    if _nothing_found(whole):
        return reading.first, whole

    spans = reading.spans()
    worst: tuple[str, WeaponsRiskAssessment] | None = None
    for position, (start, end) in enumerate(spans):
        passage = reading.passage(start, end)
        scored = score(passage)
        if scored.blocks and len(spans) > 1:
            around = reading.passage(
                max(start - _CUT_MARGIN, 0), min(end + _CUT_MARGIN, len(reading.body))
            )
            scored = min(scored, score(around), key=_restriction)
        if worst is None or _restriction(scored) > _restriction(worst[1]):
            worst = (passage, scored)
            if scored.disposition is WeaponsDisposition.HARD_REFUSE:
                break  # nothing later can be more restrictive
    if worst is None:  # pragma: no cover - a body always has a passage
        raise RuntimeError("no passage was scored")
    return worst if worst[1].blocks else (reading.first, worst[1])


#: How much is read beyond each end of a passage that blocks, to finish a
#: phrase its cut may have broken.
_CUT_MARGIN = 256


def _restriction(assessment: WeaponsRiskAssessment) -> int:
    """Where an assessment stands on the response ladder."""
    return _RESTRICTION[assessment.disposition]


def _nothing_found(assessment: WeaponsRiskAssessment) -> bool:
    """No hazard vocabulary and no offensive intent: a plain permit with nothing behind it."""
    return (
        assessment.disposition is WeaponsDisposition.ALLOW
        and assessment.hazard_domain is HazardDomain.NONE
        and not assessment.signals
    )


def _refuse_unassessed(
    subject: DecisionSubject,
    *,
    reason: str,
    unassessable: bool = False,
    cause: BaseException | None = None,
) -> NoReturn:
    """Record and raise the refusal of a decision the gate did not get to assess."""
    surface = f"{subject.surface}:{subject.operation}"
    details: dict[str, Any] = {
        "surface": subject.surface,
        "domain": sanitize_domain(subject.domain),
        "error": reason,
        "fail_closed": True,
    }
    if unassessable:
        details["reason"] = UNASSESSABLE
    record_gate_decision(
        decision="refused",
        source="decision_boundary",
        disposition=WeaponsDisposition.HARD_REFUSE.value,
        signals=("unassessable_subject" if unassessable else "gate_error",),
        reason=reason,
        query=surface,
        extra={"surface": subject.surface, "domain": sanitize_domain(subject.domain)},
    )
    raise EthicalConstraintViolationError(
        action=surface, score=0.0, threshold=1.0, check="harm_uplift", details=details
    ) from cause


#: ``details["reason"]`` of a refusal made because the decision carries more
#: than the gate reads, as against one made on what it read.
UNASSESSABLE = "unassessable"


def refused_for_size(refusal: BaseException) -> str | None:
    """Why the gate could not read a decision, when that is why it refused it.

    Returns:
        The gate's statement of what was too much ("the decision carries more
        than 1,048,576 characters of text"), or ``None`` for a refusal made on
        what the decision said. A route answers the first with 413.
    """
    details = getattr(refusal, "details", None)
    if isinstance(details, dict) and details.get("reason") == UNASSESSABLE:
        return str(details.get("error") or "more than the gate reads")
    return None


def enforce_decision_boundary(
    subject: DecisionSubject,
    *,
    advisory_scorer: Any | None = None,
    harm_classifier: Any | None = USE_SHIPPED_CLASSIFIER,
) -> BoundaryVerdict:
    """Run the enforced harm-uplift gate over a real decision; raise if it blocks.

    This is the choke point.  Every public decision surface calls it, with the
    same semantics, over the same assessment function, so a red-team decision
    that is blocked on one surface is blocked on all of them.

    Args:
        subject: The decision being made.  Built from the actual call.
        advisory_scorer: Optional
            :class:`~omni_mercury_engine.cognitive.ethical_bounding.BenevolenceScorer`
            (or the cached wrapper).  When supplied its benevolence float is
            computed, logged and attached to the verdict — **advisory only**.
            A scorer that raises is logged and ignored: an advisory signal must
            not be able to fail the decision, and must not be able to pass it.
        harm_classifier: Meaning-level classifier forwarded to
            :func:`assess_weapons_uplift`.  Consulted, never trusted: it can
            only raise a disposition.  Defaults to Mercury's shipped offline
            classifier; pass ``None`` for a strictly lexical gate.  This
            forwarder must not default to ``None``, or the choke point every
            decision surface routes through would be the one place that opts
            out of meaning-level coverage.

    Returns:
        A :class:`BoundaryVerdict` describing the permitted decision.

    Raises:
        EthicalConstraintViolationError: with ``check="harm_uplift"`` when the
            gate's disposition blocks (ESCALATE / REFUSE_REDACT / HARD_REFUSE),
            and — fail-closed — when anything in this function raises
            unexpectedly.
    """
    try:
        reading = subject._reading()
        action, assessment = _assess(reading, dict(subject.context), harm_classifier)
    except UnassessableSubjectError as exc:
        # Not an accident inside the gate: the caller sent more than it reads.
        # Refused like anything else it cannot evaluate, but said plainly and
        # put on the audit trail, where it used to be an ERROR with a
        # traceback and no record.
        logger.warning(
            "decision boundary %s (domain=%s): refused, %s",
            subject.surface,
            sanitize_domain(subject.domain),
            exc,
        )
        _refuse_unassessed(subject, reason=str(exc), unassessable=True)
    except Exception as exc:
        # Fail closed. Never return normally on an error path: an exception in
        # the gate must read as "refused", not as "allowed".
        #
        # Log provenance ONLY. The full subject carries ``request``, ``payload``
        # and ``context`` -- the caller's own data -- so logging it would spill
        # user input (credentials in a scanned payload, patient text, an
        # intercepted attacker body) into an unstructured error log with a
        # different retention and access policy than the audit trail, and would
        # emit up to MAX_SUBJECT_CHARS per failure. The refusal and the gate
        # audit record name the error and nothing of the caller's.
        logger.exception(
            "decision boundary %s (domain=%s): gate evaluation failed; failing closed",
            subject.surface,
            sanitize_domain(subject.domain),
        )
        _refuse_unassessed(subject, reason=f"{type(exc).__name__}: {exc}", cause=exc)

    if assessment.blocks:
        record_gate_decision(
            decision="refused",
            source="decision_boundary",
            disposition=assessment.disposition.value,
            hazard_domain=assessment.hazard_domain.value,
            intent=assessment.intent_tier.value,
            signals=assessment.signals,
            reason="harm-uplift gate blocked the decision at a public boundary",
            query=action,
            extra={"surface": subject.surface, "domain": sanitize_domain(subject.domain)},
        )
        raise EthicalConstraintViolationError(
            action=action,
            score=float(assessment.confidence),
            threshold=1.0,
            check="harm_uplift",
            details={
                "surface": subject.surface,
                "domain": sanitize_domain(subject.domain),
                "hazard_domain": assessment.hazard_domain.value,
                "operational_intent": assessment.intent_tier.value,
                "disposition": assessment.disposition.value,
                "signals": list(assessment.signals),
            },
        )

    benevolence: float | None = None
    if advisory_scorer is not None:
        try:
            benevolence = float(
                advisory_scorer.score_action(action, dict(subject.context)).benevolence_score
            )
        except Exception as exc:  # advisory: never blocks, never passes
            logger.warning(
                "decision boundary %s: advisory benevolence unavailable (%s)",
                subject.surface,
                exc,
            )
        else:
            logger.debug(
                "decision boundary %s: advisory benevolence=%.4f (informational; "
                "the enforced control is the harm-uplift gate)",
                subject.surface,
                benevolence,
            )

    return BoundaryVerdict(subject=subject, assessment=assessment, benevolence=benevolence)


__all__ = [
    "MAX_ASSESSED_CHARS",
    "MAX_ASSESSED_NODES",
    "MAX_SUBJECT_CHARS",
    "UNASSESSABLE",
    "BoundaryVerdict",
    "DecisionSubject",
    "HazardDomain",
    "OperationalIntent",
    "UnassessableSubjectError",
    "enforce_decision_boundary",
    "payload_text",
    "refused_for_size",
    "summarize_payload",
]
