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
  never operational specifics -- consistent with the public CBRN/
  non-proliferation vocabulary the English lexicon already uses.

Pure stdlib -- no third-party dependency -- so it loads anywhere the ethics
gate does.
"""

from __future__ import annotations

import re
import unicodedata

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
    """Return a newline-joined bundle of normalization variants for matching.

    The bundle is ``base \\n folded \\n collapsed``:

    * **base** -- lowercased, zero-width-stripped, diacritic-folded, whitespace-
      collapsed. Script-preserving; multi-word and native-script terms match here.
    * **folded** -- base with homoglyph + leetspeak folded to ASCII. Catches
      ``n3rv3 ag3nt`` and Cyrillic-glyph spoofing.
    * **collapsed** -- folded with all separators removed. Catches per-character
      spacing/punctuation insertion (``s.a.r.i.n``, ``n e r v e``).

    Existing ``keyword in haystack`` and ``pattern.search(haystack)`` logic runs
    unchanged and sees every variant. Fail-open on error: returns the plain
    lowercased text so a normalization bug never blinds the gate.
    """
    try:
        base = base_normalize(text)
        folded = _fold_obfuscation(base)
        collapsed = _collapse(folded)
        despaced = _punct_strip(folded)
        # Phonetic-Cyrillic variant: transliterate then apply the same folds, so a
        # phonetic Cyrillic obfuscation (Cyrillic н/р/в mixed with leetspeak) also
        # normalizes toward Latin. Skipped when there is no Cyrillic (no cost).
        translit = _cyrillic_translit(base)
        translit_variants = ()
        if translit != base:
            t_folded = _fold_obfuscation(translit)
            translit_variants = (t_folded, _collapse(t_folded), _punct_strip(t_folded))
        # De-dup identical variants to keep the haystack compact.
        seen: set[str] = set()
        variants: list[str] = []
        for v in (base, folded, despaced, collapsed, *translit_variants):
            if v and v not in seen:
                seen.add(v)
                variants.append(v)
        return "\n".join(variants)
    except Exception:
        return (text or "").lower()


# ---------------------------------------------------------------------------
# Multilingual hazard lexicon (Axis A extension).
#
# Taxonomy-level category terms only -- the same class of public
# CBRN/non-proliferation vocabulary the English lexicon uses -- in native
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
    "base_normalize",
    "normalized_haystack",
]
