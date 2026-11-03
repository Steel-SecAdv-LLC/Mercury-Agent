# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The agent understands a question in seven languages and frames its answer in the asker's.

Mercury's grounded answers are built in English from the detection record and
the code. This module is the layer in front of and behind them:

* :func:`detect_language` names the language a message is written in from
  its function words (``¿qué detectaste hoy?`` is Spanish, ``was kannst du?``
  German), never from a network service;
* :func:`to_english` renders the question in the English phrasing the
  classifier (:mod:`omni_mercury_engine.narrative.conversation`) already
  routes, phrase by phrase, so ``¿cuántas anomalías hoy?`` reaches the same
  builder as ``how many anomalies today?``;
* :func:`frame` gives the short, fixed replies (greeting, thanks, farewell,
  acknowledgement, the fallback menu, the note that a record answer is in
  English) in the asker's language, and :func:`localise_follow_ups`
  translates the follow-up prompts an answer suggests, each of which
  :func:`to_english` maps back when it is clicked.

What this layer does **not** do: translate a detection narrative or a
knowledge profile. Those are built from record fields and authored English
text; a machine rendering of them would be a second, unverified source. A
non-English asker therefore gets the social and menu replies in their own
language, the grounded answer in English, and a one-line note in their
language saying how the question was understood. That boundary is stated in
:func:`frame` and tested in ``tests/narrative/test_languages.py``.

Everything here is lexicon and regular expressions: no model, no dependency,
no I/O.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

__all__ = [
    "SUPPORTED",
    "detect_language",
    "frame",
    "localise_follow_ups",
    "normalise",
    "to_english",
]

#: Language code -> name in that language. ``en`` is the engine's own language.
SUPPORTED: Final[dict[str, str]] = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "pt": "Português",
    "nl": "Nederlands",
}


def normalise(text: str) -> str:
    """Lower-case ``text`` and strip its accents so one pattern matches both spellings.

    ``¿Qué detectaste hoy?`` becomes ``¿que detectaste hoy?``; ``ß`` becomes
    ``ss``. The inverted marks are kept because they identify Spanish.
    """
    lowered = text.strip().lower().replace("ß", "ss")
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _words(text: str) -> frozenset[str]:
    """The whitespace-separated words of ``text`` as a set (one marker list per line)."""
    return frozenset(text.split())


# --------------------------------------------------------------------------- #
# Detection: function words that identify a language
# --------------------------------------------------------------------------- #

_MARKERS: Final[dict[str, frozenset[str]]] = {
    "en": _words(
        "the is are was what which how you your do did can a an of in on today "
        "yesterday show me tell about this that have has any it run runs latest "
        "detect detected anomaly anomalies threshold"
    ),
    "es": _words(
        "el la los las de que es un una por para con como cual cuales hoy ayer y en del "
        "al puedes tienes hay esta estas detectaste muestrame dime sobre ultima ultimas "
        "cuantas cuantos quien eres gracias hola adios datos umbral ejecutar ejecuciones "
        "anomalia anomalias detectar fue usaste recomiendas cargo marcaron severidad "
        "puntuacion explica explicame deteccion detecciones muestras gusta mucho este tambien"
    ),
    "fr": _words(
        "le la les des est que qu une un pour avec comment quel quelle quels quelles "
        "aujourd hui hier et dans je tu vous peux peut ce cette sur montre dis moi derniere "
        "dernieres combien qui es merci bonjour salut au revoir faire dois lancer ensuite "
        "detecter detecte donnees seuil executions anomalie anomalies recommandes utilise "
        "echantillons signales severite"
    ),
    "de": _words(
        "der die das ist was wie und nicht ich du kannst hast heute gestern welche welcher "
        "welches wird mit von zu zeig zeige mir erklar erklare letzte letzten viele wer "
        "bist danke hallo tschuss bitte daten schwellenwert ausfuhren laufe anomalie "
        "anomalien erkennen erkennung empfiehlst proben markiert lade soll nachstes "
        "schweregrad bedrohungslage wert"
    ),
    "it": _words(
        "il lo la gli che cosa come e un una per con oggi ieri puoi hai quali quale qual del della "
        "dei delle mostrami dimmi ultima ultime quante quanti chi sei grazie ciao buongiorno "
        "arrivederci dati soglia eseguire esecuzioni anomalia anomalie rilevare rilevazione "
        "rilevatori consigli campioni segnalati usato carico cos spiega spiegami devo adesso "
        "severita punteggio minaccia livello molto piace sono questo questa anche"
    ),
    "pt": _words(
        "o a os as que e um uma como qual quais hoje ontem voce pode tem com para nao do da "
        "dos das mostre me diga sobre ultima ultimas quantas quantos quem obrigado obrigada "
        "ola tchau adeus dados limiar executar execucoes anomalia anomalias detectar "
        "detectou detectores recomenda amostras sinalizadas usou carrego devo seguir "
        "severidade pontuacao ameaca nivel foi"
    ),
    "nl": _words(
        "de het een wat hoe kun kan je ik is vandaag gisteren welke met van niet en zijn laat "
        "zien vertel over laatste hoeveel wie ben bent bedankt dank hallo hoi doei tot ziens "
        "jij jou data drempel draaien uitvoeren runs anomalie anomalieen detecteren detectie "
        "detectoren raad aan monsters gemarkeerd gebruikt laad moet hierna ernst score "
        "dreigingsniveau leg uit"
    ),
}

#: Words that belong to several languages decide nothing on their own.
_SHARED: Final[frozenset[str]] = frozenset(
    word
    for word in set().union(*_MARKERS.values())
    if sum(word in markers for markers in _MARKERS.values()) > 1
)

#: Spellings that only one language writes, read from the raw text before the
#: accents are stripped: an Italian ``è``, a Spanish ``ñ``, a Portuguese
#: ``ã``/``õ``/``ção``, a German umlaut, a Dutch ``ij``.
_HINTS: Final[tuple[tuple[str, re.Pattern[str], float], ...]] = (
    ("es", re.compile(r"[¿¡]|ñ"), 2.0),
    ("it", re.compile(r"\bè\b"), 2.0),
    ("it", re.compile(r"\b(?:l|un|dell|nell|all|qual|cos)'"), 1.5),
    ("pt", re.compile(r"[ãõ]|ção|ções|\bvocê\b"), 2.0),
    ("fr", re.compile(r"\b(?:qu|c|d|j|n|s)'|-(?:tu|je|vous|il|elle|on|moi)\b|ç"), 1.5),
    ("de", re.compile(r"[äöü]|ß"), 1.5),
    ("nl", re.compile(r"\bij\b|ij[a-z]"), 0.5),
)


def detect_language(text: str, *, default: str = "en") -> str:
    """Name the language ``text`` is written in, from its function words.

    Scores each supported language by the distinct marker words it finds
    (a word several languages share, such as ``la`` or ``de``, counts half
    and only once an unshared word placed the language), adds the spellings
    only one language writes (``è``, ``ñ``, ``ção``, an umlaut, a French
    elision), and answers ``default`` on a tie or on text with no markers at
    all ("bocpd?" is a name, not a sentence).
    """
    raw = text.strip().lower()
    folded = normalise(text)
    words = set(re.findall(r"[a-z]+", folded))
    scores: dict[str, float] = {}
    for code, markers in _MARKERS.items():
        hits = words & markers
        unshared = hits - _SHARED
        if unshared:
            scores[code] = len(unshared) + 0.5 * len(hits & _SHARED)
    for code, pattern, weight in _HINTS:
        if pattern.search(raw):
            scores[code] = scores.get(code, 0.0) + weight
    if not scores:
        return default
    best = max(scores.values())
    leaders = [code for code, score in scores.items() if score == best]
    if len(leaders) > 1:
        return default if default in leaders else sorted(leaders)[0]
    return leaders[0]


# --------------------------------------------------------------------------- #
# Understanding: native phrasings rendered in the English the classifier routes
# --------------------------------------------------------------------------- #

#: Per language, ``(pattern, english)`` pairs applied in order to the
#: accent-stripped, lower-cased text. The English fragments are exactly the
#: phrasings ``conversation.classify`` recognises; a message is "understood"
#: when at least one pair matched.
_PHRASES: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "es": (
        (r"^(?:hola|buenos dias|buenas tardes|buenas noches|buenas)\b", "hello"),
        (r"\b(?:muchas gracias|gracias|te lo agradezco)\b", "thank you"),
        (r"\b(?:adios|hasta luego|hasta pronto|chao|nos vemos)\b", "goodbye"),
        (r"^(?:vale|de acuerdo|si|claro|entendido|perfecto|bien|genial)\s*[.!]?$", "ok"),
        (
            r"\b(?:que puedes hacer|que sabes hacer|ayudame|ayuda|como funcionas)\b",
            "what can you do",
        ),
        (r"\b(?:estado del sistema|como esta el sistema|esta todo bien|todo bien)\b", "status"),
        (r"\b(?:cuantas anomalias|cuantas detecciones|cuantas alertas)\b", "how many anomalies"),
        (
            r"\b(?:que detectaste|que has detectado|ultima anomalia|ultima deteccion|"
            r"algo nuevo|algun problema|que encontraste|que paso|que ha pasado)\b",
            "what was the latest anomaly",
        ),
        (r"\b(?:explicame|explica|por que)\b", "explain"),
        (r"\bla ultima deteccion\b", "the latest detection"),
        (r"\b(?:que muestras|cuales muestras|que filas)\b", "which samples were flagged"),
        (r"\b(?:que umbral|cual umbral)\b", "what threshold did you use"),
        (
            r"\b(?:ultimas|ultimos)\s+(\d+)\s+(?:ejecuciones|corridas|detecciones|anomalias)\b",
            r"last \1 runs",
        ),
        (
            r"\b(?:nivel de amenaza|situacion general|que tan grave|debo preocuparme|es grave)\b",
            "threat posture",
        ),
        (r"\b(?:que recomiendas|que debo hacer|que hago)\b", "what do you recommend"),
        (
            r"\b(?:que debo ejecutar|que sigue|y ahora que|por donde empiezo|que ejecuto)\b",
            "what should i run next",
        ),
        (
            r"\b(?:como cargo datos|como recojo datos|cargar datos|fuentes de datos|"
            r"que dominios|cuales dominios)\b",
            "how do i collect data",
        ),
        (
            r"\b(?:que detectores|cuales detectores|que puedes detectar|puedes detectar)\b",
            "which detectors do you have",
        ),
        (r"\b(?:quien eres|que eres|como te llamas|presentate)\b", "who are you"),
        (r"^(?:hablame de|hablame del|cuentame sobre|cuentame del)\b", "tell me about"),
        (r"^(?:que es|que son|define|explica que es)\b", "what is"),
        (r"\bumbral\b", "threshold"),
        (r"\bpuntuacion z\b", "z-score"),
        (r"\b(?:severidad|gravedad)\b", "severity"),
        (r"\bconfianza\b", "confidence"),
        (r"\bfusion\b", "fusion"),
        (r"\banomalia\b", "anomaly"),
        (r"\bdominio\b", "domain"),
        (r"\bla puerta etica\b", "the gate"),
        (r"\bhoy\b", "today"),
        (r"\bayer\b", "yesterday"),
        (r"\besta semana\b", "this week"),
        (r"^(?:ejecuta|corre|lanza)\s+", "run "),
    ),
    "fr": (
        (r"^(?:bonjour|bonsoir|salut|coucou)\b", "hello"),
        (r"\b(?:merci beaucoup|merci)\b", "thank you"),
        (r"\b(?:au revoir|a bientot|a plus|adieu|bonne nuit)\b", "goodbye"),
        (r"^(?:ok|d'accord|oui|compris|parfait|bien|tres bien|super)\s*[.!]?$", "ok"),
        (
            r"\b(?:qu'est-ce que tu peux faire|que peux-tu faire|que sais-tu faire|aide-moi|"
            r"aide|comment fonctionnes-tu|comment tu fonctionnes)\b",
            "what can you do",
        ),
        (r"\b(?:etat du systeme|comment va le systeme|tout va bien|ca va)\b", "status"),
        (
            r"\b(?:combien d'anomalies|combien de detections|combien d'alertes)\b",
            "how many anomalies",
        ),
        (
            r"\b(?:qu'as-tu detecte|qu'est-ce que tu as detecte|derniere anomalie|"
            r"derniere detection|quoi de neuf|un probleme|qu'as-tu trouve|"
            r"qu'est-ce qui s'est passe)\b",
            "what was the latest anomaly",
        ),
        (r"\b(?:explique-moi|explique|pourquoi)\b", "explain"),
        (r"\bla derniere detection\b", "the latest detection"),
        (r"\b(?:quels echantillons|quelles lignes)\b", "which samples were flagged"),
        (r"\b(?:quel seuil)\b", "what threshold did you use"),
        (
            r"\b(?:les|des)?\s*(\d+)\s+(?:dernieres|derniers)\s+(?:executions|detections|anomalies)\b",
            r"last \1 runs",
        ),
        (
            r"\b(?:niveau de menace|situation generale|c'est grave|dois-je m'inquieter|"
            r"faut-il s'inquieter)\b",
            "threat posture",
        ),
        (r"\b(?:que recommandes-tu|que dois-je faire|que faire)\b", "what do you recommend"),
        (
            r"\b(?:que dois-je lancer|et maintenant|par ou commencer|quelle est la suite)\b",
            "what should i run next",
        ),
        (
            r"\b(?:comment charger des donnees|comment collecter des donnees|sources de donnees|"
            r"quels domaines)\b",
            "how do i collect data",
        ),
        (
            r"\b(?:quels detecteurs|que peux-tu detecter|peux-tu detecter)\b",
            "which detectors do you have",
        ),
        (r"\b(?:qui es-tu|qu'es-tu|comment t'appelles-tu|presente-toi)\b", "who are you"),
        (r"^(?:parle-moi de|parle-moi du|parle-moi des|decris)\b", "tell me about"),
        (r"^(?:qu'est-ce que|qu'est-ce qu'|c'est quoi|definis)\b", "what is"),
        (r"\bseuil\b", "threshold"),
        (r"\bscore z\b", "z-score"),
        (r"\b(?:severite|gravite)\b", "severity"),
        (r"\bconfiance\b", "confidence"),
        (r"\bfusion\b", "fusion"),
        (r"\banomalie\b", "anomaly"),
        (r"\bdomaine\b", "domain"),
        (r"\bla porte ethique\b", "the gate"),
        (r"\baujourd'hui\b", "today"),
        (r"\bhier\b", "yesterday"),
        (r"\bcette semaine\b", "this week"),
        (r"^(?:lance|execute|demarre)\s+", "run "),
    ),
    "de": (
        (r"^(?:hallo|guten morgen|guten tag|guten abend|servus|moin)\b", "hello"),
        (r"\b(?:vielen dank|danke)\b", "thank you"),
        (r"\b(?:tschuss|auf wiedersehen|bis spater|bis bald|gute nacht)\b", "goodbye"),
        (r"^(?:ok|okay|ja|gut|alles klar|verstanden|perfekt|in ordnung)\s*[.!]?$", "ok"),
        (
            r"\b(?:welche detektoren|was kannst du erkennen|kannst du .* erkennen)\b",
            "which detectors do you have",
        ),
        (r"\b(?:was kannst du|hilf mir|hilfe|wie funktionierst du)\b", "what can you do"),
        (r"\b(?:systemstatus|wie geht es dem system|ist alles in ordnung|alles ok)\b", "status"),
        (
            r"\b(?:wie viele anomalien|wie viele erkennungen|wie viele alarme)\b",
            "how many anomalies",
        ),
        (
            r"\b(?:was hast du erkannt|was hast du gefunden|letzte anomalie|letzte erkennung|"
            r"gibt es etwas neues|gibt es probleme|was ist passiert)\b",
            "what was the latest anomaly",
        ),
        (r"\b(?:erklar mir|erklare|erklar|warum)\b", "explain"),
        (r"\bdie letzte erkennung\b", "the latest detection"),
        (r"\b(?:welche proben|welche zeilen)\b", "which samples were flagged"),
        (r"\b(?:welcher schwellenwert|welche schwelle)\b", "what threshold did you use"),
        (
            r"\b(?:die\s+)?letzten\s+(\d+)\s+(?:laufe|durchlaufe|erkennungen|anomalien)\b",
            r"last \1 runs",
        ),
        (
            r"\b(?:bedrohungslage|gesamtlage|wie schlimm|muss ich mir sorgen machen|ist es ernst)\b",
            "threat posture",
        ),
        (r"\b(?:was empfiehlst du|was soll ich tun|was mache ich)\b", "what do you recommend"),
        (
            r"\b(?:was soll ich (?:als nachstes )?ausfuhren|was kommt als nachstes|als nachstes|"
            r"und jetzt|wo fange ich an)\b",
            "what should i run next",
        ),
        (
            r"\b(?:wie lade ich daten|wie sammle ich daten|datenquellen|welche domanen|"
            r"welche bereiche)\b",
            "how do i collect data",
        ),
        (r"\b(?:wer bist du|was bist du|wie heisst du|stell dich vor)\b", "who are you"),
        (r"^(?:erzahl mir von|erzahl mir uber|beschreibe)\b", "tell me about"),
        (r"^(?:was ist|was sind|definiere|was bedeutet)\b", "what is"),
        (r"\b(?:schwellenwert|schwelle)\b", "threshold"),
        (r"\bz-wert\b", "z-score"),
        (r"\b(?:schweregrad|schwere)\b", "severity"),
        (r"\b(?:konfidenz|vertrauen|zuversicht)\b", "confidence"),
        (r"\bfusion\b", "fusion"),
        (r"\banomalie\b", "anomaly"),
        (r"\b(?:domane|bereich)\b", "domain"),
        (r"\bdas ethische tor\b", "the gate"),
        (r"\bheute\b", "today"),
        (r"\bgestern\b", "yesterday"),
        (r"\bdiese woche\b", "this week"),
        (r"^(?:fuhre|starte)\s+", "run "),
    ),
    "it": (
        (r"^(?:ciao|buongiorno|buonasera|salve)\b", "hello"),
        (r"\b(?:grazie mille|grazie)\b", "thank you"),
        (r"\b(?:arrivederci|a presto|a dopo|addio|buonanotte)\b", "goodbye"),
        (r"^(?:ok|va bene|si|d'accordo|capito|perfetto|bene|ottimo)\s*[.!]?$", "ok"),
        (r"\b(?:cosa puoi fare|cosa sai fare|aiutami|aiuto|come funzioni)\b", "what can you do"),
        (r"\b(?:stato del sistema|come sta il sistema|va tutto bene|tutto bene)\b", "status"),
        (r"\b(?:quante anomalie|quante rilevazioni|quanti avvisi)\b", "how many anomalies"),
        (
            r"\b(?:cosa hai rilevato|cosa hai trovato|ultima anomalia|ultima rilevazione|"
            r"novita|qualche problema|cosa e successo)\b",
            "what was the latest anomaly",
        ),
        (r"\b(?:spiegami|spiega|perche)\b", "explain"),
        (r"\bl'ultima rilevazione\b", "the latest detection"),
        (r"\b(?:quali campioni|quali righe)\b", "which samples were flagged"),
        (r"\b(?:quale soglia)\b", "what threshold did you use"),
        (
            r"\b(?:le\s+)?ultime\s+(\d+)\s+(?:esecuzioni|rilevazioni|anomalie)\b",
            r"last \1 runs",
        ),
        (
            r"\b(?:livello di minaccia|quadro generale|quanto e grave|devo preoccuparmi|e grave)\b",
            "threat posture",
        ),
        (r"\b(?:cosa consigli|cosa devo fare|cosa faccio)\b", "what do you recommend"),
        (
            r"\b(?:cosa devo eseguire|e adesso|cosa viene dopo|da dove comincio)\b",
            "what should i run next",
        ),
        (
            r"\b(?:come carico i dati|come raccolgo i dati|fonti di dati|quali domini)\b",
            "how do i collect data",
        ),
        (
            r"\b(?:quali rilevatori|quali detector|cosa puoi rilevare|puoi rilevare)\b",
            "which detectors do you have",
        ),
        (r"\b(?:chi sei|cosa sei|come ti chiami|presentati)\b", "who are you"),
        (r"^(?:parlami di|parlami del|parlami della|descrivi)\b", "tell me about"),
        (r"^(?:cos'e|cosa e|cosa sono|che cos'e|definisci)\b", "what is"),
        (r"\bsoglia\b", "threshold"),
        (r"\bpunteggio z\b", "z-score"),
        (r"\b(?:severita|gravita)\b", "severity"),
        (r"\bconfidenza\b", "confidence"),
        (r"\bfusione\b", "fusion"),
        (r"\banomalia\b", "anomaly"),
        (r"\bdominio\b", "domain"),
        (r"\bla porta etica\b", "the gate"),
        (r"\boggi\b", "today"),
        (r"\bieri\b", "yesterday"),
        (r"\bquesta settimana\b", "this week"),
        (r"^(?:esegui|lancia|avvia)\s+", "run "),
    ),
    "pt": (
        (r"^(?:ola|oi|bom dia|boa tarde|boa noite)\b", "hello"),
        (r"\b(?:muito obrigad[oa]|obrigad[oa]|valeu)\b", "thank you"),
        (r"\b(?:tchau|adeus|ate logo|ate mais|boa noite)\b", "goodbye"),
        (r"^(?:ok|tudo bem|sim|certo|entendido|perfeito|beleza|combinado)\s*[.!]?$", "ok"),
        (
            r"\b(?:o que voce pode fazer|o que voce sabe fazer|o que pode fazer|me ajude|ajuda|"
            r"como voce funciona)\b",
            "what can you do",
        ),
        (r"\b(?:estado do sistema|como esta o sistema|esta tudo bem|tudo certo)\b", "status"),
        (r"\b(?:quantas anomalias|quantas detec?coes|quantos alertas)\b", "how many anomalies"),
        (
            r"\b(?:o que voce detectou|o que detectou|o que voce encontrou|ultima anomalia|"
            r"ultima detec?cao|algo novo|algum problema|o que aconteceu)\b",
            "what was the latest anomaly",
        ),
        (r"\b(?:me explique|explique|explica|por que)\b", "explain"),
        (r"\ba ultima detec?cao\b", "the latest detection"),
        (r"\b(?:quais amostras|quais linhas)\b", "which samples were flagged"),
        (r"\b(?:qual limiar|qual limite)\b", "what threshold did you use"),
        (
            r"\b(?:as\s+)?ultimas\s+(\d+)\s+(?:execucoes|deteccoes|anomalias)\b",
            r"last \1 runs",
        ),
        (
            r"\b(?:nivel de ameaca|quadro geral|quao grave|devo me preocupar|e grave)\b",
            "threat posture",
        ),
        (
            r"\b(?:o que voce recomenda|o que recomenda|o que devo fazer|o que faco)\b",
            "what do you recommend",
        ),
        (
            r"\b(?:o que devo executar|o que vem a seguir|e agora|por onde comeco)\b",
            "what should i run next",
        ),
        (
            r"\b(?:como carrego dados|como coleto dados|fontes de dados|quais dominios)\b",
            "how do i collect data",
        ),
        (
            r"\b(?:quais detectores|o que voce pode detectar|voce pode detectar|pode detectar)\b",
            "which detectors do you have",
        ),
        (r"\b(?:quem e voce|o que e voce|como voce se chama|apresente-se)\b", "who are you"),
        (r"^(?:fale sobre|fale-me sobre|me fale sobre|descreva)\b", "tell me about"),
        (r"^(?:o que e|o que sao|defina|explique o que e)\b", "what is"),
        (r"\b(?:limiar|limite)\b", "threshold"),
        (r"\bpontuacao z\b", "z-score"),
        (r"\b(?:severidade|gravidade)\b", "severity"),
        (r"\bconfianca\b", "confidence"),
        (r"\bfusao\b", "fusion"),
        (r"\banomalia\b", "anomaly"),
        (r"\bdominio\b", "domain"),
        (r"\bo portao etico\b", "the gate"),
        (r"\bhoje\b", "today"),
        (r"\bontem\b", "yesterday"),
        (r"\besta semana\b", "this week"),
        (r"^(?:execute|rode|inicie)\s+", "run "),
    ),
    "nl": (
        (r"^(?:hallo|hoi|goedemorgen|goedemiddag|goedenavond|hey)\b", "hello"),
        (r"\b(?:bedankt|dank je|dank je wel|dankjewel|dank u)\b", "thank you"),
        (r"\b(?:doei|tot ziens|tot later|dag|welterusten)\b", "goodbye"),
        (r"^(?:ok|oke|prima|ja|goed|begrepen|perfect|in orde)\s*[.!]?$", "ok"),
        (
            r"\b(?:welke detectoren|wat kun je detecteren|kun je .* detecteren)\b",
            "which detectors do you have",
        ),
        (r"\b(?:wat kun je|wat kan je|wat kun jij|help me|help|hoe werk je)\b", "what can you do"),
        (
            r"\b(?:systeemstatus|hoe gaat het met het systeem|is alles in orde|alles goed)\b",
            "status",
        ),
        (r"\b(?:hoeveel anomalieen|hoeveel detecties|hoeveel meldingen)\b", "how many anomalies"),
        (
            r"\b(?:wat heb je gedetecteerd|wat heb je gevonden|laatste anomalie|laatste detectie|"
            r"iets nieuws|zijn er problemen|wat is er gebeurd)\b",
            "what was the latest anomaly",
        ),
        (r"\b(?:leg (?:me )?uit|verklaar|waarom)\b|^leg\b", "explain"),
        (r"\bde laatste detectie\b", "the latest detection"),
        (r"\b(?:welke monsters|welke rijen|welke samples)\b", "which samples were flagged"),
        (r"\b(?:welke drempel|welke drempelwaarde)\b", "what threshold did you use"),
        (
            r"\b(?:de\s+)?laatste\s+(\d+)\s+(?:runs|detecties|anomalieen)\b",
            r"last \1 runs",
        ),
        (
            r"\b(?:dreigingsniveau|totaalbeeld|hoe erg|moet ik me zorgen maken|is het ernstig)\b",
            "threat posture",
        ),
        (r"\b(?:wat raad je aan|wat moet ik doen|wat doe ik)\b", "what do you recommend"),
        (
            r"\b(?:wat moet ik (?:hierna )?(?:draaien|uitvoeren)|wat nu|wat is de volgende stap|"
            r"waar begin ik)\b",
            "what should i run next",
        ),
        (
            r"\b(?:hoe laad ik data|hoe verzamel ik data|databronnen|welke domeinen)\b",
            "how do i collect data",
        ),
        (r"\b(?:wie ben je|wat ben je|hoe heet je|stel je voor)\b", "who are you"),
        (r"^(?:vertel me over|vertel over|beschrijf)\b", "tell me about"),
        (r"^(?:wat is|wat zijn|definieer|wat betekent)\b", "what is"),
        (r"\b(?:drempelwaarde|drempel)\b", "threshold"),
        (r"\bz-score\b", "z-score"),
        (r"\b(?:ernst|ernstniveau)\b", "severity"),
        (r"\b(?:betrouwbaarheid|vertrouwen)\b", "confidence"),
        (r"\bfusie\b", "fusion"),
        (r"\banomalie\b", "anomaly"),
        (r"\bdomein\b", "domain"),
        (r"\bde ethische poort\b", "the gate"),
        (r"\bvandaag\b", "today"),
        (r"\bgisteren\b", "yesterday"),
        (r"\bdeze week\b", "this week"),
        (r"^(?:draai|voer|start)\s+", "run "),
    ),
}

_COMPILED: Final[dict[str, tuple[tuple[re.Pattern[str], str], ...]]] = {
    code: tuple((re.compile(pattern), english) for pattern, english in pairs)
    for code, pairs in _PHRASES.items()
}


def to_english(text: str, language: str) -> str | None:
    """Render ``text`` in the English phrasing the classifier routes, or ``None``.

    Every matching phrase pair of ``language`` is applied in order to the
    accent-stripped text; ``None`` means no pair matched, so the message was
    not understood and the caller answers with the fallback menu in that
    language. English input is returned unchanged.
    """
    if language == "en":
        return text
    pairs = _COMPILED.get(language)
    if not pairs:
        return None
    out = re.sub(r"[¿¡]", "", normalise(text)).strip()
    matched = False
    for pattern, english in pairs:
        out, n = pattern.subn(english, out)
        matched = matched or n > 0
    if not matched:
        return None
    return " ".join(out.split())


# --------------------------------------------------------------------------- #
# Framing: the fixed replies in the asker's language
# --------------------------------------------------------------------------- #

_FRAMES: Final[dict[str, dict[str, str]]] = {
    "en": {
        "greeting": (
            "Hello. I am Mercury Agent, answering from this server's detection record and its "
            "code in the {domain} domain. Ask what ran, what was flagged, what to run next, or "
            "what any of Mercury's terms mean."
        ),
        "thanks": "You are welcome. Ask again whenever a run lands in memory.",
        "farewell": (
            "Goodbye. The detection record stays here; ask me about it whenever you are back."
        ),
        "ack": "Noted. Whenever you are ready, here is what I can take next.",
        "confusion": (
            "Nothing has been asked yet that I could put another way. Ask me about a run, a "
            "detector or one of Mercury's terms and I will answer, then 'what do you mean?' "
            "gets the plainer version."
        ),
        "repeat": "There is no previous answer to repeat yet.",
        "correction": (
            "Understood, that was not what you asked. I answer from the record (what ran, what "
            "was flagged, when), from the code (detectors, domains, commands, terms) and from "
            "the narrative engine (why); say which, or pick one of these."
        ),
        "brief": "In short: {summary}",
        "fallback": (
            "I have nothing in the detection record or the code that answers '{query}'. I can "
            "tell you what ran and when (latest anomaly, flagged samples, threshold, how many "
            "today), explain a run through the narrative engine, describe the detectors and "
            "capabilities this server has, define Mercury's terms (severity, threshold, z-score, "
            "AMA, FINDΩYOU, the gate), and say what to run next or how to collect data."
        ),
        "understood": "",
    },
    "es": {
        "greeting": (
            "Hola. Soy Mercury Agent y respondo a partir del registro de detecciones de este "
            "servidor y de su código, en el dominio {domain}. Pregunta qué se ejecutó, qué se "
            "marcó, qué ejecutar a continuación o qué significa cualquier término de Mercury."
        ),
        "thanks": "De nada. Vuelve a preguntar cuando una ejecución llegue a la memoria.",
        "farewell": "Adiós. El registro de detecciones se queda aquí; pregúntame cuando vuelvas.",
        "ack": "Entendido. Cuando quieras, esto es lo que puedo hacer a continuación.",
        "confusion": (
            "Todavía no se ha preguntado nada que pueda explicar de otra forma. Pregúntame por "
            "una ejecución, un detector o un término de Mercury y después '¿qué quieres decir?' "
            "te da la versión más sencilla."
        ),
        "repeat": "Todavía no hay una respuesta anterior que repetir.",
        "correction": (
            "Entendido, no era lo que preguntabas. Respondo desde el registro (qué se ejecutó, "
            "qué se marcó, cuándo), desde el código (detectores, dominios, comandos, términos) y "
            "desde el motor narrativo (por qué); dime cuál, o elige una de estas."
        ),
        "brief": "En resumen: {summary}",
        "fallback": (
            "No tengo nada en el registro de detecciones ni en el código que responda a "
            "'{query}'. Puedo decirte qué se ejecutó y cuándo (última anomalía, muestras "
            "marcadas, umbral, cuántas hoy), explicar una ejecución con el motor narrativo, "
            "describir los detectores y capacidades de este servidor, definir los términos de "
            "Mercury y decir qué ejecutar a continuación o cómo cargar datos."
        ),
        "understood": (
            "(Entendí la pregunta como «{english}». Las respuestas construidas desde el "
            "registro y el código están en inglés.)"
        ),
    },
    "fr": {
        "greeting": (
            "Bonjour. Je suis Mercury Agent et je réponds à partir du journal de détection de ce "
            "serveur et de son code, dans le domaine {domain}. Demandez ce qui a tourné, ce qui "
            "a été signalé, quoi lancer ensuite ou ce que signifie un terme de Mercury."
        ),
        "thanks": "Je vous en prie. Revenez dès qu'une exécution arrive en mémoire.",
        "farewell": "Au revoir. Le journal de détection reste ici ; interrogez-moi à votre retour.",
        "ack": "Compris. Quand vous voulez, voici ce que je peux faire ensuite.",
        "confusion": (
            "Rien n'a encore été demandé que je puisse reformuler. Posez une question sur une "
            "exécution, un détecteur ou un terme de Mercury, puis « que veux-tu dire ? » en "
            "donne la version plus simple."
        ),
        "repeat": "Il n'y a pas encore de réponse précédente à répéter.",
        "correction": (
            "Compris, ce n'était pas votre question. Je réponds depuis le journal (ce qui a "
            "tourné, ce qui a été signalé, quand), depuis le code (détecteurs, domaines, "
            "commandes, termes) et depuis le moteur narratif (pourquoi) ; dites lequel, ou "
            "choisissez ci-dessous."
        ),
        "brief": "En bref : {summary}",
        "fallback": (
            "Rien dans le journal de détection ni dans le code ne répond à « {query} ». Je peux "
            "dire ce qui a tourné et quand (dernière anomalie, échantillons signalés, seuil, "
            "combien aujourd'hui), expliquer une exécution avec le moteur narratif, décrire les "
            "détecteurs et capacités de ce serveur, définir les termes de Mercury et dire quoi "
            "lancer ensuite ou comment charger des données."
        ),
        "understood": (
            "(J'ai compris la question comme « {english} ». Les réponses construites depuis le "
            "journal et le code sont en anglais.)"
        ),
    },
    "de": {
        "greeting": (
            "Hallo. Ich bin Mercury Agent und antworte aus dem Erkennungsprotokoll dieses Servers "
            "und seinem Code, im Bereich {domain}. Frag, was gelaufen ist, was markiert wurde, "
            "was als Nächstes zu starten ist oder was ein Begriff von Mercury bedeutet."
        ),
        "thanks": "Gern geschehen. Frag wieder, sobald ein Lauf im Speicher landet.",
        "farewell": "Tschüss. Das Erkennungsprotokoll bleibt hier; frag mich, wenn du zurück bist.",
        "ack": "Verstanden. Wann immer du bereit bist, das kann ich als Nächstes tun.",
        "confusion": (
            "Es wurde noch nichts gefragt, das ich anders ausdrücken könnte. Frag mich nach "
            "einem Lauf, einem Detektor oder einem Begriff von Mercury; danach liefert "
            "'was meinst du?' die einfachere Fassung."
        ),
        "repeat": "Es gibt noch keine vorherige Antwort zum Wiederholen.",
        "correction": (
            "Verstanden, das war nicht deine Frage. Ich antworte aus dem Protokoll (was lief, "
            "was markiert wurde, wann), aus dem Code (Detektoren, Bereiche, Befehle, Begriffe) "
            "und aus der Erzähl-Engine (warum); sag welches, oder wähle unten."
        ),
        "brief": "Kurz gesagt: {summary}",
        "fallback": (
            "Weder das Erkennungsprotokoll noch der Code beantworten '{query}'. Ich kann sagen, "
            "was wann gelaufen ist (letzte Anomalie, markierte Proben, Schwellenwert, wie viele "
            "heute), einen Lauf mit der Erzähl-Engine erklären, die Detektoren und Fähigkeiten "
            "dieses Servers beschreiben, Mercurys Begriffe definieren und sagen, was als "
            "Nächstes zu starten ist oder wie Daten geladen werden."
        ),
        "understood": (
            "(Ich habe die Frage als '{english}' verstanden. Antworten aus dem Protokoll und "
            "dem Code sind auf Englisch.)"
        ),
    },
    "it": {
        "greeting": (
            "Ciao. Sono Mercury Agent e rispondo dal registro delle rilevazioni di questo server "
            "e dal suo codice, nel dominio {domain}. Chiedi cosa è stato eseguito, cosa è stato "
            "segnalato, cosa eseguire dopo o cosa significa un termine di Mercury."
        ),
        "thanks": "Prego. Chiedi di nuovo quando un'esecuzione arriva in memoria.",
        "farewell": "Arrivederci. Il registro delle rilevazioni resta qui; chiedimi quando torni.",
        "ack": "Capito. Quando vuoi, ecco cosa posso fare dopo.",
        "confusion": (
            "Non è ancora stato chiesto nulla che io possa riformulare. Chiedimi di "
            "un'esecuzione, di un rilevatore o di un termine di Mercury, poi 'cosa intendi?' "
            "dà la versione più semplice."
        ),
        "repeat": "Non c'è ancora una risposta precedente da ripetere.",
        "correction": (
            "Capito, non era la tua domanda. Rispondo dal registro (cosa è stato eseguito, cosa "
            "è stato segnalato, quando), dal codice (rilevatori, domini, comandi, termini) e dal "
            "motore narrativo (perché); dimmi quale, o scegli qui sotto."
        ),
        "brief": "In breve: {summary}",
        "fallback": (
            "Nel registro delle rilevazioni e nel codice non c'è nulla che risponda a "
            "'{query}'. Posso dire cosa è stato eseguito e quando (ultima anomalia, campioni "
            "segnalati, soglia, quante oggi), spiegare un'esecuzione con il motore narrativo, "
            "descrivere i rilevatori e le capacità di questo server, definire i termini di "
            "Mercury e dire cosa eseguire dopo o come caricare i dati."
        ),
        "understood": (
            "(Ho inteso la domanda come '{english}'. Le risposte costruite dal registro e dal "
            "codice sono in inglese.)"
        ),
    },
    "pt": {
        "greeting": (
            "Olá. Sou o Mercury Agent e respondo a partir do registo de deteções deste servidor "
            "e do seu código, no domínio {domain}. Pergunte o que foi executado, o que foi "
            "sinalizado, o que executar a seguir ou o que significa um termo do Mercury."
        ),
        "thanks": "De nada. Pergunte de novo quando uma execução chegar à memória.",
        "farewell": "Adeus. O registo de deteções fica aqui; pergunte-me quando voltar.",
        "ack": "Entendido. Quando quiser, é isto que posso fazer a seguir.",
        "confusion": (
            "Ainda não foi perguntado nada que eu possa dizer de outra forma. Pergunte-me "
            "sobre uma execução, um detetor ou um termo do Mercury; depois 'o que quer dizer?' "
            "dá a versão mais simples."
        ),
        "repeat": "Ainda não há uma resposta anterior para repetir.",
        "correction": (
            "Entendido, não era isso que perguntou. Respondo a partir do registo (o que foi "
            "executado, o que foi sinalizado, quando), do código (detetores, domínios, "
            "comandos, termos) e do motor narrativo (porquê); diga qual, ou escolha abaixo."
        ),
        "brief": "Em resumo: {summary}",
        "fallback": (
            "Não há nada no registo de deteções nem no código que responda a '{query}'. Posso "
            "dizer o que foi executado e quando (última anomalia, amostras sinalizadas, limiar, "
            "quantas hoje), explicar uma execução com o motor narrativo, descrever os detetores "
            "e as capacidades deste servidor, definir os termos do Mercury e dizer o que "
            "executar a seguir ou como carregar dados."
        ),
        "understood": (
            "(Entendi a pergunta como '{english}'. As respostas construídas a partir do registo "
            "e do código estão em inglês.)"
        ),
    },
    "nl": {
        "greeting": (
            "Hallo. Ik ben Mercury Agent en antwoord vanuit het detectielogboek van deze server "
            "en zijn code, in het domein {domain}. Vraag wat er is gedraaid, wat is gemarkeerd, "
            "wat je hierna moet draaien of wat een term van Mercury betekent."
        ),
        "thanks": "Graag gedaan. Vraag opnieuw zodra een run in het geheugen staat.",
        "farewell": "Tot ziens. Het detectielogboek blijft hier; vraag het me als je terug bent.",
        "ack": "Begrepen. Wanneer je klaar bent, dit kan ik hierna doen.",
        "confusion": (
            "Er is nog niets gevraagd dat ik anders kan verwoorden. Vraag me naar een run, een "
            "detector of een term van Mercury; daarna geeft 'wat bedoel je?' de eenvoudigere "
            "versie."
        ),
        "repeat": "Er is nog geen eerder antwoord om te herhalen.",
        "correction": (
            "Begrepen, dat was niet je vraag. Ik antwoord vanuit het logboek (wat draaide, wat "
            "is gemarkeerd, wanneer), vanuit de code (detectoren, domeinen, commando's, termen) "
            "en vanuit de verhaal-engine (waarom); zeg welke, of kies hieronder."
        ),
        "brief": "Kortom: {summary}",
        "fallback": (
            "Niets in het detectielogboek of de code beantwoordt '{query}'. Ik kan zeggen wat "
            "er wanneer draaide (laatste anomalie, gemarkeerde monsters, drempel, hoeveel "
            "vandaag), een run uitleggen met de verhaal-engine, de detectoren en mogelijkheden "
            "van deze server beschrijven, de termen van Mercury definiëren en zeggen wat je "
            "hierna draait of hoe je data laadt."
        ),
        "understood": (
            "(Ik heb de vraag begrepen als '{english}'. Antwoorden uit het logboek en de code "
            "zijn in het Engels.)"
        ),
    },
}


def frame(language: str, key: str, **values: str) -> str:
    """The fixed reply ``key`` in ``language``, falling back to English.

    Keys: ``greeting`` (``domain``), ``thanks``, ``farewell``, ``ack``,
    ``confusion``, ``repeat``, ``correction``, ``brief`` (``summary``),
    ``fallback`` (``query``) and ``understood`` (``english``; empty for
    English, where nothing was rendered).
    """
    table = _FRAMES.get(language) or _FRAMES["en"]
    template = table.get(key)
    if template is None:
        template = _FRAMES["en"][key]
    return template.format(**values)


#: The follow-up prompts answers suggest, in each language. Every entry maps
#: back to its English line through :func:`to_english`, so a clicked prompt is
#: understood (``tests/narrative/test_languages.py`` checks the round trip).
_FOLLOW_UPS: Final[dict[str, dict[str, str]]] = {
    "es": {
        "What was the latest anomaly?": "¿Cuál fue la última anomalía?",
        "What can you do?": "¿Qué puedes hacer?",
        "What should I run next?": "¿Qué debo ejecutar a continuación?",
        "Which detectors do you have?": "¿Qué detectores tienes?",
        "Explain the latest detection": "Explica la última detección",
        "Show me the last 5 runs": "Muéstrame las últimas 5 ejecuciones",
        "Show me the last 10 runs": "Muéstrame las últimas 10 ejecuciones",
        "How do I collect data?": "¿Cómo cargo datos?",
        "What are your capabilities?": "¿Qué puedes detectar?",
        "Recommend an action": "¿Qué recomiendas?",
        "Which samples were flagged?": "¿Qué muestras se marcaron?",
        "What threshold did you use?": "¿Qué umbral usaste?",
        "How many anomalies today?": "¿Cuántas anomalías hoy?",
        "Summarize threat posture": "¿Cuál es el nivel de amenaza?",
        "What is the threshold?": "¿Qué es el umbral?",
        "What is severity?": "¿Qué es la severidad?",
        "What is a z-score?": "¿Qué es la puntuación z?",
    },
    "fr": {
        "What was the latest anomaly?": "Quelle est la dernière anomalie ?",
        "What can you do?": "Que peux-tu faire ?",
        "What should I run next?": "Que dois-je lancer ensuite ?",
        "Which detectors do you have?": "Quels détecteurs as-tu ?",
        "Explain the latest detection": "Explique la dernière détection",
        "Show me the last 5 runs": "Montre les 5 dernières exécutions",
        "Show me the last 10 runs": "Montre les 10 dernières exécutions",
        "How do I collect data?": "Comment charger des données ?",
        "What are your capabilities?": "Que peux-tu détecter ?",
        "Recommend an action": "Que recommandes-tu ?",
        "Which samples were flagged?": "Quels échantillons ont été signalés ?",
        "What threshold did you use?": "Quel seuil as-tu utilisé ?",
        "How many anomalies today?": "Combien d'anomalies aujourd'hui ?",
        "Summarize threat posture": "Quel est le niveau de menace ?",
        "What is the threshold?": "Qu'est-ce que le seuil ?",
        "What is severity?": "Qu'est-ce que la sévérité ?",
        "What is a z-score?": "Qu'est-ce que le score z ?",
    },
    "de": {
        "What was the latest anomaly?": "Was war die letzte Anomalie?",
        "What can you do?": "Was kannst du?",
        "What should I run next?": "Was soll ich als Nächstes ausführen?",
        "Which detectors do you have?": "Welche Detektoren hast du?",
        "Explain the latest detection": "Erkläre die letzte Erkennung",
        "Show me the last 5 runs": "Zeig mir die letzten 5 Läufe",
        "Show me the last 10 runs": "Zeig mir die letzten 10 Läufe",
        "How do I collect data?": "Wie lade ich Daten?",
        "What are your capabilities?": "Was kannst du erkennen?",
        "Recommend an action": "Was empfiehlst du?",
        "Which samples were flagged?": "Welche Proben wurden markiert?",
        "What threshold did you use?": "Welcher Schwellenwert wurde verwendet?",
        "How many anomalies today?": "Wie viele Anomalien heute?",
        "Summarize threat posture": "Wie ist die Bedrohungslage?",
        "What is the threshold?": "Was ist der Schwellenwert?",
        "What is severity?": "Was ist der Schweregrad?",
        "What is a z-score?": "Was ist ein Z-Wert?",
    },
    "it": {
        "What was the latest anomaly?": "Qual è l'ultima anomalia?",
        "What can you do?": "Cosa puoi fare?",
        "What should I run next?": "Cosa devo eseguire adesso?",
        "Which detectors do you have?": "Quali rilevatori hai?",
        "Explain the latest detection": "Spiega l'ultima rilevazione",
        "Show me the last 5 runs": "Mostrami le ultime 5 esecuzioni",
        "Show me the last 10 runs": "Mostrami le ultime 10 esecuzioni",
        "How do I collect data?": "Come carico i dati?",
        "What are your capabilities?": "Cosa puoi rilevare?",
        "Recommend an action": "Cosa consigli?",
        "Which samples were flagged?": "Quali campioni sono stati segnalati?",
        "What threshold did you use?": "Quale soglia hai usato?",
        "How many anomalies today?": "Quante anomalie oggi?",
        "Summarize threat posture": "Qual è il livello di minaccia?",
        "What is the threshold?": "Cos'è la soglia?",
        "What is severity?": "Cos'è la severità?",
        "What is a z-score?": "Cos'è il punteggio z?",
    },
    "pt": {
        "What was the latest anomaly?": "Qual foi a última anomalia?",
        "What can you do?": "O que você pode fazer?",
        "What should I run next?": "O que devo executar a seguir?",
        "Which detectors do you have?": "Quais detectores você tem?",
        "Explain the latest detection": "Explique a última deteção",
        "Show me the last 5 runs": "Mostre as últimas 5 execuções",
        "Show me the last 10 runs": "Mostre as últimas 10 execuções",
        "How do I collect data?": "Como carrego dados?",
        "What are your capabilities?": "O que você pode detectar?",
        "Recommend an action": "O que você recomenda?",
        "Which samples were flagged?": "Quais amostras foram sinalizadas?",
        "What threshold did you use?": "Qual limiar você usou?",
        "How many anomalies today?": "Quantas anomalias hoje?",
        "Summarize threat posture": "Qual é o nível de ameaça?",
        "What is the threshold?": "O que é o limiar?",
        "What is severity?": "O que é a severidade?",
        "What is a z-score?": "O que é a pontuação z?",
    },
    "nl": {
        "What was the latest anomaly?": "Wat was de laatste anomalie?",
        "What can you do?": "Wat kun je?",
        "What should I run next?": "Wat moet ik hierna draaien?",
        "Which detectors do you have?": "Welke detectoren heb je?",
        "Explain the latest detection": "Leg de laatste detectie uit",
        "Show me the last 5 runs": "Laat de laatste 5 runs zien",
        "Show me the last 10 runs": "Laat de laatste 10 runs zien",
        "How do I collect data?": "Hoe laad ik data?",
        "What are your capabilities?": "Wat kun je detecteren?",
        "Recommend an action": "Wat raad je aan?",
        "Which samples were flagged?": "Welke monsters zijn gemarkeerd?",
        "What threshold did you use?": "Welke drempel is gebruikt?",
        "How many anomalies today?": "Hoeveel anomalieën vandaag?",
        "Summarize threat posture": "Wat is het dreigingsniveau?",
        "What is the threshold?": "Wat is de drempel?",
        "What is severity?": "Wat is de ernst?",
        "What is a z-score?": "Wat is een z-score?",
    },
}


def localise_follow_ups(language: str, prompts: list[str]) -> list[str]:
    """The follow-up prompts in ``language`` where a translation exists.

    A prompt with no translation (a domain name, a detector name) is kept in
    English so it still routes when clicked.
    """
    table = _FOLLOW_UPS.get(language)
    if not table:
        return list(prompts)
    return [table.get(p, p) for p in prompts]


def follow_up_translations(language: str) -> dict[str, str]:
    """The English-to-``language`` prompt table (empty for English)."""
    return dict(_FOLLOW_UPS.get(language, {}))
