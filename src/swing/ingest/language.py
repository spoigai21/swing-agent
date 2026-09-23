"""Keep non-English copy out of the evidence.

GOOGL 2026-09-21 was offered, as its two best pre-move items:

    ARTPRICE-NACHRICHTEN: EIN WELTBUCH WIRD ZUM SPIEGEL DER KÜNSTLICHEN INTELLIGENZ
    Noticia de Artprice: una obra literaria universal se convierte en espejo...

Both are tier 2 (PR Newswire), so they outranked every newsroom story. 590 of
6,066 PR Newswire releases in the corpus are non-English.

They are worse than merely irrelevant. The embedder is `bge-base-en-v1.5`, an
ENGLISH model, so foreign text lands somewhere arbitrary in the vector space —
its similarity to the query is noise, not a low score. And the attribution
prompt is English, so a cited German release tells the reader nothing.

⚠️ Demoted, never deleted. A tier-4 article is still stored and still counts in
coverage statistics; it simply cannot be cited. The same story usually exists in
English on the same wire, and that copy is unaffected.
"""
from __future__ import annotations

import re

#: Wire services prefix translated copy predictably.
FOREIGN_MARKERS = re.compile(
    r"\b(NACHRICHTEN|Noticia|Notícia|COMUNICADO|Communiqué|Comunicato|Persbericht|"
    r"Pressemitteilung|Pressemeddelelse|Meddelande|Tiedote|Anuncio|Informacja)\b",
    re.IGNORECASE)

#: Letters that essentially never appear in English wire copy.
NON_ENGLISH_LETTERS = re.compile(r"[ÄÖÜäöüßÑñÕõÅåØøÆæŒœÞþĐđŁłČčŠšŽžĂăŢţİı]")

#: CJK, Cyrillic, Greek, Arabic, Hebrew, Thai, Devanagari.
NON_LATIN = re.compile(
    r"[Ѐ-ӿͰ-Ͽ֐-׿؀-ۿ"
    r"ऀ-ॿ฀-๿぀-ヿ㐀-鿿가-힯]")

#: If a headline contains none of these, it is probably not an English sentence.
ENGLISH_STOPWORDS = frozenset(["a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "from", "at", "by", "as", "is", "are", "was", "were", "be", "been", "has", "have", "had", "will", "would", "can", "could", "its", "it", "this", "that", "these", "those", "after", "before", "over", "into", "new", "more", "than", "not", "no", "up", "down", "out", "off", "about", "against", "between", "during"])

MIN_WORDS_FOR_STOPWORD_TEST = 6


def looks_non_english(text: str) -> bool:
    """A deliberately conservative check: three independent signals, any of which
    is strong on its own. Accents alone are not enough — "Moët", "Nestlé" and
    "São Paulo" appear in perfectly English copy — so the accent test also
    requires the stopword test to fail.
    """
    if not text or not text.strip():
        return False
    if NON_LATIN.search(text):
        return True
    if FOREIGN_MARKERS.search(text):
        return True
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text.lower())
    if len(words) < MIN_WORDS_FOR_STOPWORD_TEST:
        return False                     # too short to judge; keep it
    has_english = any(w in ENGLISH_STOPWORDS for w in words)
    return bool(NON_ENGLISH_LETTERS.search(text)) and not has_english


def demote_non_english(headline: str, summary: str | None, tier: int) -> int:
    """Tier 4 for foreign-language copy: stored, never cited."""
    text = f"{headline or ''} {(summary or '')[:200]}"
    return 4 if looks_non_english(text) else tier
