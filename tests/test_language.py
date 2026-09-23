"""Foreign-language wire copy was outranking newsrooms.

GOOGL 2026-09-21 was offered, as its two best pre-move items, a German and a
Spanish Artprice release — tier 2, because that is PR Newswire's tier. They are
worse than irrelevant: the embedder is `bge-base-en-v1.5`, so foreign text lands
somewhere arbitrary in the vector space and its similarity to the query is noise
rather than a low score. 482 citable articles were affected.
"""
from __future__ import annotations

from swing.ingest.language import demote_non_english, looks_non_english


class TestTheCasesThatPromptedThis:
    def test_the_german_artprice_release(self):
        assert looks_non_english(
            "ARTPRICE-NACHRICHTEN: EIN WELTBUCH WIRD ZUM SPIEGEL DER "
            "KÜNSTLICHEN INTELLIGENZ")

    def test_the_spanish_one(self):
        assert looks_non_english(
            "Noticia de Artprice: una obra literaria universal se convierte en espejo")


class TestItDoesNotOverreach:
    def test_english_copy_with_accented_brand_names_survives(self):
        """"Moët", "Nestlé" and "São Paulo" appear in perfectly English copy, so
        accents alone must never be enough."""
        assert not looks_non_english(
            "Nvidia and Moët Hennessy announce a partnership in São Paulo for the year")

    def test_ordinary_english_headlines_survive(self):
        for text in ("Qualcomm unveils AI200 and AI250 data-center accelerators",
                     "Apple Q3 revenue rises as iPhone demand holds up",
                     "AMD 8-K — Item 1.01 — AMD AND OPENAI ANNOUNCE STRATEGIC PARTNERSHIP"):
            assert not looks_non_english(text)

    def test_a_headline_too_short_to_judge_is_kept(self):
        """Three words carry no stopword evidence either way; keeping it is the
        conservative error."""
        assert not looks_non_english("Apple Q3 revenue")

    def test_empty_text_is_not_foreign(self):
        assert not looks_non_english("")
        assert not looks_non_english("   ")


class TestNonLatinScripts:
    def test_japanese(self):
        assert looks_non_english("エヌビディア、新型GPUを発表")

    def test_chinese_and_korean_and_cyrillic(self):
        for text in ("英伟达发布新一代GPU", "엔비디아, 새로운 GPU 발표", "Нвидиа представила GPU"):
            assert looks_non_english(text)


class TestDemotion:
    def test_foreign_copy_lands_in_tier_four(self):
        assert demote_non_english("ARTPRICE-NACHRICHTEN: EIN WELTBUCH", None, 2) == 4

    def test_english_copy_keeps_its_publisher_tier(self):
        assert demote_non_english("Qualcomm unveils new accelerators today", None, 2) == 2

    def test_the_summary_is_considered_too(self):
        assert demote_non_english("Artprice", "Noticia de Artprice: una obra "
                                  "literaria universal se convierte en espejo", 2) == 4

    def test_it_demotes_rather_than_deletes(self):
        """Tier 4 is stored and counted, just never cited — and the same story
        usually exists in English on the same wire."""
        assert demote_non_english("エヌビディア、新型GPUを発表", None, 1) == 4


class TestWiredIntoNormalisation:
    def test_normalize_applies_it(self):
        import inspect

        from swing.ingest import normalize

        assert "demote_non_english" in inspect.getsource(normalize)
