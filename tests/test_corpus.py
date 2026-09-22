from src.paths import LEXICON
from src.prompt import FRAGMENT_COUNT, load_fragments


def test_shipped_corpus_has_expected_size() -> None:
    fragments = load_fragments(LEXICON)
    assert len(fragments) == FRAGMENT_COUNT
