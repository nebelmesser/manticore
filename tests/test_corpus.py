from src.paths import NEGATIVE, POSITIVE
from src.prompt import FRAGMENT_COUNT, choose_negative, load_fragments


def test_shipped_corpus_has_expected_size() -> None:
    fragments = load_fragments(POSITIVE)
    assert len(fragments) == FRAGMENT_COUNT


def test_shipped_negative_is_the_former_default() -> None:
    assert choose_negative(NEGATIVE) == (
        "text letters label title panels comics captions subtitle frame border squares"
    )
