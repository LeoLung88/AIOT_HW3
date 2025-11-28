import re
import string
from typing import Iterable, Optional, Set

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


PUNCT_TRANSLATION = str.maketrans({ch: " " for ch in string.punctuation})
DEFAULT_STOPWORDS: Set[str] = set(ENGLISH_STOP_WORDS)


def clean_text(
    text: str,
    *,
    remove_stopwords: bool = False,
    extra_stopwords: Optional[Iterable[str]] = None,
) -> str:
    """Lowercase, remove punctuation, collapse whitespace, and optionally drop stopwords."""
    if text is None:
        text = ""
    normalized = str(text).lower()
    no_punct = normalized.translate(PUNCT_TRANSLATION)
    collapsed = re.sub(r"\s+", " ", no_punct).strip()

    if not remove_stopwords:
        return collapsed

    stopwords = DEFAULT_STOPWORDS.union(set(extra_stopwords or []))
    tokens = [token for token in collapsed.split(" ") if token and token not in stopwords]
    return " ".join(tokens)


def clean_corpus(
    texts: Iterable[str],
    *,
    remove_stopwords: bool = False,
    extra_stopwords: Optional[Iterable[str]] = None,
) -> list[str]:
    """Apply clean_text across an iterable of texts."""
    return [
        clean_text(text, remove_stopwords=remove_stopwords, extra_stopwords=extra_stopwords)
        for text in texts
    ]
