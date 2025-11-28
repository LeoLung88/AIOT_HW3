from sms_spam.preprocess import clean_text


def test_clean_text_lowercase_and_punctuation():
    assert clean_text("Hello!!!  WORLD??") == "hello world"


def test_clean_text_stopwords_removed():
    text = "This is a simple message for testing"
    assert clean_text(text, remove_stopwords=True) == "simple message testing"
