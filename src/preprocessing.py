import re
def clean_text(text: str) -> str:
    """
    Clean the input text.

    Steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Remove extra spaces
    """

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text