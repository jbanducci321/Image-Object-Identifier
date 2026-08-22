"""
query_parser.py

Rule-based extraction of a short object phrase from a full sentence.

OWL-ViT's text encoder was trained on short noun phrases like "a photo of a
red umbrella", not conversational requests. If we feed it the raw sentence
"please find the red umbrella", the extra words dilute the text embedding
and hurt detection quality. This module strips known lead-in phrases so the
model only ever sees the object phrase itself.

No LLM involved — just string matching.
"""

import re

# Ordered longest-first so a more specific phrase is stripped before a
# shorter one that could also match as a prefix of it.
_LEAD_IN_PHRASES = [
    "can you please find",
    "could you please find",
    "can you find me",
    "could you find me",
    "can you find",
    "could you find",
    "can you see",
    "can you spot",
    "can you locate",
    "do you see",
    "please find me",
    "please find",
    "please locate",
    "please look for",
    "find me the",
    "find me",
    "find the",
    "find",
    "look for the",
    "look for",
    "locate the",
    "locate",
    "spot the",
    "spot",
    "search for the",
    "search for",
    "where is the",
    "where's the",
    "where is",
    "where's",
    "i'm looking for the",
    "i'm looking for",
    "i am looking for the",
    "i am looking for",
    "show me the",
    "show me",
]

_LEADING_ARTICLES = ("a ", "an ", "the ")


def extract_object_phrase(sentence: str) -> str:
    """Extract a short object phrase from a full sentence.

    Example: "please find the red umbrella" -> "red umbrella"
    """
    text = sentence.strip().lower()

    # Drop trailing punctuation like "?" or "."
    text = re.sub(r"[?.!]+$", "", text).strip()

    # Repeatedly strip lead-in phrases in case of stacked phrasing
    # (e.g. "can you please find where is the umbrella").
    changed = True
    while changed:
        changed = False
        for phrase in _LEAD_IN_PHRASES:
            if text.startswith(phrase + " "):
                text = text[len(phrase):].strip()
                changed = True
                break
            if text == phrase:
                text = ""
                changed = True
                break

    # Strip a single leading article left over after removing the lead-in
    # (e.g. "find a red umbrella" -> "a red umbrella" -> "red umbrella").
    for article in _LEADING_ARTICLES:
        if text.startswith(article):
            text = text[len(article):].strip()
            break

    return text if text else sentence.strip().lower()
