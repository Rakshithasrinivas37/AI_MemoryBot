import re

STOP_WORDS = {
    "i", "am", "is", "are", "the", "a", "an", "my", "me", "what", "how",
    "who", "do", "did", "was", "were", "have", "has", "you", "your",
    "we", "our", "they", "of", "to", "in", "and", "or", "it", "that",
    "this", "for", "on", "with", "about", "tell",
}


def tokenize(text):
    """Lowercase, split into words, drop punctuation and stop words."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [w for w in words if w not in STOP_WORDS]