# bank_detect.py
import re

def detect_bank(big_text: str) -> str:
    """
    Return a simple bank code like 'CIBC', 'RBC', 'SCOTIA', 'TD', 'BMO'.
    If nothing matches, return 'UNKNOWN'.
    """
    t = big_text.upper()
    if "CIBC" in t:
        return "CIBC"
    if re.search(r"\bROYAL BANK\b|\bRBC\b", t):
        return "RBC"
    if "SCOTIABANK" in t or "SCOTIA" in t:
        return "SCOTIA"
    if "TORONTO-DOMINION" in t or re.search(r"\bTD\b", t):
        return "TD"
    if "BANK OF MONTREAL" in t or "BMO" in t:
        return "BMO"
    return "UNKNOWN"
