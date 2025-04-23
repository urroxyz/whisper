from .core import whisperer
from bidi.mirror import MIRRORED

def HYPHEN(count: int = 1) -> str:
    return " " + "-" * count

def GREATER_THAN(count: int = 1) -> str:
    return " " + ">" * count

def SPEAKER(ch: str = "[") -> str:
    mirror = MIRRORED.get(ch, ch)
    return " " + ch + "SPEAKER 1" + mirror

def PERSON(ch: str = "[") -> str:
    mirror = MIRRORED.get(ch, ch)
    return " " + ch + "PERSON 1" + mirror

__license__ = "GPLv3"
__version__ = "0.1.0"
