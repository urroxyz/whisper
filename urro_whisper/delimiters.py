from bidi.mirror import MIRRORED

def HYPHEN(count: int = 1) -> str:
    return " " + "-" * count

HYPHEN.regex = r"(\s)?\-\s"

def GREATER_THAN(count: int = 1) -> str:
    return " " + ">" * count

GREATER_THAN.regex = r"(\s)?\>\s"

def SPEAKER(ch: str = "[", short: bool = False) -> str:
    mirror = MIRRORED.get(ch, ch)
    if not short:
        return " " + ch + "SPEAKER 1" + mirror
    else:
        return " " + ch + "S1" + mirror

SPEAKER.regex = r"(\s)?\[SPEAKER\s\d\]\s"

def PERSON(ch: str = "[", short: bool = False) -> str:
    mirror = MIRRORED.get(ch, ch)
    if not short:
        return " " + ch + "PERSON 1" + mirror
    else:
        return " " + ch + "P1" + mirror

PERSON.regex = r"(\s)?\[PERSON\s\d\]\s"
