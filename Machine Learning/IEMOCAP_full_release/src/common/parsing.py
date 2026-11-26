"""Parsing of IEMOCAP label files and IDs. Utterance IDs are parsed from the RIGHT."""
from __future__ import annotations
import re

SUMMARY_RE = re.compile(r"^\[([\d.]+) - ([\d.]+)\]\t(\S+)\t(\S+)\t\[([^\]]*)\]")
VOTE_RE = re.compile(r"^C-([A-Z]\d+):\s*(.*)$")
UTT_RE = re.compile(r"^(?P<dialog>.+)_(?P<spk>[FM])(?P<idx>\d{3})$")
DIALOG_RE = re.compile(r"^(Ses\d\d)(?P<mocap>[FM])_(?P<kind>impro|script)")

CATEGORY_MAP = {"neutral state": "neu", "neutral": "neu", "happiness": "hap", "excited": "exc",
                "sadness": "sad", "anger": "ang", "frustration": "fru", "surprise": "sur",
                "fear": "fea", "disgust": "dis"}


def parse_utt_id(utt_id: str) -> dict:
    """'Ses05M_script01_1b_F000' -> dialog 'Ses05M_script01_1b', speaker 'F', index 0."""
    m = UTT_RE.match(utt_id)
    if not m:
        raise ValueError(f"bad utterance id: {utt_id}")
    dialog = m.group("dialog")
    d = DIALOG_RE.match(dialog)
    if not d:
        raise ValueError(f"bad dialog name: {dialog}")
    session = int(d.group(1)[3:])
    spk = m.group("spk")
    return dict(dialog=dialog, session=session, speaker_letter=spk, index=int(m.group("idx")),
                speaker_id=f"Ses{session:02d}_{spk}", mocap_letter=d.group("mocap"),
                dialog_type=d.group("kind"))


def map_category(raw: str):
    """Raw category string -> 3-letter code, or None if unknown."""
    s = raw.strip().strip(":").strip().lower()
    if s.startswith("other"):
        return "oth"
    return CATEGORY_MAP.get(s)


def rater_type(rater_id: str) -> str:
    """E<n> = external rater; F<n>/M<n> = actor self-evaluation."""
    return "external" if rater_id.startswith("E") else "self"


def split_categories(field: str) -> list[str]:
    """'Frustration; Anger;  (comment)' or ':Frustration; :Anger; (c)' -> ['Frustration','Anger'].
    Free-text comments start at the first '(' (categories never contain one)."""
    field = field.split("(", 1)[0]
    return [p.strip().lstrip(":").strip() for p in field.split(";") if p.strip().lstrip(":").strip()]


def parse_summary_text(text: str) -> list[dict]:
    """Parse one EmoEvaluation summary file. Returns one dict per utterance with its categorical votes."""
    utts, cur = [], None
    for line in text.splitlines():
        m = SUMMARY_RE.match(line)
        if m:
            vad = [float(x) for x in m.group(5).split(",")]
            cur = dict(utt_id=m.group(3), start=float(m.group(1)), end=float(m.group(2)),
                       label_official=m.group(4), val=vad[0], act=vad[1], dom=vad[2], votes=[])
            utts.append(cur)
            continue
        v = VOTE_RE.match(line.strip("\n"))
        if v and cur is not None:
            cur["votes"].append((v.group(1), split_categories(v.group(2))))
    return utts


def parse_cat_text(text: str) -> dict[str, list[str]]:
    """Parse a per-rater *_cat.txt file: 'UTT :Cat; :Cat; (comment)' -> {utt: [raw categories]}."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        utt, _, rest = line.partition(" ")
        out[utt] = split_categories(rest)
    return out


SEG_LINE_RE = re.compile(r"^(?P<seg_id>Ses\d\d[FM]_\S+?_(?P<spk>[FM])(?P<idx>XX\d*|\d{3})) \[(?P<s>[\d.]+)-(?P<e>[\d.]+)\]:")


def parse_transcription_line(line: str):
    """'ID [start-end]: text' -> dict, or None if the line is not a timed segment line."""
    m = SEG_LINE_RE.match(line)
    if not m:
        return None
    return dict(seg_id=m.group("seg_id"), speaker_letter=m.group("spk"),
                start=float(m.group("s")), end=float(m.group("e")))


def parse_lab_text(text: str):
    """'.lab' file: 'start end label' per line; label 'd' marks gaps. Returns [(seg_id, start, end)]."""
    out = []
    for line in text.splitlines():
        p = line.split()
        if len(p) == 3 and p[2] != "d":
            out.append((p[2], float(p[0]), float(p[1])))
    return out
