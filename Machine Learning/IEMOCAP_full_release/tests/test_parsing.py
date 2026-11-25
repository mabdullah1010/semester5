import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.parsing import parse_summary_text, parse_utt_id, map_category

BLOCK = """[19.2900 - 20.7875]\tSes01F_impro01_F003\txxx\t[2.5000, 3.0000, 3.0000]
C-E2:\tNeutral;\t()
C-E3:\tFrustration;\t()
C-E4:\tAnger;\t()
C-F1:\tFrustration; Anger;\t()
A-E3:\tval 3; act 3; dom  3;\t()
"""


def test_documented_example_block():
    (u,) = parse_summary_text(BLOCK)
    assert u["utt_id"] == "Ses01F_impro01_F003"
    assert (u["start"], u["end"], u["label_official"]) == (19.29, 20.7875, "xxx")
    assert (u["val"], u["act"], u["dom"]) == (2.5, 3.0, 3.0)
    assert u["votes"] == [("E2", ["Neutral"]), ("E3", ["Frustration"]), ("E4", ["Anger"]),
                          ("F1", ["Frustration", "Anger"])]


def test_utt_id_parsed_from_right():
    a = parse_utt_id("Ses05M_script01_1b_F000")
    assert (a["dialog"], a["session"], a["speaker_letter"], a["index"], a["speaker_id"],
            a["mocap_letter"], a["dialog_type"]) == ("Ses05M_script01_1b", 5, "F", 0, "Ses05_F", "M", "script")
    b = parse_utt_id("Ses03M_impro05a_M012")
    assert (b["dialog"], b["session"], b["speaker_letter"], b["index"], b["speaker_id"],
            b["mocap_letter"], b["dialog_type"]) == ("Ses03M_impro05a", 3, "M", 12, "Ses03_M", "M", "impro")
    c = parse_utt_id("Ses01F_script01_1_F000")
    assert c["dialog"] == "Ses01F_script01_1" and c["speaker_letter"] == "F"


def test_category_mapping():
    assert map_category("Neutral state") == "neu" and map_category("Neutral") == "neu"
    assert map_category("Other") == "oth" and map_category("Other: bored") == "oth"
    assert map_category("Overlap") is None
