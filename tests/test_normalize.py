"""Normalization views (plan.md Step 2)."""

import pandas as pd
import pytest

from ber.normalize import (clean_text, no_space, normalize_frame, script_code,
                           strip_legal, transliterate)


@pytest.mark.parametrize("raw, expected", [
    ("-- Holloway Peak Inc Seafood", "holloway peak inc seafood"),
    ("Café Élysée SARL", "cafe elysee sarl"),
    ("Straße Œuvre", "strasse oeuvre"),
    ("KAIROSSONS.COM", "kairossons com"),
    ("Kairos & Sons", "kairos and sons"),
    ("#201 & 202, PRESTIGE LOKA", "201 and 202 prestige loka"),
    ("null", ""), ("  N/A ", ""), ("--", ""), ("NULL", ""),
    ("४५ मार्ग", "45 मार्ग"),                       # Devanagari digits -> ASCII
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_legal_and_nospace():
    assert strip_legal("ap hospitality inc") == "ap hospitality"
    assert strip_legal("the limited") == "the limited"          # never empties a name
    assert no_space("kairossons com") == "kairossons"
    assert no_space("www guha consultants com") == "guhaconsultants"


@pytest.mark.parametrize("raw, expected", [
    ("विजन फूड प्राइवेट लिमिटेड", "vijan fud praivet limited"),     # Devanagari
    ("ಸೂರ್ಯ", "surya"),                                          # Kannada keeps final a
    ("బాబా స్కై", "baba skai"),                                   # Telugu
    ("ஸ்டார்", "star"),                                           # Tamil
    ("ലിമിറ്റഡ്", "limittad"),                                     # Malayalam rra-rra = tt
    ("abc 12", "abc 12"),                                        # Latin passes through
])
def test_transliterate(raw, expected):
    assert transliterate(clean_text(raw)) == expected


def test_script_code():
    assert script_code("abc") == 0
    assert script_code("विजन") == 1      # Devanagari
    assert script_code("ലിമി") == 9      # Malayalam


def test_normalize_frame_columns():
    df = pd.DataFrame({"entity_id": ["S2-1"], "business_name": ["रेड बिल्डर्स प्रा. लि."],
                       "business_address": ["DOOR NO 01/5, MUMBAI"], "country": ["India"]})
    out = normalize_frame(df).iloc[0]
    assert out["script"] == 1
    assert out["name_tr"] == "red bildars pra li"
    assert out["addr_digits"] == "01 5"
    assert strip_legal(out["name_tr"]) == "red bildars"
