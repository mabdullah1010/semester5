# COM 307 
# Project 2 
# Naive Bayes classifier for wrongful convictions

# Muhammad Abdullah
# October 7th 2026
# Due on October 12th


# Features used:  offense (OFFGUIDE categories), state, race (NEWRACE categories)

import os
import numpy as np
import pandas as pd


# three runs included the write-up.
#   output_all_exonerations/   FEDERAL_ONLY = False, ALPHA = 1.0
#   output_federal_alpha1/     FEDERAL_ONLY = True,  ALPHA = 1.0
#   output_federal_alpha0.1/   FEDERAL_ONLY = True,  ALPHA = 0.1



USSC_FILE = "opafy18nid.csv"

SLIM_FILE = "ussc2018_slim.csv"          # small copy of the file

EXON_FILE = "publicspreadsheet.csv"



ALPHA = 0.1              # Laplace smoothing


FEDERAL_ONLY = True     # if true then keep only federal exonerations (114 cases)


if FEDERAL_ONLY:
    OUT_DIR = f"output_federal_alpha{ALPHA:g}"
else:
    OUT_DIR = f"output_all_exonerations_alpha{ALPHA:g}"


CONVICTED_FROM = 1989
CONVICTED_TO = 2018  

SCALE_CONVICTIONS = True # treat 2018 as a typical year and multiply by the
                         # number of years in the window (has an impact on the prior)

# also tried with 0.1

ANNUALIZE = 1.0          # initially I wrote code for the file on moodle whcih did not include the full dataset
                         # truncated moodle copy covered only Oct - mid Dec 2017
                         # that copy would need 12 / 2.2 = 5.45.

# Lookup tables (from the USSC codebook)

OFFGUIDE = {
    1: "Administration of Justice", 2: "Antitrust", 3: "Arson", 4: "Assault",
    5: "Bribery/Corruption", 6: "Burglary/Trespass", 7: "Child Pornography",
    8: "Commercialized Vice", 9: "Drug Possession", 10: "Drug Trafficking",
    11: "Environmental", 12: "Extortion/Racketeering", 13: "Firearms",
    14: "Food and Drug", 15: "Forgery/Counter/Copyright",
    16: "Fraud/Theft/Embezzlement", 17: "Immigration", 18: "Individual Rights",
    19: "Kidnapping", 20: "Manslaughter", 21: "Money Launder", 22: "Murder",
    23: "National Defense", 24: "Obscenity/Other Sex Offenses",
    25: "Prison Offenses", 26: "Robbery", 27: "Sex Abuse",
    28: "Stalking/Harassing", 29: "Tax", 30: "Other",
}


NEWRACE = {1: "White", 2: "Black", 3: "Hispanic", 6: "Other"}

DISTRICT_TO_STATE = {
    0: "Maine", 1: "Massachusetts", 2: "New Hampshire", 3: "Rhode Island",
    4: "Puerto Rico", 5: "Connecticut", 6: "New York", 7: "New York",
    8: "New York", 9: "New York", 10: "Vermont", 11: "Delaware",
    12: "New Jersey", 13: "Pennsylvania", 14: "Pennsylvania",
    15: "Pennsylvania", 16: "Maryland", 17: "North Carolina",
    18: "North Carolina", 19: "North Carolina", 20: "South Carolina",
    22: "Virginia", 23: "Virginia", 24: "West Virginia", 25: "West Virginia",
    26: "Alabama", 27: "Alabama", 28: "Alabama", 29: "Florida", 30: "Florida",
    31: "Florida", 32: "Georgia", 33: "Georgia", 34: "Georgia",
    35: "Louisiana", 36: "Louisiana", 37: "Mississippi", 38: "Mississippi",
    39: "Texas", 40: "Texas", 41: "Texas", 42: "Texas", 43: "Kentucky",
    44: "Kentucky", 45: "Michigan", 46: "Michigan", 47: "Ohio", 48: "Ohio",
    49: "Tennessee", 50: "Tennessee", 51: "Tennessee", 52: "Illinois",
    53: "Illinois", 54: "Illinois", 55: "Indiana", 56: "Indiana",
    57: "Wisconsin", 58: "Wisconsin", 60: "Arkansas", 61: "Arkansas",
    62: "Iowa", 63: "Iowa", 64: "Minnesota", 65: "Missouri", 66: "Missouri",
    67: "Nebraska", 68: "North Dakota", 69: "South Dakota", 70: "Arizona",
    71: "California", 72: "California", 73: "California", 74: "California",
    75: "Hawaii", 76: "Idaho", 77: "Montana", 78: "Nevada", 79: "Oregon",
    80: "Washington", 81: "Washington", 82: "Colorado", 83: "Kansas",
    84: "New Mexico", 85: "Oklahoma", 86: "Oklahoma", 87: "Oklahoma",
    88: "Utah", 89: "Wyoming", 90: "District of Columbia",
    91: "Virgin Islands", 93: "Guam", 94: "Northern Mariana Islands",
    95: "Alaska", 96: "Louisiana",
}

FED_ABBR = {
    "AL": "Alabama", "AZ": "Arizona", "CA": "California", "CT": "Connecticut",
    "DC": "District of Columbia", "FL": "Florida", "IL": "Illinois",
    "KY": "Kentucky", "LA": "Louisiana", "MA": "Massachusetts",
    "MD": "Maryland", "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri",
    "MT": "Montana", "ND": "North Dakota", "NJ": "New Jersey",
    "NM": "New Mexico", "NV": "Nevada", "NY": "New York", "OH": "Ohio",
    "PA": "Pennsylvania", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "VA": "Virginia", "WA": "Washington", "WI": "Wisconsin",
}

# Registry "Worst Crime Display" --> OFFGUIDE code
CRIME_TO_OFFGUIDE = {
    "Murder": 22,
    "Accessory to Murder": 1,          
    "Manslaughter": 20,
    "Attempted Murder": 4,           
    "Attempt, Violent": 4,
    "Assault": 4,
    "Child Abuse": 4,                  # judgment call
    "Dependent Adult Abuse": 4,        # judgment call
    "Menacing": 4,                     # judgment call
    "Other Violent Felony": 4,         # judgment call
    "Sexual Assault": 27,
    "Child Sex Abuse": 27,
    "Sex Offender Registration": 27,   
    "Drug Possession or Sale": 10,  
    "Robbery": 26,
    "Kidnapping": 19,
    "Arson": 3,
    "Burglary/Unlawful Entry": 6,
    "Fraud": 16,
    "Theft": 16,
    "Possession of Stolen Property": 16,
    "Forgery": 15,
    "Tax Evasion/Fraud": 29,
    "Bribery": 5,
    "Official Misconduct": 5,          # judgment call
    "Perjury": 1,
    "Obstruction of Justice": 1,
    "Filing a False Report": 1,
    "Weapon Possession or Sale": 13,
    "Immigration": 17,
    "Supporting Terrorism": 23,
    "Threats": 28,
    "Stalking": 28,
    "Harassment": 28,
    "Solicitation": 30,             
    "Destruction of Property": 30,
    "Traffic Offense": 30,
    "Failure to Pay Child Support": 30,
    "Military Justice Offense": 30,
    "Other Nonviolent Felony": 30,
    "Other Nonviolent Misdemeanor": 30,
    "Other": 30,
}

REGISTRY_RACE = {"White": "White", "Black": "Black", "Hispanic": "Hispanic",
                 "Asian": "Other", "Native American": "Other", "Other": "Other"}

FEATURES = ["offense", "state", "race"]



def load_ussc():
    if os.path.exists(SLIM_FILE):
        raw = pd.read_csv(SLIM_FILE)
    else:
        cols = ["USSCIDN", "OFFGUIDE", "DISTRICT", "MONRACE", "HISPORIG",
                "NEWRACE", "MONSEX"]
        raw = pd.read_csv(USSC_FILE, usecols=cols, low_memory=False)
        raw.to_csv(SLIM_FILE, index=False)

    print(f"USSC rows loaded: {len(raw)}")
    for c in ["OFFGUIDE", "DISTRICT", "NEWRACE"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")  # "." become NaN

    df = pd.DataFrame({
        "offense": raw["OFFGUIDE"].map(OFFGUIDE),
        "state": raw["DISTRICT"].map(DISTRICT_TO_STATE),
        "race": raw["NEWRACE"].map(NEWRACE),
    })


    before = len(df)
    df = df.dropna()


    print(f"USSC rows dropped for missing/unknown codes: {before - len(df)}")
    print(f"USSC distinct districts present: {raw['DISTRICT'].nunique()}")
    return df


def load_exonerations():
    raw = pd.read_csv(EXON_FILE, encoding="cp1252")

    print(f"Exoneration rows loaded: {len(raw)}")
    

    raw = raw[(raw["Convicted"] >= CONVICTED_FROM) & (raw["Convicted"] <= CONVICTED_TO)]
    print(f"  in conviction window {CONVICTED_FROM}-{CONVICTED_TO}: {len(raw)}")

    is_fed = raw["State"].str.startswith("Fed")
    if FEDERAL_ONLY:
        raw = raw[is_fed]
        print(f"  federal only: {len(raw)}")

    def to_state(s):
        if s.startswith("Fed-"):
            return FED_ABBR.get(s[4:])   #Fed-Military give None (no state)
        return s

    df = pd.DataFrame({
        "offense": raw["Worst Crime Display"].map(CRIME_TO_OFFGUIDE).map(OFFGUIDE),
        "state": raw["State"].map(to_state),
        "race": raw["Race"].map(REGISTRY_RACE),
        "federal": is_fed[raw.index],
        "convicted": raw["Convicted"],
    })
    unmapped = raw.loc[df["offense"].isna(), "Worst Crime Display"].unique()
    if len(unmapped):
        print(f"  WARNING unmapped crimes: {list(unmapped)}")
    before = len(df)
    df = df.dropna(subset=FEATURES)
    print(f"  dropped (unmapped/military): {before - len(df)}")
    return df


# Naive Bayes

def fit(exon, conv, n_conv_effective):

    n_exon = len(exon)
    prior = {"EXON": n_exon / (n_exon + n_conv_effective),
             "CONV": n_conv_effective / (n_exon + n_conv_effective)}


    tables = {}
    for f in FEATURES:
        values = sorted(set(exon[f]) | set(conv[f]))
        k = len(values)
        ce = exon[f].value_counts().reindex(values, fill_value=0)
        cc = conv[f].value_counts().reindex(values, fill_value=0)


        t = pd.DataFrame({
            "count_exon": ce,
            "count_conv": cc,
            "P(x|EXON)": (ce + ALPHA) / (ce.sum() + ALPHA * k),
            "P(x|CONV)": (cc + ALPHA) / (cc.sum() + ALPHA * k),
        })


        t["likelihood_ratio"] = t["P(x|EXON)"] / t["P(x|CONV)"]
        t.index.name = f


        tables[f] = t.sort_values("likelihood_ratio", ascending=False)
    return prior, tables


def posterior(prior, tables, offense, state, race):

    x = {"offense": offense, "state": state, "race": race}
    log_e, log_c = np.log(prior["EXON"]), np.log(prior["CONV"])


    for f in FEATURES:
        t = tables[f]
        if x[f] not in t.index:
            raise ValueError(f"unknown {f}: {x[f]}")
        log_e += np.log(t.at[x[f], "P(x|EXON)"])
        log_c += np.log(t.at[x[f], "P(x|CONV)"])


    m = max(log_e, log_c)
    pe = np.exp(log_e - m)
    pc = np.exp(log_c - m)
    return pe / (pe + pc)




def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    conv = load_ussc()
    exon = load_exonerations()


    # "Convictions that are not exonerations" = total 2018 convictions minus federal exonerations of people convicted in 2018

    overlap = exon[exon["federal"] & (exon["convicted"] == 2018)]

    print(f"Federal exonerations convicted in 2018 (subtracted): {len(overlap)}")

    for _, row in overlap.iterrows():
        match = conv.index[(conv["offense"] == row["offense"]) &
                           (conv["state"] == row["state"]) &
                           (conv["race"] == row["race"])]
        
        if len(match):
            conv = conv.drop(match[0])

    if SCALE_CONVICTIONS:
        years = CONVICTED_TO - CONVICTED_FROM + 1
    else:
        years = 1



    n_conv_eff = len(conv) * ANNUALIZE * years

    print(f"\nExonerated: {len(exon)}   2018 convictions in file: {len(conv)}   "
          f"x {ANNUALIZE} (to a full year) x {years} years = "
          f"{n_conv_eff:,.0f} effective convictions")

    prior, tables = fit(exon, conv, n_conv_eff)

    # save probabilities
    pd.Series(prior, name="prior").to_csv(f"{OUT_DIR}/priors.csv")

    for f, t in tables.items():
        t.to_csv(f"{OUT_DIR}/likelihoods_{f}.csv", float_format="%.6g")


    rows = []

    for o in tables["offense"].index:

        for s in tables["state"].index:

            for r in tables["race"].index:

                p = posterior(prior, tables, o, s, r)

                rows.append((o, s, r, p, p / prior["EXON"]))


    allp = pd.DataFrame(rows, columns=["offense", "state", "race",
                                       "P(EXON|x)", "times_base_rate"])
    allp.sort_values("P(EXON|x)", ascending=False).to_csv(
        f"{OUT_DIR}/all_posteriors.csv", index=False, float_format="%.6g")




    # Summary
    print(f"Prior P(EXON) = {prior['EXON']:.6f}")


    for f, t in tables.items():
        print(f"Top likelihood ratios for {f}:")
        print(t[["count_exon", "count_conv", "likelihood_ratio"]].head(8)
              .to_string(float_format="%.3f"))

    examples = [("Murder", "Illinois", "Black"),
                ("Murder", "Texas", "White"),
                ("Drug Trafficking", "Texas", "Hispanic"),
                ("Immigration", "Texas", "Hispanic"),
                ("Fraud/Theft/Embezzlement", "New York", "White"),
                ("Sex Abuse", "Connecticut", "Black")]

    
    print("Example posteriors:")

    for e in examples:
        p = posterior(prior, tables, *e)
        print(f"  {e}: P(EXON) = {p:.6f}  ({p / prior['EXON']:.1f}x base rate)")


    print(f"All tables written to {OUT_DIR}/")


main()