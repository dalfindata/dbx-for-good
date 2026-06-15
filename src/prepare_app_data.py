"""
Medical Desert Planner — data preparation.

Builds the two tables the Streamlit app reads:
  app_data/district_gaps.parquet      one row per NFHS-5 district: per-capita care-gap score,
                                      NFHS driver indicators, supply, population, confidence flags, map centroid.
  app_data/facility_citations.parquet one row per facility (assigned to its NFHS district) with the
                                      underlying text used as evidence (description / capability / specialties / sources).

Trust decisions encoded here (see README):
  - facilities deduped by the dataset's own entity-resolution key (cluster_id)
  - facilities geocoded to a district by NEAREST POST OFFICE (coordinates), not by self-reported pincode string
  - supply expressed PER 100k using Census-2011 district population (the denominator the naive count lacked)
  - every weak value is FLAGGED (supply_confidence, pop_match) rather than hidden
  - NFHS '(x)' small-sample values and '*' suppressed values are tracked, not silently treated as solid

Run from the repo root (the folder that contains the source CSVs):
    python medical_desert_planner/prepare_app_data.py
"""
import os, re, json, difflib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)            # parent folder holds the source CSVs
OUT = os.path.join(HERE, "app_data")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------- helpers
def norm(s):
    if pd.isna(s):
        return None
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower().strip())
    s = re.sub(r"\s+", " ", s).strip()
    return s or None

_STATE_FIX = {
    "maharastra": "maharashtra", "nct of delhi": "delhi", "orissa": "odisha",
    "pondicherry": "puducherry", "jammu kashmir": "jammu and kashmir",
    "andaman nicobar islands": "andaman and nicobar islands",
    "uttaranchal": "uttarakhand", "chattisgarh": "chhattisgarh",
}
def snorm(s):
    n = norm(s)
    if n is None:
        return None
    n = re.sub(r"\s+", " ", n.replace("&", "and")).strip()
    return _STATE_FIX.get(n, n)

def to_num(x):
    """NFHS value -> float. '(x)' = small unweighted sample, '*' = suppressed."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("(", "").replace(")", "").replace(",", "")
    if s in ("*", "", "-", "na", "nan"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

def is_flagged(x):
    """True if the NFHS cell is a small-sample '(x)' value (reliability caveat)."""
    return bool(re.search(r"\(", str(x))) if pd.notna(x) else False

def ffloat(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan

# ----------------------------------------------------------------------------- load
fac = pd.read_csv(f"{SRC}/facilities.csv", dtype=str, low_memory=False)
pin = pd.read_csv(f"{SRC}/india_post_pincode_directory.csv", dtype=str, low_memory=False)
nfhs = pd.read_csv(f"{SRC}/nfhs_5_district_health_indicators.csv", dtype=str, low_memory=False)
pop = pd.read_csv(f"{SRC}/census2011_district_population.csv")

# ----------------------------------------------------------------------------- 0. data-readiness scorecard (Track 4 lens)
# Audit the facility dataset AS DELIVERED (before our fixes) so the app can show *what had to be
# repaired before planning could trust it*. Written to app_data/data_readiness.json.
_N = len(fac)
def _cov(col):
    if col not in fac.columns:
        return 0
    return int(fac[col].apply(
        lambda x: isinstance(x, str) and x.strip().lower() not in ("", "nan", "none", "null")).sum())

# key fields grouped by the role each plays for a planner
_FIELD_ROLE = [
    ("description", "Evidence text (cite)"), ("capability", "Evidence text (cite)"),
    ("specialties", "Evidence text (cite)"), ("procedure", "Evidence text (cite)"),
    ("equipment", "Evidence text (cite)"),
    ("latitude", "Geography"), ("longitude", "Geography"),
    ("operatorTypeId", "Ownership (bias)"), ("facilityTypeId", "Facility-size proxy"),
    ("capacity", "Capacity (weighting)"), ("numberDoctors", "Capacity (weighting)"),
    ("yearEstablished", "Capacity (weighting)"),
]
_coverage = [{"field": c, "role": role, "n": _cov(c), "pct": round(100 * _cov(c) / _N, 1)}
             for c, role in _FIELD_ROLE]

# ownership split + column-bleed: operatorTypeId must be private/public/government — anything else
# is a CSV parse artifact (unescaped commas/quotes/newlines in the source text shift values sideways)
_VALID_OWN = {"private", "public", "government"}
_own = fac["operatorTypeId"].fillna("").str.strip().str.lower()
_own_counts = {k: int((_own == k).sum()) for k in _VALID_OWN}
_bleed = (~_own.isin(_VALID_OWN)) & (_own != "")
_bleed_examples = [{
    "unique_id": str(r.get("unique_id"))[:40],
    "name": str(r.get("name"))[:60] if pd.notna(r.get("name")) else "(unnamed)",
    "leaked_value": str(r.get("operatorTypeId"))[:160],
} for _, r in fac[_bleed].head(6).iterrows()]

# geocoding readiness
_lat = fac["latitude"].map(ffloat); _lon = fac["longitude"].map(ffloat)
_has = _lat.notna() & _lon.notna()
_inb = _has & _lat.between(6, 38) & _lon.between(68, 98)

# need-matching coverage (scan the evidence text)
_blob = (fac[["specialties", "capability", "description", "procedure"]]
         .fillna("").astype(str).agg(" ".join, axis=1).str.lower())
_mat = int(_blob.str.contains(r"obstet|ob/gyn|gyn|maternit|delivery|natal", regex=True).sum())
_ped = int(_blob.str.contains(r"pediatr|paediatr|neonat|child", regex=True).sum())

_readiness = {
    "n_rows": _N,
    "coverage": _coverage,
    "duplicates": {
        "cluster_id": int(_N - fac["cluster_id"].nunique(dropna=True)),
        "unique_id": int(_N - fac["unique_id"].nunique(dropna=True)),
    },
    "geocoding": {
        "has_coords": int(_has.sum()), "has_coords_pct": round(100 * _has.sum() / _N, 1),
        "in_india": int(_inb.sum()), "out_of_range": int((_has & ~_inb).sum()),
        "no_coords": int((~_has).sum()),
    },
    "ownership": {**_own_counts, "missing_or_leaked": int((_own == "").sum()) + int(_bleed.sum()),
                  "private_pct": round(100 * _own_counts["private"] / _N, 1)},
    "column_bleed": {"n_bad": int(_bleed.sum()), "examples": _bleed_examples},
    "specialty": {"maternal_pct": round(100 * _mat / _N, 1),
                  "pediatric_pct": round(100 * _ped / _N, 1)},
}
with open(f"{OUT}/data_readiness.json", "w", encoding="utf-8") as _fh:
    json.dump(_readiness, _fh, indent=2)
print(f"[readiness] wrote data_readiness.json  (column-bleed rows={_readiness['column_bleed']['n_bad']})")

# ----------------------------------------------------------------------------- 1. dedupe
before = len(fac)
m = fac["cluster_id"].notna()
fac = pd.concat([fac[m].drop_duplicates("cluster_id"), fac[~m]], ignore_index=True)
print(f"[dedupe] {before} -> {len(fac)} rows")

# ----------------------------------------------------------------------------- 2. coordinate geocoding (nearest post office)
pin["la"] = pin["latitude"].map(ffloat); pin["lo"] = pin["longitude"].map(ffloat)
pg = pin.dropna(subset=["la", "lo", "district", "statename"]).copy()
pg = pg[(pg.la.between(6, 38)) & (pg.lo.between(67, 98))]
tree = cKDTree(pg[["la", "lo"]].values)
po_state = pg["statename"].map(snorm).values
po_dist = pg["district"].map(norm).values

fac["la"] = fac["latitude"].map(ffloat); fac["lo"] = fac["longitude"].map(ffloat)
good = (fac.la.between(6, 38)) & (fac.lo.between(67, 98))
_, idx = tree.query(fac.loc[good, ["la", "lo"]].values, k=1)
fac["geo_state"] = np.array([None] * len(fac), dtype=object)
fac["geo_dist"] = np.array([None] * len(fac), dtype=object)
fac.loc[good, "geo_state"] = po_state[idx]
fac.loc[good, "geo_dist"] = po_dist[idx]

# pincode-string fallback for facilities without usable coordinates
pin["pin6"] = pin["pincode"].str.extract(r"(\d{6})")
fac["pin6"] = fac["address_zipOrPostcode"].str.extract(r"(\d{6})")
p2 = pin.dropna(subset=["pin6", "district", "statename"]).copy()
p2["s"] = p2["statename"].map(snorm); p2["d"] = p2["district"].map(norm)
dom = (p2.groupby(["pin6", "s", "d"]).size().reset_index(name="c")
         .sort_values("c").drop_duplicates("pin6", keep="last"))
fb = fac["geo_dist"].isna()
fac.loc[fb, "geo_state"] = fac.loc[fb, "pin6"].map(dom.set_index("pin6")["s"])
fac.loc[fb, "geo_dist"] = fac.loc[fb, "pin6"].map(dom.set_index("pin6")["d"])
print(f"[geocode] coords={int(good.sum())}, fallback={int((fb & fac['geo_dist'].notna()).sum())}, "
      f"unplaced={int(fac['geo_dist'].isna().sum())}")

# ----------------------------------------------------------------------------- 3. assign each facility to an NFHS district
nfhs["s"] = nfhs["state_ut"].map(snorm)
nfhs["d"] = nfhs["district_name"].map(norm)
# per-state index of NFHS districts {state: {district_norm: nfhs_row_index}}
by_state = {}
for i, r in nfhs.iterrows():
    by_state.setdefault(r["s"], {})[r["d"]] = i

def assign_nfhs(gs, gd):
    d = by_state.get(gs)
    if not d:
        return np.nan
    if gd in d:
        return d[gd]
    mm = difflib.get_close_matches(gd, list(d), n=1, cutoff=0.85)
    return d[mm[0]] if mm else np.nan

fac["nfhs_idx"] = [assign_nfhs(s, d) for s, d in zip(fac["geo_state"], fac["geo_dist"])]
placed = fac["geo_dist"].notna().sum()
matched = fac["nfhs_idx"].notna().sum()
print(f"[assign] facilities matched to an NFHS district: {matched} / placed {placed} "
      f"(orphans {placed - matched})")

def has_kw(s, *kw):
    return bool(s) and any(k in str(s).lower() for k in kw)
fac["is_hospital"] = fac["facilityTypeId"].eq("hospital")
fac["is_obgyn"] = fac["specialties"].apply(lambda s: has_kw(s, "obstet", "gyn", "matern"))

# supply per NFHS district
sup = (fac.dropna(subset=["nfhs_idx"]).assign(nfhs_idx=lambda d: d["nfhs_idx"].astype(int))
       .groupby("nfhs_idx")
       .agg(n_fac=("unique_id", "size"),
            n_hosp=("is_hospital", "sum"),
            n_obgyn=("is_obgyn", "sum")))

# centroid per NFHS district from its facilities; pincode centroid as fallback for zero-facility districts
fac_cent = (fac.dropna(subset=["nfhs_idx", "la", "lo"]).assign(nfhs_idx=lambda d: d["nfhs_idx"].astype(int))
            .groupby("nfhs_idx").agg(lat=("la", "mean"), lon=("lo", "mean")))
pin_cent = pg.assign(s=pg["statename"].map(snorm), d=pg["district"].map(norm)) \
             .groupby(["s", "d"]).agg(lat=("la", "mean"), lon=("lo", "mean"))

# ----------------------------------------------------------------------------- 4. population join (within-state -> nationwide fuzzy)
pop["s"] = pop["State name"].map(snorm); pop["d"] = pop["District name"].map(norm)
pbs = {}
for _, x in pop.iterrows():
    pbs.setdefault(x["s"], {})[x["d"]] = x["Population"]
all_pd = pop.set_index("d")["Population"].to_dict()
all_d = list(all_pd)

def lk_pop(st, d):
    dd = pbs.get(st, {})
    if d in dd:
        return dd[d], "exact"
    mm = difflib.get_close_matches(d, list(dd), n=1, cutoff=0.85) if dd else []
    if mm:
        return dd[mm[0]], "fuzzy_state"
    mm = difflib.get_close_matches(d, all_d, n=1, cutoff=0.9)   # Telangana/Ladakh under old parents
    return (all_pd[mm[0]], "fuzzy_nat") if mm else (np.nan, "unresolved")

# ----------------------------------------------------------------------------- 5. need index
DEFICIT = {  # higher coverage = better -> deficit = 100 - value
    "anc4": "mothers_who_had_at_least_4_anc_visits_lb5y_pct",
    "inst_birth": "institutional_birth_5y_pct",
    "skilled_birth": "births_attended_by_skilled_hp_5y_10_pct",
    "pnc": "mothers_who_received_pnc_from_a_doctor_nurse_lhv_anm_midwif_pct",
    "full_immun": "child_12_23m_fully_vaccinated_based_on_information_from_eit_pct",
    "insurance": "hh_member_covered_health_insurance_pct",
    "ari_careseek": "children_with_fever_or_symptoms_of_ari_2wk_taken_to_a_healt_pct",
    "cerv_screen": "women_age_30_49_years_ever_undergone_a_cervical_screen_pct",
}
BURDEN = {  # higher value = worse
    "unmet_fp": "fp_unmet_total_cm_w15_49_7_pct",
    "stunting": "child_u5_who_are_stunted_height_for_age_18_pct",
    "wasting": "child_u5_who_are_wasted_weight_for_height_18_pct",
    "underweight": "child_u5_who_are_underweight_weight_for_age_18_pct",
    "child_anaemia": "child_6_59m_who_are_anaemic_lt_11_0_g_dl_22_pct",
    "women_anaemia": "all_w15_49_who_are_anaemic_pct",
}
MATERNAL = ["anc4", "inst_birth", "skilled_birth", "pnc"]
# friendly labels + direction for the UI ("low"=low coverage is bad, "high"=high value is bad)
LABELS = {
    "anc4": ("4+ antenatal care visits", "low", "%"),
    "inst_birth": ("Institutional births", "low", "%"),
    "skilled_birth": ("Births w/ skilled attendant", "low", "%"),
    "pnc": ("Postnatal care from skilled provider", "low", "%"),
    "full_immun": ("Children fully immunised", "low", "%"),
    "insurance": ("Households w/ health insurance", "low", "%"),
    "ari_careseek": ("Child ARI/fever taken to facility", "low", "%"),
    "cerv_screen": ("Women ever had cervical screen", "low", "%"),
    "unmet_fp": ("Unmet need for family planning", "high", "%"),
    "stunting": ("Children stunted (u5)", "high", "%"),
    "wasting": ("Children wasted (u5)", "high", "%"),
    "underweight": ("Children underweight (u5)", "high", "%"),
    "child_anaemia": ("Children anaemic (6-59m)", "high", "%"),
    "women_anaemia": ("Women anaemic (15-49)", "high", "%"),
}
worse = pd.DataFrame(index=nfhs.index)
raw = pd.DataFrame(index=nfhs.index)
flag = pd.DataFrame(index=nfhs.index)
for k, c in DEFICIT.items():
    raw[k] = nfhs[c].map(to_num); worse[k] = 100 - raw[k]; flag[k] = nfhs[c].map(is_flagged)
for k, c in BURDEN.items():
    raw[k] = nfhs[c].map(to_num); worse[k] = raw[k]; flag[k] = nfhs[c].map(is_flagged)

def z(s):
    return (s - s.mean()) / s.std(ddof=0)
Z = worse.apply(z)

out = nfhs[["district_name", "state_ut"]].copy()
out["access_z"] = Z[list(DEFICIT) + ["unmet_fp"]].mean(axis=1, skipna=True)
out["burden_z"] = Z[[k for k in BURDEN if k != "unmet_fp"]].mean(axis=1, skipna=True)
out["maternal_deficit_z"] = Z[MATERNAL].mean(axis=1, skipna=True)
out["need_z"] = Z.mean(axis=1, skipna=True)
out["need_valid"] = Z.notna().sum(axis=1)
out["n_flagged_indicators"] = flag.sum(axis=1)

# attach raw indicator values (for the driver panel) as a packed dict-per-row
for k in LABELS:
    out[f"ind_{k}"] = raw[k]
    out[f"indflag_{k}"] = flag[k]

# supply + population + centroid per district
out["n_fac"] = sup["n_fac"].reindex(nfhs.index).fillna(0).astype(int)
out["n_hosp"] = sup["n_hosp"].reindex(nfhs.index).fillna(0).astype(int)
out["n_obgyn"] = sup["n_obgyn"].reindex(nfhs.index).fillna(0).astype(int)
popvals = [lk_pop(s, d) for s, d in zip(nfhs["s"], nfhs["d"])]
out["pop"] = [p for p, _ in popvals]
out["pop_match"] = [m for _, m in popvals]

lat = fac_cent["lat"].reindex(nfhs.index)
lon = fac_cent["lon"].reindex(nfhs.index)
for i in nfhs.index:
    if pd.isna(lat[i]):
        key = (nfhs.loc[i, "s"], nfhs.loc[i, "d"])
        if key in pin_cent.index:
            lat[i] = pin_cent.loc[key, "lat"]; lon[i] = pin_cent.loc[key, "lon"]
out["lat"] = lat.values
out["lon"] = lon.values

# confidence: state-level orphan rate
placed_state = fac.dropna(subset=["geo_state"]).groupby("geo_state").size()
matched_state = out.assign(s=nfhs["s"]).groupby("s")["n_fac"].sum()
orate = (1 - (matched_state / placed_state)).clip(lower=0).to_dict()
def confidence(row, s):
    if row["n_fac"] > 0:
        return "measured"
    return "real_sparse" if orate.get(s, 0) < 0.15 else "low_join_conf"
out["supply_confidence"] = [confidence(out.loc[i], nfhs.loc[i, "s"]) for i in nfhs.index]

# per-capita supply + gap (only where population resolved and enough indicators)
out["fac_per_100k"] = out["n_fac"] / (out["pop"] / 1e5)
out["obgyn_per_100k"] = out["n_obgyn"] / (out["pop"] / 1e5)
pk = out["pop"].notna() & (out["need_valid"] >= 10)
out["supply_pc_z"] = np.nan
out.loc[pk, "supply_pc_z"] = z(np.log1p(out.loc[pk, "fac_per_100k"]))
out["obgyn_pc_z"] = np.nan
out.loc[pk, "obgyn_pc_z"] = z(np.log1p(out.loc[pk, "obgyn_per_100k"]))
out["gap"] = out["need_z"] - out["supply_pc_z"]
out["maternal_gap"] = out["maternal_deficit_z"] - out["obgyn_pc_z"]
out["pop_resolved"] = pk

# overall priority rank: pop-known + confident first, by gap
out["rank_basis"] = np.where(out["pop_resolved"] & (out["supply_confidence"] != "low_join_conf"),
                             out["gap"], np.nan)
out = out.sort_values(["rank_basis", "need_z"], ascending=False, na_position="last").reset_index(drop=True)
out["priority_rank"] = np.arange(1, len(out) + 1)

out.to_parquet(f"{OUT}/district_gaps.parquet", index=False)
print(f"[write] district_gaps.parquet  rows={len(out)}  pop-resolved={int(pk.sum())}")

# ----------------------------------------------------------------------------- 6. facility citations
cit_cols = ["unique_id", "name", "facilityTypeId", "operatorTypeId", "address_city",
            "address_stateOrRegion", "specialties", "description", "capability",
            "procedure", "equipment", "yearEstablished", "capacity", "numberDoctors",
            "officialWebsite", "source_urls", "is_obgyn", "is_hospital", "la", "lo"]
cit = fac.dropna(subset=["nfhs_idx"]).copy()
cit["nfhs_idx"] = cit["nfhs_idx"].astype(int)
cit["district_name"] = cit["nfhs_idx"].map(nfhs["district_name"])
cit["state_ut"] = cit["nfhs_idx"].map(nfhs["state_ut"])
cit = cit[["district_name", "state_ut"] + cit_cols].rename(columns={"la": "lat", "lo": "lon"})
cit.to_parquet(f"{OUT}/facility_citations.parquet", index=False)
print(f"[write] facility_citations.parquet  rows={len(cit)}")
print("DONE")
