"""
paths_v4.py -- single source of truth for v4 data locations.

Design (agreed):
  - data/      (DATA_RAW)  : read-only RAW inputs + EXISTING spectra (unchanged).
  - data_v4/   (DATA_OUT)  : everything the v4 pipeline BUILDS + NEW spectra downloads.
  - spectra are referenced in place: lookups search data_v4/ then data/.

Import this in the build scripts. The trainer/eval read config_v2.yml (whose built
paths point at data_v4/) and resolve spectra via datasets_v2._resolve, which uses the
same two-root search.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data")        # inputs + existing spectra
DATA_OUT = os.path.join(BASE_DIR, "data_v4")     # built outputs + new spectra
os.makedirs(DATA_OUT, exist_ok=True)

# spectra resolution: prefer data_v4, fall back to data
SPECTRA_ROOTS = [DATA_OUT, DATA_RAW]


def spec_exists(relpath: str) -> bool:
    return any(os.path.exists(os.path.join(r, relpath)) for r in SPECTRA_ROOTS)


def spec_path(relpath: str) -> str:
    for r in SPECTRA_ROOTS:
        p = os.path.join(r, relpath)
        if os.path.exists(p):
            return p
    return os.path.join(DATA_OUT, relpath)   # default: new download target


# ---- RAW inputs (stay in data/) ------------------------------------
CROSSMATCH    = os.path.join(DATA_RAW, "dr16-sdssv_crossmatch.pkl")
LOWZ          = os.path.join(DATA_RAW, "dr16_sdssv_crossmatch_lowz.pkl")
PAPER2_MASTER = os.path.join(DATA_RAW, "paper2_master.pkl")
PAPER2_TRAIN  = os.path.join(DATA_RAW, "paper2_train_pairs.pkl")
PAPER2_TEST   = os.path.join(DATA_RAW, "paper2_test_pairs.pkl")
CLAGN_LIST    = os.path.join(DATA_RAW, "cl_agn_list_dr16.pkl")
DESI_TYPE_CSV = os.path.join(DATA_RAW, "full_data", "desi_type1_type2.csv")
SSL_BASE_FILTERED = os.path.join(DATA_RAW, "ssl_unified_dr7_desi_filtered.parquet")

# spAll catalogs (Thesis dir)
SPALL_DR16  = "/Users/amir/Documents/Msc/Thesis/data/spAll-v5_13_0.fits"
SPALL_SDSSV = "/Users/amir/Documents/Msc/Thesis/data/spAll-lite-v6_2_1-epoch.fits"

# ---- BUILT outputs (data_v4/) --------------------------------------
DR16_OBS      = os.path.join(DATA_OUT, "dr16_spall_obs.pkl")
SDSSV_OBS     = os.path.join(DATA_OUT, "sdssv_spall_obs.pkl")
SDSSV_NEG     = os.path.join(DATA_OUT, "sdssv_dr16_negatives.pkl")
DESI_NEG      = os.path.join(DATA_OUT, "desi_dr16_negatives.pkl")
PHASE2_TRAIN  = os.path.join(DATA_OUT, "dr16_sdssv_phase2_train.pkl")
CLAGN_TEST    = os.path.join(DATA_OUT, "clagn_test.pkl")
SSL_TYPE2     = os.path.join(DATA_OUT, "ssl_type2.parquet")
SSL_DR7CAPPED = os.path.join(DATA_OUT, "ssl_unified_dr7capped_desi.parquet")
SSL_EXTENSION = os.path.join(DATA_OUT, "ssl_dr16_sdssv_extension.parquet")

# NEW spectra download dirs (data_v4/)
DR16_SPECTRA_SUBDIR  = "dr16_spectra"            # DR16/BOSS early epochs
SDSSV_T2_SPECTRA_SUBDIR = "sdssv_type2_spectra"  # SDSS-V type-2 SSL
SDSSV_PAIR_SUBDIR = "dr16_sdssv_crossmatch"      # SDSS-V negative late epochs
DESI_SPEC_SUBDIR = os.path.join("desi", "spectra")
