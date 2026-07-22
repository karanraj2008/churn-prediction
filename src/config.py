"""
config.py
Single shared setting used across the pipeline. Bump DATA_VERSION here when
new raw data arrives — every script picks it up automatically, so there's
no risk of one script using v1 data while another accidentally uses v2.
"""

DATA_VERSION = "v1"


# Maps each data version to its raw filename. When new data arrives,
# add a new entry here (e.g. "v2": "...") rather than overwriting v1 —
# that way old raw files stay available for reference/rollback.
RAW_DATA_FILENAMES = {
    "v1": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
}