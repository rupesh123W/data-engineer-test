import os
from datetime import datetime

# ----------------------------
# Project Root
# ----------------------------
PROJECT_ROOT = r"C:\Users\Rupesh.shelar\data-engineer-test"

# ----------------------------
# Input Folders
# ----------------------------
OLYMPICS_DIR = os.path.join(PROJECT_ROOT, "datasets", "olympics")
MAPPING_DIR = os.path.join(PROJECT_ROOT, "datasets", "country_code_mapping")
COUNTRIES_DIR = os.path.join(PROJECT_ROOT, "datasets", "countries")

# ----------------------------
# Date Partition (Dynamic)
# ----------------------------
TODAY = datetime.today().strftime("%Y%m%d")   # e.g., 20250916

# ----------------------------
# Bronze Layer Output
# ----------------------------
BRONZE_DIR = os.path.join(PROJECT_ROOT, "datasets", "solution", "output", TODAY, "bronze")

# ----------------------------
# Silver Layer Output
# ----------------------------
SILVER_DIR = os.path.join(PROJECT_ROOT, "datasets", "solution", "output", TODAY, "silver")

# ----------------------------
# Utility: pick first CSV in folder
# ----------------------------
def get_first_csv(folder: str) -> str:
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return os.path.join(folder, csv_files[0])

# ----------------------------
# Input CSVs (Mapping + Countries)
# ----------------------------
MAPPING_PATH = get_first_csv(MAPPING_DIR)
COUNTRIES_PATH = get_first_csv(COUNTRIES_DIR)
