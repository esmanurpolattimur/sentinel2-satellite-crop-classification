# Yolları elle uzun metinler halinde yazmak yerine parçaları birleştirir.
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "phase1"
BREIZHCROPS_DIR = DATA_DIR / "breizhcrops"
TUM_CROPTYPES_DIR = DATA_DIR / "tum_croptypes"

BREIZHCROPS_FILES = [
    "classmapping.csv",
    "frh01.csv",
    "frh02.csv",
    "frh03.csv",
    "frh04.csv",
]

TUM_CROPTYPES_FILES = [
    "data2016-2018.xlsx",
    "TestData.xlsx",
]


def check_required_files(directory, file_names):
    all_files_exist = True

    for file_name in file_names:
        file_path = directory / file_name

        if file_path.is_file():
            print(f"[OK] {file_name}")
        else:
            print(f"[EKSİK] {file_name}")
            all_files_exist = False

    return all_files_exist


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"BreizhCrops data directory: {BREIZHCROPS_DIR}")
    print(f"TUM CropTypes data directory: {TUM_CROPTYPES_DIR}")
    print(f"Data directory exists: {DATA_DIR.exists()}")
    print(f"BreizhCrops directory exists: {BREIZHCROPS_DIR.exists()}")
    print(f"TUM CropTypes directory exists: {TUM_CROPTYPES_DIR.exists()}")

    print("\nBreizhCrops dosya kontrolü:")
    breizhcrops_ready = check_required_files(
        BREIZHCROPS_DIR,
        BREIZHCROPS_FILES,
    )

    print("\nTUM CropTypes dosya kontrolü:")
    tum_croptypes_ready = check_required_files(
        TUM_CROPTYPES_DIR,
        TUM_CROPTYPES_FILES,
    )

    if breizhcrops_ready and tum_croptypes_ready:
        print("\nTüm Faz 1 veri dosyaları hazır.")
    else:
        print("\nEksik Faz 1 veri dosyaları var.")


if __name__ == "__main__":
    main()
