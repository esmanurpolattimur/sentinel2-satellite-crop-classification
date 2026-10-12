import subprocess
import sys
from pathlib import Path


# Bu kısa dosya, uzun XGBoost komutunu tek satırla çalıştırır.
proje_klasoru = Path(__file__).resolve().parent
xgboost_kodu = proje_klasoru / "analysis" / "phase6_breizh_xgboost.py"


def main():
    komut = [sys.executable, str(xgboost_kodu)] + sys.argv[1:]
    subprocess.run(komut, cwd=proje_klasoru, check=True)


if __name__ == "__main__":
    main()
