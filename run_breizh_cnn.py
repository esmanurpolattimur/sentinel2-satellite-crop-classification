import subprocess
import sys
from pathlib import Path


# Bu kısa dosya, uzun 1D-CNN komutunu tek satırla çalıştırır.
proje_klasoru = Path(__file__).resolve().parent
cnn_kodu = proje_klasoru / "analysis" / "phase7_breizh_cnn.py"


def main():
    komut = [sys.executable, str(cnn_kodu)] + sys.argv[1:]
    subprocess.run(komut, cwd=proje_klasoru, check=True)


if __name__ == "__main__":
    main()
