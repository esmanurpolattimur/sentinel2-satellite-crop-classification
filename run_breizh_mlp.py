import subprocess
import sys
from pathlib import Path


# Bu kısa dosya, uzun MLP komutunu tek satırla çalıştırır.
proje_klasoru = Path(__file__).resolve().parent
mlp_kodu = proje_klasoru / "analysis" / "phase5_breizh_mlp.py"


def main():
    komut = [sys.executable, str(mlp_kodu)] + sys.argv[1:]
    subprocess.run(komut, cwd=proje_klasoru, check=True)


if __name__ == "__main__":
    main()
