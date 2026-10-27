"""
download_data.py
================
AILA dataset Kaggle se download karta hai.
Setup: kaggle.json ~/.kaggle/ mein rakhna padega
Run:   python download_data.py
"""

import os
import shutil
from pathlib import Path

import kagglehub

print("=" * 50)
print("  INDIAN LEGAL AI — Dataset Download (AILA)")
print("=" * 50)

os.makedirs("data/raw", exist_ok=True)

print("\n[1/2] Kaggle se AILA dataset download ho raha hai...")
print("      (pehli baar ~200MB download hoga)\n")

try:
    path = kagglehub.dataset_download("ananyapam7/legalai")
    print(f"\n[2/2] Downloaded to cache: {path}")

    # Files apne project mein copy karo
    src = Path(path)
    dst = Path("data/raw")

    for f in src.rglob("*"):
        if f.is_file():
            rel = f.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            print(f"      Copied: {rel}")

    print("\n Done! Files saved to data/raw/")
    print("      Ab run karo: python preprocess.py")

except Exception as e:
    print(f"\n Error: {e}")
    print("\n Kaggle API setup karo:")
    print("   1. kaggle.com/settings → API → Generate New Token")
    print("   2. kaggle.json ko C:/Users/<aapka_naam>/.kaggle/ mein rakho")
    print("   3. pip install kagglehub")
