"""
download_data.py
================
Stage 1: Environment and Dataset Acquisition Script

Purpose:
--------
Programmatically downloads and extracts the official UCI SMS Spam Collection
dataset from the UCI Machine Learning Repository. If downloading encounters
network or SSL issues, clear manual fallback instructions are displayed.

Target Dataset:
---------------
UCI SMS Spam Collection (ID: 228)
URL: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
Direct Archive: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
"""

import io
import os
import sys
import urllib.request
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE):
        print(f"[INFO] Dataset file already exists at: {DATA_FILE}")
        file_size = os.path.getsize(DATA_FILE)
        print(f"[INFO] File size: {file_size:,} bytes")
        return True

    print(f"[INFO] Downloading UCI SMS Spam Collection from:\n       {ZIP_URL}")
    req = urllib.request.Request(
        ZIP_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SpamDetectionProject/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            zip_bytes = response.read()
            print(f"[INFO] Downloaded {len(zip_bytes):,} bytes. Extracting archive...")

            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                z.extractall(DATA_DIR)

        if os.path.exists(DATA_FILE):
            print(f"[SUCCESS] Dataset successfully extracted to:\n          {DATA_FILE}")
            print(f"[INFO] File size: {os.path.getsize(DATA_FILE):,} bytes")
            return True
        else:
            print(f"[WARNING] Extracted, but '{DATA_FILE}' was not found in archive contents.")
            print(f"[INFO] Files extracted to data/: {os.listdir(DATA_DIR)}")
            return False

    except Exception as exc:
        print("\n" + "=" * 60)
        print("[ERROR] Automatic download failed:")
        print(f"        {exc}")
        print("=" * 60)
        print("\n--- MANUAL DOWNLOAD INSTRUCTIONS ---")
        print("1. Open your browser and go to:")
        print("   https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
        print("   or directly:")
        print("   https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip")
        print("2. Download the 'sms+spam+collection.zip' archive.")
        print("3. Extract the file named 'SMSSpamCollection' (no extension).")
        print("4. Place it into the project folder under:")
        print(f"   {DATA_FILE}")
        print("=" * 60 + "\n")
        return False


if __name__ == "__main__":
    success = download_and_extract()
    if not success:
        sys.exit(1)
