"""
download_data.py
================
Email Dataset Acquisition and Extraction Script for Email Spam Detection System.

Target Dataset:
---------------
SpamAssassin Public Corpus (Apache Software Foundation)
- Easy Ham: 20030228_easy_ham.tar.bz2 (~1.6 MB compressed, ~2,551 legitimate emails)
- Spam:     20030228_spam.tar.bz2     (~1.2 MB compressed, ~501 unsolicited spam emails)

URL Sources:
- Primary: https://spamassassin.apache.org/old/publiccorpus/
- Fallback mirror instructions provided if network or certificate errors occur.

Workflow:
1. Downloads .tar.bz2 archives for Ham and Spam into data/raw_archives/.
2. Extracts individual raw RFC 822 email files into data/easy_ham/ and data/spam/.
3. Verifies extracted message counts and file integrity.
"""

import os
import sys
import tarfile
import urllib.request
from typing import Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARCHIVE_DIR = os.path.join(DATA_DIR, "raw_archives")
EASY_HAM_DIR = os.path.join(DATA_DIR, "easy_ham")
SPAM_DIR = os.path.join(DATA_DIR, "spam")

CORPORA: Dict[str, Dict[str, str]] = {
    "easy_ham": {
        "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
        "archive_name": "20030228_easy_ham.tar.bz2",
        "target_dir": EASY_HAM_DIR,
        "label": "ham",
    },
    "spam": {
        "url": "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2",
        "archive_name": "20030228_spam.tar.bz2",
        "target_dir": SPAM_DIR,
        "label": "spam",
    },
}


def download_file(url: str, dest_path: str) -> bool:
    """Download archive with user-agent and progress reporting."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100000:
        print(f"[INFO] Archive already exists: {dest_path} ({os.path.getsize(dest_path):,} bytes)")
        return True

    print(f"[INFO] Downloading from: {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SpamAssassinCorpusDownloader/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out_f:
            total_size = resp.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size else None
            downloaded = 0
            chunk_size = 64 * 1024

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    percent = (downloaded / total_bytes) * 100
                    print(f"\r -> Progress: {downloaded:,} / {total_bytes:,} bytes ({percent:.1f}%)", end="", flush=True)
                else:
                    print(f"\r -> Downloaded: {downloaded:,} bytes", end="", flush=True)
            print()
        print(f"[SUCCESS] Download completed: {dest_path}")
        return True
    except Exception as exc:
        print(f"\n[ERROR] Download failed for {url}: {exc}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def extract_tar_archive(archive_path: str, target_dir: str) -> bool:
    """Safely unpack .tar.bz2 archive into target directory."""
    os.makedirs(target_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
    if len(existing_files) > 100:
        print(f"[INFO] Target directory '{target_dir}' already contains {len(existing_files)} files. Skipping extraction.")
        return True

    print(f"[INFO] Extracting archive '{archive_path}'...")
    try:
        with tarfile.open(archive_path, "r:bz2") as tar:
            # Extract all members
            members = tar.getmembers()
            print(f" -> Found {len(members)} entries in archive. Extracting...")
            tar.extractall(path=DATA_DIR)

        # Confirm target directory has extracted files
        if os.path.exists(target_dir):
            count = len(os.listdir(target_dir))
            print(f"[SUCCESS] Extracted to '{target_dir}' ({count} items)")
            return True
        else:
            print(f"[WARNING] Expected directory '{target_dir}' not found after extraction.")
            return False
    except Exception as exc:
        print(f"[ERROR] Failed to extract '{archive_path}': {exc}")
        return False


def print_manual_instructions():
    """Print clear instructions if automatic downloading fails."""
    print("\n" + "=" * 70)
    print("           MANUAL EMAIL DATASET DOWNLOAD INSTRUCTIONS            ")
    print("=" * 70)
    print("If your network blocks direct downloads, please follow these steps:")
    print("1. Open your browser and navigate to:")
    print("   https://spamassassin.apache.org/old/publiccorpus/")
    print("2. Download the following two archives:")
    print("   - 20030228_easy_ham.tar.bz2 (~1.6 MB)")
    print("   - 20030228_spam.tar.bz2     (~1.2 MB)")
    print("3. Place both archives into the folder:")
    print(f"   {ARCHIVE_DIR}")
    print("4. Extract 20030228_easy_ham.tar.bz2 so the 'easy_ham' directory is at:")
    print(f"   {EASY_HAM_DIR}")
    print("5. Extract 20030228_spam.tar.bz2 so the 'spam' directory is at:")
    print(f"   {SPAM_DIR}")
    print("6. Run 'python train_model.py' once extracted.")
    print("=" * 70 + "\n")


def prepare_dataset() -> bool:
    """Download and extract both Ham and Spam email corpora."""
    print("=" * 70)
    print("SpamAssassin Email Corpus Dataset Preparation")
    print("=" * 70)

    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    all_success = True

    for corpus_key, info in CORPORA.items():
        archive_path = os.path.join(ARCHIVE_DIR, info["archive_name"])
        target_dir = info["target_dir"]

        # Download if needed
        downloaded = download_file(info["url"], archive_path)
        if not downloaded:
            all_success = False
            continue

        # Extract
        extracted = extract_tar_archive(archive_path, target_dir)
        if not extracted:
            all_success = False

    if all_success:
        ham_count = len(os.listdir(EASY_HAM_DIR)) if os.path.exists(EASY_HAM_DIR) else 0
        spam_count = len(os.listdir(SPAM_DIR)) if os.path.exists(SPAM_DIR) else 0
        print("\n" + "=" * 70)
        print("[SUCCESS] Email dataset is ready for training!")
        print(f" -> Ham (Legitimate) email files: {ham_count}")
        print(f" -> Spam (Unsolicited) email files: {spam_count}")
        print(f" -> Total raw files: {ham_count + spam_count}")
        print("=" * 70 + "\n")
        return True
    else:
        print_manual_instructions()
        return False


if __name__ == "__main__":
    success = prepare_dataset()
    if not success:
        sys.exit(1)
