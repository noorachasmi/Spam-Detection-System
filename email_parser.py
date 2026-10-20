"""
email_parser.py
===============
Email parsing and preprocessing utilities for the Email Spam Detection System.

Handles:
- Raw RFC 822 / MIME encoded emails (.eml files or text)
- Extraction of Subject and Body
- Multipart emails (traversing text/plain, text/html)
- Character encoding detection and safe decoding (ASCII, UTF-8, Latin-1, Windows-1252)
- HTML tag stripping and entity unescaping via BeautifulSoup
- Heuristic indicator extraction (suspicious keywords, links, urgency, exclamation marks)
"""

import email
from email import policy
from email.header import decode_header
import re
from typing import Dict, Tuple
from bs4 import BeautifulSoup

# Suspicious words and phrases commonly observed in unsolicited/phishing emails
SUSPICIOUS_PHRASES = [
    "urgent action required",
    "account suspended",
    "verify your account",
    "verify your identity",
    "password reset",
    "security alert",
    "unauthorized access",
    "winner",
    "you have won",
    "claim your prize",
    "wire transfer",
    "lottery",
    "million dollars",
    "risk-free",
    "limited time offer",
    "act now",
    "click here immediately",
    "credit card details",
    "confirm payment",
    "western union",
    "cryptocurrency investment",
    "free gift card",
    "inheritance",
    "guaranteed return",
]


def decode_mime_header(header_value: str) -> str:
    """Decode RFC 2047 MIME-encoded headers (e.g. =?UTF-8?B?...?=)."""
    if not header_value:
        return ""
    try:
        decoded_fragments = decode_header(header_value)
        text_parts = []
        for fragment, encoding in decoded_fragments:
            if isinstance(fragment, bytes):
                encoding = encoding or "utf-8"
                try:
                    text_parts.append(fragment.decode(encoding, errors="replace"))
                except (LookupError, UnicodeDecodeError):
                    text_parts.append(fragment.decode("latin-1", errors="replace"))
            else:
                text_parts.append(str(fragment))
        return " ".join(text_parts).strip()
    except Exception:
        return str(header_value).strip()


def clean_html_content(raw_html: str) -> str:
    """Extract readable text from HTML markup, removing script and style elements."""
    if not raw_html:
        return ""
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style", "meta", "noscript", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception:
        # Fallback regex strip if BeautifulSoup encounters parser errors
        clean = re.sub(r"<[^>]+>", " ", raw_html)
        return re.sub(r"\s+", " ", clean).strip()


def extract_payload_from_message(msg: email.message.EmailMessage) -> str:
    """
    Extract plaintext content from an EmailMessage.
    Traverses multipart trees and handles text/plain and text/html bodies.
    """
    body_parts = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            # Skip binary attachments
            if "attachment" in content_disposition:
                continue

            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue

                charset = part.get_content_charset() or "utf-8"
                try:
                    text = payload.decode(charset, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    text = payload.decode("latin-1", errors="replace")

                if content_type == "text/html":
                    text = clean_html_content(text)
                elif content_type == "text/plain":
                    text = text.strip()

                if text:
                    body_parts.append(text)
            except Exception:
                continue
    else:
        content_type = msg.get_content_type()
        try:
            payload = msg.get_payload(decode=True)
            if payload is not None:
                charset = msg.get_content_charset() or "utf-8"
                try:
                    text = payload.decode(charset, errors="replace")
                except (LookupError, UnicodeDecodeError):
                    text = payload.decode("latin-1", errors="replace")
            else:
                text = str(msg.get_payload() or "")

            if content_type == "text/html":
                text = clean_html_content(text)
            body_parts.append(text.strip())
        except Exception:
            body_parts.append(str(msg.get_payload() or "").strip())

    return "\n\n".join(part for part in body_parts if part).strip()


def parse_raw_email(raw_bytes_or_str) -> Tuple[str, str]:
    """
    Parse a raw .eml file or raw email string.
    Returns (subject, body).
    """
    if isinstance(raw_bytes_or_str, str):
        raw_bytes = raw_bytes_or_str.encode("utf-8", errors="replace")
    else:
        raw_bytes = raw_bytes_or_str

    try:
        msg = email.message_from_bytes(raw_bytes, policy=policy.default)
        raw_subject = msg.get("Subject", "") or ""
        subject = decode_mime_header(raw_subject)
        body = extract_payload_from_message(msg)
        return subject, body
    except Exception as exc:
        raise ValueError(f"Failed to parse email message: {exc}") from exc


def combine_subject_and_body(subject: str, body: str) -> str:
    """
    Concatenate subject and body into a single structured string for classification.
    Gives appropriate natural prominence to subject lines.
    """
    subject_clean = (subject or "").strip()
    body_clean = (body or "").strip()

    if subject_clean and body_clean:
        return f"Subject: {subject_clean}\n\n{body_clean}"
    if subject_clean:
        return f"Subject: {subject_clean}"
    return body_clean


def analyze_heuristics(subject: str, body: str) -> Dict:
    """
    Analyze heuristic indicators in email text (educational signals, NOT definitive proof).
    Identifies:
    - Suspicious keywords and trigger phrases
    - URL/link counts
    - Excessive capitalization
    - Exclamation mark counts
    - Currency / financial symbols
    """
    full_text = f"{subject}\n{body}".strip()
    text_lower = full_text.lower()

    # Found suspicious phrases
    matched_phrases = [phrase for phrase in SUSPICIOUS_PHRASES if phrase in text_lower]

    # Link / URL detection (http://, https://, www., .com/.net/.org/.xyz urls)
    url_pattern = re.compile(
        r"https?://[^\s<>\"']+|www\.[^\s<>\"']+|[a-zA-Z0-9.\-_]+\.(?:com|org|net|xyz|ru|biz|top|cc)[^\s<>\"']*",
        re.IGNORECASE,
    )
    urls_found = url_pattern.findall(full_text)

    # Exclamation mark count
    exclamation_count = full_text.count("!")

    # Question mark count
    question_count = full_text.count("?")

    # Currency symbols ($, £, €, ₹)
    currency_count = len(re.findall(r"[\$£€₹]|USD|EUR|GBP", full_text))

    # All-caps words (3+ uppercase letters)
    caps_words = re.findall(r"\b[A-Z]{3,}\b", full_text)
    # Filter out common abbreviations like MIME, HTML, USA, ID
    common_acronyms = {"HTML", "MIME", "URL", "HTTP", "HTTPS", "PDF", "USA", "CSS", "API", "FAQ", "EST", "PST"}
    suspicious_caps = [w for w in caps_words if w not in common_acronyms]

    return {
        "matched_phrases": matched_phrases,
        "url_count": len(urls_found),
        "sample_urls": urls_found[:3],
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "currency_count": currency_count,
        "all_caps_words": suspicious_caps[:6],
        "all_caps_count": len(suspicious_caps),
    }
