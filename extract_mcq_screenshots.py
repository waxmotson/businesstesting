#!/usr/bin/env python3
"""
Extract screenshots of each multiple-choice question (Questions 1-20)
from HSC Business Studies past papers.

Usage:
    python extract_mcq_screenshots.py [--year YEAR] [--all] [--out-dir DIR]

Requires: pymupdf (fitz), Pillow
"""

import argparse
import os
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import fitz  # pymupdf
from PIL import Image
import numpy as np

# Rendering scale (2.0 => ~144 dpi, matches typical screenshot width of 1190px for A4)
SCALE = 2.0

# PDF coordinate padding (points)
TOP_PAD_PTS = 12
BETWEEN_PAD_PTS = 6
FOOTER_Y = 775  # stop before page number footers (~794)

# Pixel padding after whitespace trim
FINAL_PAD_PX = 22


def find_mcq_positions(doc):
    """Locate the top of each MCQ (1-20) by finding left-aligned question numbers."""
    questions = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b.get("type") != 0:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text.isdigit():
                        continue
                    n = int(text)
                    if 1 <= n <= 20 and span["bbox"][0] < 90 and 9 < span["size"] < 14:
                        questions.append({
                            "num": n,
                            "page": page_idx,
                            "y": span["bbox"][1],
                        })
    # Sort and deduplicate (keep first occurrence of each number)
    questions = sorted(questions, key=lambda q: (q["page"], q["y"]))
    seen = set()
    unique = []
    for q in questions:
        if q["num"] not in seen:
            seen.add(q["num"])
            unique.append(q)
    return unique


def get_content_bottom(page, start_y, footer_y):
    """Find the lowest y of content (text or drawings) after start_y, before footer."""
    max_y = start_y
    # Text
    blocks = page.get_text("dict")["blocks"]
    for b in blocks:
        if b.get("type") != 0:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                y1 = span["bbox"][3]
                if start_y - 2 < span["bbox"][1] < footer_y and y1 < footer_y:
                    max_y = max(max_y, y1)
    # Vector drawings (diagrams, lines, tables)
    for d in page.get_drawings():
        r = d["rect"]
        if r.y0 > start_y - 5 and r.y1 < footer_y:
            max_y = max(max_y, r.y1)
    return max_y


def trim_vertical_whitespace(img: Image.Image, pad: int = FINAL_PAD_PX, bg_thresh: int = 250) -> Image.Image:
    """Remove excess white space top/bottom while keeping left/right margins and adding pad."""
    arr = np.array(img)
    # Consider a pixel non-white if any channel is below threshold
    mask = np.any(arr < bg_thresh, axis=2)
    rows_with_content = np.any(mask, axis=1)
    if not np.any(rows_with_content):
        return img
    rmin, rmax = np.where(rows_with_content)[0][[0, -1]]
    rmin = max(0, rmin - pad)
    rmax = min(arr.shape[0] - 1, rmax + pad)
    return img.crop((0, rmin, arr.shape[1], rmax + 1))


def extract_year(pdf_path: Path, out_dir: Path, year: int, dry_run: bool = False):
    """Extract all 20 MCQ screenshots for one paper."""
    doc = fitz.open(pdf_path)
    qs = find_mcq_positions(doc)

    if len(qs) != 20:
        print(f"  WARNING: expected 20 questions, found {len(qs)}")
        found_nums = {q["num"] for q in qs}
        missing = sorted(set(range(1, 21)) - found_nums)
        if missing:
            print(f"  Missing: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, q in enumerate(qs):
        page = doc[q["page"]]
        page_height = page.rect.height

        top = max(0.0, q["y"] - TOP_PAD_PTS)

        if i + 1 < len(qs) and qs[i + 1]["page"] == q["page"]:
            # Next question on same page
            bottom = qs[i + 1]["y"] - BETWEEN_PAD_PTS
        else:
            # Last question on this page (or last overall)
            content_bottom = get_content_bottom(page, q["y"], FOOTER_Y)
            bottom = min(FOOTER_Y, content_bottom + 12)

        if bottom <= top + 30:
            bottom = top + 80  # safety minimum height

        # Render page region
        mat = fitz.Matrix(SCALE, SCALE)
        # Full page render then crop (simple & reliable)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        top_px = int(round(top * SCALE))
        bot_px = int(round(bottom * SCALE))
        # Clamp
        top_px = max(0, top_px)
        bot_px = min(pix.height, bot_px)
        crop = img.crop((0, top_px, pix.width, bot_px))

        # Tight vertical crop with consistent padding
        crop = trim_vertical_whitespace(crop, pad=FINAL_PAD_PX)

        out_name = f"HSC_BUSINESS_{year}_Q{q['num']:02d}.png"
        out_path = out_dir / out_name

        if not dry_run:
            crop.save(out_path, "PNG", optimize=True)
        results.append((out_name, crop.size))
        print(f"  Q{q['num']:02d} -> {out_name}  ({crop.size[0]}x{crop.size[1]})")

    doc.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract HSC Business Studies MCQ screenshots")
    parser.add_argument("--year", type=int, help="Single year to process (e.g. 2018)")
    parser.add_argument("--all", action="store_true", help="Process all available years")
    parser.add_argument("--out-dir", type=str, default="mcq_screenshots",
                        help="Output directory (default: mcq_screenshots)")
    parser.add_argument("--pdf-dir", type=str, default=".",
                        help="Directory containing the PDF files")
    parser.add_argument("--dry-run", action="store_true", help="Detect only, do not write images")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)

    if args.year:
        years = [args.year]
    elif args.all:
        years = list(range(2016, 2026))
    else:
        # default: all
        years = list(range(2016, 2026))

    for year in years:
        pdf_path = pdf_dir / f"{year}-hsc-business-studies.pdf"
        if not pdf_path.exists():
            print(f"Skipping {year}: PDF not found at {pdf_path}")
            continue
        print(f"\n=== {year} HSC Business Studies ===")
        extract_year(pdf_path, out_dir, year, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
