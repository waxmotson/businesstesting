#!/usr/bin/env python3
"""
Extract screenshots of each multiple-choice question (Questions 1–20)
from NSW HSC Business Studies past papers.

Handles shared stimulus material, e.g.:
  "Use the following information to answer Questions 17 and 18."
  "Refer to the following diagram to answer Questions 17−18."

For a shared stimulus the image for each question contains:
  [stimulus] + [that question only]
(the other questions that share the stimulus are cut out)

Expected PDF layout (relative to where you run the script):

    ./HSC_PAPERS/BusinessStudies/
        2016/2016-hsc-business-studies.pdf
        2017/2017-hsc-business-studies.pdf
        ...
        2025/2025-hsc-business-studies.pdf

Usage:
    python extract_hsc_business_mcqs.py              # all years found
    python extract_hsc_business_mcqs.py --year 2024  # single year
    python extract_hsc_business_mcqs.py --out-dir screenshots

Requirements:
    pip install pymupdf Pillow numpy
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import fitz  # pymupdf
except ImportError:
    print("Missing dependency: pymupdf")
    print("Install with:  pip install pymupdf")
    sys.exit(1)

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Missing dependency: Pillow / numpy")
    print("Install with:  pip install Pillow numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SCALE = 2.0                 # ~144 dpi → 1190 px wide for A4
TOP_PAD_PTS = 12
BETWEEN_PAD_PTS = 6
FOOTER_Y = 775              # avoid page-number footers
FINAL_PAD_PX = 22           # padding after whitespace trim
STITCH_GAP_PX = 12          # small white gap when joining stimulus + question

DEFAULT_PDF_ROOT = Path("HSC_PAPERS") / "BusinessStudies"
DEFAULT_OUT_DIR = Path("mcq_screenshots")
YEARS = list(range(2016, 2026))

# Matches phrases like:
#   "Use the following information to answer Questions 17 and 18."
#   "Refer to the following diagram to answer Questions 17−18."
#   "Use the information to answer Question 15."
STIMULUS_RE = re.compile(
    r"(?:use|refer\s+to).{0,120}?(?:question|questions)\s+(\d+)\s*"
    r"(?:[-–−—]|and|,)?\s*(\d+)?",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def find_mcq_positions(doc: fitz.Document) -> list[dict]:
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
    questions = sorted(questions, key=lambda q: (q["page"], q["y"]))
    seen = set()
    unique = []
    for q in questions:
        if q["num"] not in seen:
            seen.add(q["num"])
            unique.append(q)
    return unique


def find_stimuli(doc: fitz.Document, questions: list[dict]) -> tuple[dict[int, dict], dict[int, list[float]]]:
    """
    Detect shared (or single) stimulus blocks.

    Returns:
        stimuli_for_q : { qnum: info_dict }
        stimulus_y_by_page : { page_idx: [y_intro, ...] }  sorted
    """
    q_by_num = {q["num"]: q for q in questions}
    stimuli_for_q: dict[int, dict] = {}
    stimulus_y_by_page: dict[int, list[float]] = {}

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b.get("type") != 0:
                continue
            for line in b["lines"]:
                full = " ".join(s["text"] for s in line["spans"]).strip()
                m = STIMULUS_RE.search(full)
                if not m:
                    continue
                q1 = int(m.group(1))
                q2 = int(m.group(2)) if m.group(2) else q1
                # Ignore the big section header "Questions 1–20"
                if q1 == 1 and q2 >= 15:
                    continue
                if not (1 <= q1 <= 20):
                    continue
                if q2 < q1:
                    q1, q2 = q2, q1

                y_intro = line["spans"][0]["bbox"][1]
                stimulus_y_by_page.setdefault(page_idx, []).append(y_intro)

                first_q = q_by_num.get(q1)
                if first_q is None:
                    continue
                first_q_y = first_q["y"] if first_q["page"] == page_idx else 0

                info = {
                    "page": page_idx,
                    "y_intro": y_intro,
                    "q_first": q1,
                    "q_last": q2,
                    "first_q_y": first_q_y,
                    "first_q_page": first_q["page"],
                }
                for qn in range(q1, q2 + 1):
                    if qn not in stimuli_for_q:
                        stimuli_for_q[qn] = info

    # sort the y lists
    for p in stimulus_y_by_page:
        stimulus_y_by_page[p] = sorted(set(stimulus_y_by_page[p]))

    return stimuli_for_q, stimulus_y_by_page


def get_content_bottom(page: fitz.Page, start_y: float, footer_y: float) -> float:
    """Lowest y of content (text or drawings) after start_y, before footer."""
    max_y = start_y
    blocks = page.get_text("dict")["blocks"]
    for b in blocks:
        if b.get("type") != 0:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                y1 = span["bbox"][3]
                if start_y - 2 < span["bbox"][1] < footer_y and y1 < footer_y:
                    max_y = max(max_y, y1)
    for d in page.get_drawings():
        r = d["rect"]
        if r.y0 > start_y - 5 and r.y1 < footer_y:
            max_y = max(max_y, r.y1)
    return max_y


def trim_vertical_whitespace(img: Image.Image, pad: int = FINAL_PAD_PX, bg_thresh: int = 250) -> Image.Image:
    """Remove excess white space top/bottom, keep left/right margins, add pad."""
    arr = np.array(img)
    mask = np.any(arr < bg_thresh, axis=2)
    rows_with_content = np.any(mask, axis=1)
    if not np.any(rows_with_content):
        return img
    rmin, rmax = np.where(rows_with_content)[0][[0, -1]]
    rmin = max(0, rmin - pad)
    rmax = min(arr.shape[0] - 1, rmax + pad)
    return img.crop((0, rmin, arr.shape[1], rmax + 1))


def render_region(page: fitz.Page, top: float, bottom: float) -> Image.Image:
    """Render a vertical slice of a page and return a PIL Image."""
    mat = fitz.Matrix(SCALE, SCALE)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    top_px = max(0, int(round(top * SCALE)))
    bot_px = min(pix.height, int(round(bottom * SCALE)))
    if bot_px <= top_px + 10:
        bot_px = top_px + 40
    return img.crop((0, top_px, pix.width, bot_px))


def stitch_vertical(top_img: Image.Image, bottom_img: Image.Image, gap: int = STITCH_GAP_PX) -> Image.Image:
    """Stack two images with a small white gap between them."""
    w = max(top_img.width, bottom_img.width)
    h = top_img.height + gap + bottom_img.height
    out = Image.new("RGB", (w, h), (255, 255, 255))
    out.paste(top_img, (0, 0))
    out.paste(bottom_img, (0, top_img.height + gap))
    return out


def extract_year(pdf_path: Path, out_dir: Path, year: int) -> int:
    """Extract all MCQ screenshots for one paper. Returns number of questions saved."""
    doc = fitz.open(pdf_path)
    qs = find_mcq_positions(doc)
    stimuli, stim_y_by_page = find_stimuli(doc, qs)

    if len(qs) != 20:
        print(f"  WARNING: expected 20 questions, found {len(qs)}")
        found = {q["num"] for q in qs}
        missing = sorted(set(range(1, 21)) - found)
        if missing:
            print(f"  Missing: {missing}")

    if stimuli:
        groups = {}
        for qn, info in stimuli.items():
            key = (info["q_first"], info["q_last"])
            groups.setdefault(key, []).append(qn)
        for (a, b), members in sorted(groups.items()):
            label = f"Q{a}" if a == b else f"Q{a}–{b}"
            print(f"  Stimulus detected for {label}")

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for i, q in enumerate(qs):
        page = doc[q["page"]]
        qnum = q["num"]
        page_idx = q["page"]

        # ----- normal question vertical bounds -----
        q_top = max(0.0, q["y"] - TOP_PAD_PTS)

        # Candidate bottoms (we take the earliest one)
        candidates = []

        # 1. Next question on the same page
        if i + 1 < len(qs) and qs[i + 1]["page"] == page_idx:
            candidates.append(qs[i + 1]["y"] - BETWEEN_PAD_PTS)

        # 2. Any stimulus intro that appears later on the same page
        #    (prevents a preceding question from swallowing the next stimulus)
        for sy in stim_y_by_page.get(page_idx, []):
            if sy > q["y"] + 5:
                candidates.append(sy - BETWEEN_PAD_PTS)

        # 3. Natural content bottom / footer
        content_bottom = get_content_bottom(page, q["y"], FOOTER_Y)
        candidates.append(min(FOOTER_Y, content_bottom + 12))

        q_bottom = min(candidates) if candidates else (q_top + 80)
        if q_bottom <= q_top + 30:
            q_bottom = q_top + 80

        stim = stimuli.get(qnum)

        if stim is None:
            # No stimulus – ordinary crop
            crop = render_region(page, q_top, q_bottom)
            crop = trim_vertical_whitespace(crop)
        else:
            is_first_in_group = (qnum == stim["q_first"])

            if is_first_in_group:
                # Extend the top of the crop up to the stimulus intro
                if stim["page"] == page_idx:
                    top = max(0.0, stim["y_intro"] - TOP_PAD_PTS)
                    crop = render_region(page, top, q_bottom)
                    crop = trim_vertical_whitespace(crop)
                else:
                    crop = render_region(page, q_top, q_bottom)
                    crop = trim_vertical_whitespace(crop)
            else:
                # Later question in a shared group → stitch stimulus + this question only
                stim_page = doc[stim["page"]]
                stim_top = max(0.0, stim["y_intro"] - TOP_PAD_PTS)

                if stim["first_q_page"] == stim["page"] and stim["first_q_y"] > 0:
                    stim_bottom = stim["first_q_y"] - BETWEEN_PAD_PTS
                else:
                    stim_bottom = FOOTER_Y

                if stim_bottom <= stim_top + 20:
                    stim_bottom = stim_top + 60

                stim_img = render_region(stim_page, stim_top, stim_bottom)
                stim_img = trim_vertical_whitespace(stim_img, pad=12)

                q_img = render_region(page, q_top, q_bottom)
                q_img = trim_vertical_whitespace(q_img, pad=12)

                crop = stitch_vertical(stim_img, q_img)
                crop = trim_vertical_whitespace(crop, pad=FINAL_PAD_PX)

        out_name = f"HSC_BUSINESS_{year}_Q{qnum:02d}.png"
        out_path = out_dir / out_name
        crop.save(out_path, "PNG", optimize=True)
        print(f"  Q{qnum:02d} → {out_name}  ({crop.size[0]}×{crop.size[1]})")
        count += 1

    doc.close()
    return count


def find_pdf(year: int, pdf_root: Path) -> Path | None:
    """Return the PDF path for a year, or None if missing."""
    candidates = [
        pdf_root / str(year) / f"{year}-hsc-business-studies.pdf",
        pdf_root / f"{year}-hsc-business-studies.pdf",
        pdf_root / str(year) / f"{year}_hsc_business_studies.pdf",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract HSC Business Studies multiple-choice question screenshots"
    )
    parser.add_argument(
        "--year", type=int,
        help="Process only this year (e.g. 2024). Default: all years found."
    )
    parser.add_argument(
        "--pdf-root", type=str, default=str(DEFAULT_PDF_ROOT),
        help=f"Root folder containing year subfolders (default: {DEFAULT_PDF_ROOT})"
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
        help=f"Where to save the PNG screenshots (default: {DEFAULT_OUT_DIR})"
    )
    args = parser.parse_args()

    pdf_root = Path(args.pdf_root)
    out_dir = Path(args.out_dir)

    if not pdf_root.is_dir():
        print(f"ERROR: PDF root folder not found: {pdf_root.resolve()}")
        print()
        print("Expected structure:")
        print("  HSC_PAPERS/BusinessStudies/")
        print("    2016/2016-hsc-business-studies.pdf")
        print("    2017/2017-hsc-business-studies.pdf")
        print("    ...")
        print("    2025/2025-hsc-business-studies.pdf")
        print()
        print("Run this script from the folder that contains the HSC_PAPERS directory.")
        sys.exit(1)

    years = [args.year] if args.year else YEARS
    total = 0
    processed = 0

    for year in years:
        pdf_path = find_pdf(year, pdf_root)
        if pdf_path is None:
            print(f"Skipping {year}: PDF not found under {pdf_root / str(year)}")
            continue

        print(f"\n=== {year} HSC Business Studies ===")
        print(f"  Source: {pdf_path}")
        n = extract_year(pdf_path, out_dir, year)
        total += n
        processed += 1

    print()
    if processed == 0:
        print("No papers were processed. Check that the PDFs exist in the expected locations.")
        sys.exit(1)
    else:
        print(f"Done. Extracted {total} question screenshots into: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
