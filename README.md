# businesstesting

Extracts individual screenshots of every multiple-choice question (Section I, Questions 1–20) from the NSW HSC Business Studies past papers (2016–2025).

## Contents

- `*-hsc-business-studies.pdf` — official past papers
- `extract_mcq_screenshots.py` — the extraction program
- `mcq_screenshots/` — generated PNG screenshots named `HSC_BUSINESS_YYYY_QNN.png` (run the script to produce)

## Requirements

```bash
pip install pymupdf Pillow numpy
```

## Usage

```bash
# Extract all years (places images in ./mcq_screenshots/)
python extract_mcq_screenshots.py --all

# Single year
python extract_mcq_screenshots.py --year 2024

# Custom output directory
python extract_mcq_screenshots.py --all --out-dir my_screenshots
```

The script automatically:

1. Locates each question number (1–20) using precise text position analysis
2. Determines the vertical bounds of the question (including any tables/diagrams via vector drawings)
3. Renders the region at high resolution (~144 dpi → 1190 px wide)
4. Trims excess whitespace while keeping consistent padding and full page-width margins
5. Saves a clean PNG for each question

## Notes

- Works across all papers 2016–2025 despite minor layout variations
- Captures associated tables, figures and diagrams
- Output images are suitable for flashcards, study apps, Anki decks, or further processing
