# Document Scanner, OCR, and Translation

## Overview
This project provides a complete pipeline to scan documents from a camera, extract text via OCR with post-correction, and train an English→French translation model with attention using deep learning. It includes tools for scanning with perspective correction, text cleanup, and a seq2seq GRU model with attention. 

## Features
- Page detection using color mask, edges, and Hough line voting with stability-based auto-capture. 
- OCR with error correction for common character confusions, spelling, grammar, and sentence capitalization. 
- Neural machine translation: encoder–decoder GRU with dot-product attention for English→French. 

## Project structure
- scan.py — camera-based page detector and perspective correction; saves scanned images. 
- ocr.ipynb — OCR + text cleanup; outputs raw and corrected text files. 
- main.ipynb or main.py — English→French translation model training and evaluation. 
- eng-french.csv — parallel corpus used by the translation model. 

## Prerequisites
- Python 3.8+ and pip. 
- Tesseract OCR installed system-wide (examples): 
  - macOS: brew install tesseract 
  - Ubuntu/Debian: sudo apt-get install tesseract-ocr 
  - Windows: Install from tesseract-ocr.github.io and add to PATH, or set pytesseract.pytesseract.tesseract_cmd in code. 
- Recommended: a virtual environment to isolate dependencies. 

## Installation
```bash
# 1) Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install Python dependencies
pip install --upgrade pip
pip install opencv-python numpy pillow pytesseract textblob pyspellchecker wordcloud \
            tensorflow keras seaborn scikit-learn matplotlib pandas

# 3) (Optional) Download TextBlob corpora for better corrections
python -m textblob.download_corpora
```

## Run order (quick start)

### 1) Scan a page
```bash
python scan.py
```
- Align the page; when the yellow quadrilateral is stable, it auto-saves:
  - scanned_page.png (color) 
  - scanned_page_bw.png (binary/thresholded) 
- Keys: ESC to quit, SPACE for manual capture. 

### 2) Run OCR and cleanup
Option A — Jupyter Notebook:
- Open ocr.ipynb and run all cells; it reads scanned_page_bw.png and creates:
  - extracted_text_raw.txt 
  - extracted_text_corrected.txt 

Option B — Convert to a script (if you prefer CLI):
```bash
jupyter nbconvert --to script ocr.ipynb
python ocr.py
```

### 3) Train the translation model
Option A — Jupyter Notebook:
- Ensure eng-french.csv is present.
- Open main.ipynb and run all cells to:
  - tokenize datasets 
  - train GRU encoder–decoder with attention 
  - evaluate on the test split 

Option B — Python script:
```bash
python main.py
```
- If your project uses main.ipynb, either run it in Jupyter or convert to a script as shown above. 

## Configuration tips
- If Tesseract is not on PATH, set in the notebook/script:
```python
# Example (adjust to your installation path)
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
# or on macOS (Homebrew on Apple Silicon)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
```
- scan.py parameters:
  - auto_capture=True/False 
  - stable_frames=8 
  - process_width=900–1200 (speed vs accuracy) 

## GitHub: first push
```bash
# From your project root
git init
git add .
git commit -m "Initial commit: scanner, OCR, translation"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Troubleshooting
- No page detected: improve lighting, ensure high page contrast, increase process_width, or place a darker background under the page. 
- OCR poor quality: use scanned_page_bw.png, ensure 300+ DPI equivalent, adjust thresholding or try adaptive methods in scan.py. 
- CUDA/GPU issues: TensorFlow defaults to CPU if no compatible GPU; confirm versions or install CPU-only builds if needed. 

## Acknowledgments
- OpenCV for vision; Tesseract for OCR; Keras/TensorFlow for sequence modeling.
