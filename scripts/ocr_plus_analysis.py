import pytesseract
from PIL import Image
import cv2
import numpy as np
from bs4 import BeautifulSoup
import argparse
import os

# Configure Tesseract to use HOCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\nihaa\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'  # Change if needed

def extract_hocr(image_path):
    image = Image.open(image_path)
    hocr = pytesseract.image_to_pdf_or_hocr(image, extension='hocr')
    return hocr.decode('utf-8')

def parse_hocr(hocr_data):
    soup = BeautifulSoup(hocr_data, 'xml')
    words = soup.find_all(class_='ocrx_word')
    word_info = []

    for word in words:
        title = word.get('title', '')
        if 'bbox' in title:
            parts = title.split(';')
            bbox_str = [p.strip() for p in parts if p.strip().startswith('bbox')]
            if bbox_str:
                bbox = list(map(int, bbox_str[0].split()[1:]))
                font = [p.strip() for p in parts if p.strip().startswith('x_font')]

                word_info.append({
                    'text': word.get_text(),
                    'bbox': bbox,
                    'font': font[0].split()[1] if font else None
                })
    return word_info

def analyze_word_data(word_info):
    spacing_inconsistencies = []
    font_anomalies = []
    vertical_align_issues = []

    previous_line_y = None
    fonts_seen = set()

    for i in range(len(word_info) - 1):
        current = word_info[i]
        next_word = word_info[i + 1]

        # Check spacing (irregular horizontal spacing)
        cur_right = current['bbox'][2]
        next_left = next_word['bbox'][0]
        gap = next_left - cur_right
        if 0 < gap > 30:  # arbitrary threshold
            spacing_inconsistencies.append((current['text'], next_word['text'], gap))

        # Check vertical alignment
        cur_baseline = current['bbox'][3]
        next_baseline = next_word['bbox'][3]
        if abs(cur_baseline - next_baseline) > 10:  # threshold for baseline misalignment
            vertical_align_issues.append((current['text'], next_word['text'], cur_baseline, next_baseline))

        # Font anomaly detection
        if current['font']:
            fonts_seen.add(current['font'])
        if next_word['font']:
            fonts_seen.add(next_word['font'])

    if len(fonts_seen) > 1:
        font_anomalies = list(fonts_seen)

    return {
        'spacing_inconsistencies': spacing_inconsistencies,
        'vertical_align_issues': vertical_align_issues,
        'font_anomalies': font_anomalies
    }

def run_analysis(image_path):
    hocr_data = extract_hocr(image_path)
    word_info = parse_hocr(hocr_data)
    result = analyze_word_data(word_info)

    # Extract and print raw text using pytesseract
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    print("\n--- Extracted Text ---")
    print(extracted_text)

    print("\n---  OCR Analysis Report ---")
    if result['font_anomalies']:
        print(f"[!] Font Anomalies Detected: {result['font_anomalies']}")
    else:
        print("[-] No Font Anomalies Found.")

    if result['spacing_inconsistencies']:
        print(f"[!] Spacing Inconsistencies: {len(result['spacing_inconsistencies'])}")
        for item in result['spacing_inconsistencies'][:5]:
            print(f"    Between '{item[0]}' and '{item[1]}': gap = {item[2]}px")
    else:
        print("[-] No Spacing Issues Found.")

    if result['vertical_align_issues']:
        print(f"[!] Vertical Misalignments: {len(result['vertical_align_issues'])}")
        for item in result['vertical_align_issues'][:5]:
            print(f"    Between '{item[0]}' and '{item[1]}': baselines = {item[2]} vs {item[3]}")
    else:
        print("[-] No Vertical Misalignments Detected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect forgery imperfections using Tesseract OCR + HOCR")
    parser.add_argument("image_path", help="Path to input image (jpg, png, etc.)")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"File not found: {args.image_path}")
    else:
        run_analysis(args.image_path)
