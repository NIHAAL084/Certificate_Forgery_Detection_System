import os
import re
import tempfile
import hashlib
import logging
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi import Body

from PIL import Image as PILImage
from exif import Image as ExifImage
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata

import pytesseract
from pytesseract import TesseractNotFoundError
from bs4 import BeautifulSoup

import torch
from transformers import ViTForImageClassification, ViTImageProcessor

from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("forgery_api")

# Configure Tesseract to use HOCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\nihaa\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'  # Change if needed

# --- Load ViT model & processor once at startup ---
MODEL_DIR = "./vit_forgery_output/checkpoint-best"
processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
model = ViTForImageClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- Load AI text detector ---
DET_MODEL = "openai-community/roberta-base-openai-detector"
tokenizer = AutoTokenizer.from_pretrained(DET_MODEL)
detector  = AutoModelForSequenceClassification.from_pretrained(DET_MODEL)

app = FastAPI(title="Certificate Forgery Detection API")

# List of known suspicious editing tools
SUSPICIOUS_SOFTWARE = [
    'Photoshop', 'GIMP', 'Paint', 'Canva',
    'Affinity Photo', 'Pixelmator', 'Snapseed',
    'Lightroom', 'Capture One', 'Darktable',
]

class TextPayload(BaseModel):
    text: str


# ======================
# Metadata helpers
# ======================

def compute_hashes(path: str) -> dict:
    hashes = {}
    for algo in ('md5', 'sha1'):
        h = hashlib.new(algo)
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        hashes[algo] = h.hexdigest()
    return hashes

def extract_exif(path: str) -> dict:
    with open(path, 'rb') as f:
        try:
            img = ExifImage(f)
            if img.has_exif:
                data = {}
                for tag in img.list_all():
                    try:
                        data[tag] = getattr(img, tag)
                    except Exception:
                        pass
                return data
        except Exception:
            pass
    return {}

def extract_basic_pil(path: str) -> dict:
    try:
        with PILImage.open(path) as img:
            return {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'info': img.info,
                'quantization': getattr(img, 'quantization', None)
            }
    except Exception:
        return {}

def extract_hachoir_tools(path: str) -> list:
    tools = []
    parser = createParser(path)
    if not parser:
        return tools
    metadata = extractMetadata(parser)
    if not metadata:
        return tools
    for key in metadata.exportDictionary().get('metadata', {}):
        if any(k in key.lower() for k in ('tool','creator','producer')):
            val = metadata.get(key)
            if val:
                tools.append(str(val))
    return tools

def scan_xmp_creator(path: str) -> str:
    try:
        with open(path, 'rb') as f:
            data = f.read()
            m = re.search(rb'<xmp:CreatorTool>([^<]+)</xmp:CreatorTool>', data)
            if m:
                return m.group(1).decode('utf-8', errors='ignore')
    except Exception:
        pass
    return ''

def detect_editing_software(exif_data: dict, basic_info: dict, hachoir_tools: list, xmp_tool: str) -> dict:
    suspects = []
    if exif_data.get('software'):
        suspects.append(exif_data['software'])
    if basic_info.get('info', {}).get('Software'):
        suspects.append(basic_info['info']['Software'])
    if xmp_tool:
        suspects.append(xmp_tool)
    suspects.extend(hachoir_tools)
    found = set()
    for entry in suspects:
        for sw in SUSPICIOUS_SOFTWARE:
            if sw.lower() in str(entry).lower():
                found.add(sw)
    return {'raw': suspects, 'suspicious': list(found)}

def compare_timestamps(path: str, exif_data: dict) -> dict:
    stats = {}
    fs_ct = os.path.getctime(path)
    fs_mt = os.path.getmtime(path)
    stats['fs_creation'] = datetime.fromtimestamp(fs_ct).isoformat()
    stats['fs_modification'] = datetime.fromtimestamp(fs_mt).isoformat()
    exif_dt = exif_data.get('datetime_original') or exif_data.get('datetime')
    if exif_dt:
        try:
            exif_ts = datetime.strptime(exif_dt, '%Y:%m:%d %H:%M:%S')
            stats['exif_datetime'] = exif_ts.isoformat()
            stats['delta_fs_ct_exif'] = (exif_ts - datetime.fromtimestamp(fs_ct)).total_seconds()
            stats['delta_fs_mt_exif'] = (exif_ts - datetime.fromtimestamp(fs_mt)).total_seconds()
        except Exception:
            stats['exif_datetime'] = exif_dt
    return stats

def analyze_metadata(path: str) -> dict:
    exif = extract_exif(path)
    basic = extract_basic_pil(path)
    tools = extract_hachoir_tools(path)
    xmp = scan_xmp_creator(path)
    report = {
        'hashes': compute_hashes(path),
        'exif': exif,
        'basic': basic,
        'editing_software': detect_editing_software(exif, basic, tools, xmp),
        'timestamps': compare_timestamps(path, exif)
    }
    if basic.get('quantization') and len(basic['quantization']) != 2:
        report['quantization_warning'] = f"Abnormal quantization tables: {len(basic['quantization'])}"
    return report

# ======================
# OCR helpers
# ======================

def extract_hocr(path: str) -> str:
    with PILImage.open(path) as img:
        data = pytesseract.image_to_pdf_or_hocr(img, extension='hocr')
    return data.decode('utf-8')

def parse_hocr(hocr_data: str) -> list:
    soup = BeautifulSoup(hocr_data, 'xml')
    words = soup.find_all(class_='ocrx_word')
    out = []
    for w in words:
        t = w.get('title','')
        if 'bbox' in t:
            parts = t.split(';')
            bbox = next((p for p in parts if p.strip().startswith('bbox')), '')
            coords = list(map(int, bbox.split()[1:])) if bbox else []
            font = next((p.split()[1] for p in parts if p.strip().startswith('x_font')), None)
            out.append({'text': w.get_text(), 'bbox': coords, 'font': font})
    return out

def analyze_word_data(word_info: list) -> dict:
    spacing, vertical, fonts = [], [], set()
    for curr, nxt in zip(word_info, word_info[1:]):
        if curr['bbox'] and nxt['bbox']:
            gap = nxt['bbox'][0] - curr['bbox'][2]
            if gap > 30:
                spacing.append({'pair':(curr['text'], nxt['text']), 'gap': gap})
            baseline = abs(curr['bbox'][3] - nxt['bbox'][3])
            if baseline > 10:
                vertical.append({'pair':(curr['text'], nxt['text']), 'baseline_diff': baseline})
        if curr['font']: fonts.add(curr['font'])
        if nxt['font']: fonts.add(nxt['font'])
    anomalies = list(fonts) if len(fonts)>1 else []
    return {
        'spacing_inconsistencies': spacing,
        'vertical_align_issues': vertical,
        'font_anomalies': anomalies
    }

def perform_ocr_analysis(path: str) -> dict:
    try:
        hocr = extract_hocr(path)
        words = parse_hocr(hocr)
        analysis = analyze_word_data(words)
        with PILImage.open(path) as img:
            text = pytesseract.image_to_string(img)
        return {'extracted_text': text, 'analysis_report': analysis}
    except TesseractNotFoundError as e:
        logger.error("Tesseract not found: %s", e)
        return {'extracted_text':'', 'analysis_report':{}, 'ocr_error':'Tesseract not installed'}
    except Exception as e:
        logger.exception("OCR error")
        return {'extracted_text':'', 'analysis_report':{}, 'ocr_error': str(e)}

# ======================
# ViT classification
# ======================

def classify_with_vit(path: str) -> dict:
    try:
        img = PILImage.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits
            idx = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1)[0]
        label = "fraudulent" if idx == 1 else "unedited"
        confidence = float(probs[idx])
        return {'label': label, 'confidence': confidence}
    except Exception as e:
        logger.exception("ViT classify error")
        return {'label':'error','confidence':0.0,'error':str(e)}

# ======================
# API endpoint
# ======================

@app.post("/analyze_image")
async def analyze_image(request: Request):
    content_type = request.headers.get('content-type','')
    logger.debug("Content-Type: %s", content_type)

    # pull bytes & filename
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        if 'data' not in form:
            raise HTTPException(400, "Missing 'data' field")
        up = form['data']
        data = await up.read()
        fname = up.filename or "upload"
    else:
        data = await request.body()
        if not data:
            raise HTTPException(400, "Empty request body")
        fname = "upload"

    # write to temp
    suffix = os.path.splitext(fname)[1] or ".jpg"
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    path = tmpf.name
    try:
        tmpf.write(data)
        tmpf.close()

        metadata = analyze_metadata(path)
        ocr_res   = perform_ocr_analysis(path)
        vit_res   = classify_with_vit(path)
    except Exception as e:
        logger.exception("Processing error")
        raise HTTPException(500, f"Processing error: {e}")
    finally:
        try:
            os.unlink(path)
        except:
            logger.warning("Could not delete temp file %s", path)

    resp = {
        'metadata_report':       metadata,
        'extracted_text':        ocr_res.get('extracted_text',''),
        'ocr_analysis_report':   ocr_res.get('analysis_report', {}),
        'VIT_classifier_report': vit_res
    }
    if 'ocr_error' in ocr_res:
        resp['ocr_error'] = ocr_res['ocr_error']

    return JSONResponse(content=resp)

@app.post("/classify_text")
async def classify_text_endpoint(payload: dict = Body(...)):
    if 'text' not in payload:
        raise HTTPException(400, "Missing 'text' field in JSON body")
    text = payload['text']

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = detector(**inputs)
        probs   = F.softmax(outputs.logits, dim=1)[0]

    pred_idx   = torch.argmax(probs).item()
    label      = ["Human-written", "AI-generated"][pred_idx]
    confidence = float(probs[pred_idx])

    return JSONResponse(content={"AI_detection": {"label": label, "confidence": confidence}})

