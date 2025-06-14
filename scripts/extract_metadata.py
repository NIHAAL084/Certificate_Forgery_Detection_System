import os
import sys
import re
import json
import hashlib
import argparse
from datetime import datetime

from PIL import Image
from exif import Image as ExifImage
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata

# List of suspicious editing tools
SUSPICIOUS_SOFTWARE = [
    'Photoshop', 'GIMP', 'Paint', 'Canva',
    'Affinity Photo', 'Pixelmator', 'Snapseed',
    'Lightroom', 'Capture One', 'Darktable',
]

def compute_hashes(path):
    hashes = {}
    for algo in ('md5', 'sha1'):
        h = hashlib.new(algo)
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        hashes[algo] = h.hexdigest()
    return hashes

def extract_exif(path):
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
    return None

def extract_basic_pil(path):
    try:
        img = Image.open(path)
        info = {
            'format': img.format,
            'mode': img.mode,
            'size': img.size,
            'info': img.info,
            'quantization': getattr(img, 'quantization', None)
        }
        return info
    except Exception:
        return None

def compare_timestamps(path, exif_data):
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

def detect_editing_software(exif_data, basic_info, hachoir_tools, xmp_tool):
    suspects = []
    software = None

    if exif_data:
        software = exif_data.get('software')
        if software:
            suspects.append(software)
    if basic_info and 'Software' in basic_info.get('info', {}):
        suspects.append(basic_info['info']['Software'])
    if xmp_tool:
        suspects.append(xmp_tool)
    if hachoir_tools:
        suspects.extend(hachoir_tools)

    found = set()
    for entry in suspects:
        for sw in SUSPICIOUS_SOFTWARE:
            if sw.lower() in str(entry).lower():
                found.add(sw)

    return {
        'raw': suspects,
        'suspicious': list(found)
    }

def extract_hachoir_tools(path):
    parser = createParser(path)
    if not parser:
        return []
    metadata = extractMetadata(parser)
    if not metadata:
        return []

    tools = []
    for key in metadata.exportDictionary().get('metadata', {}):
        if 'tool' in key.lower() or 'creator' in key.lower() or 'producer' in key.lower():
            val = metadata.get(key)
            if val:
                tools.append(str(val))
    return tools

def scan_xmp_creator(path):
    try:
        with open(path, 'rb') as f:
            data = f.read()
            match = re.search(rb'<xmp:CreatorTool>([^<]+)</xmp:CreatorTool>', data)
            if match:
                return match.group(1).decode('utf-8', errors='ignore')
    except Exception:
        pass
    return None

def analyze_file(path):
    report = {'file': path}
    report['hashes'] = compute_hashes(path)

    exif = extract_exif(path)
    report['exif'] = exif or {}

    basic = extract_basic_pil(path)
    report['basic'] = basic or {}

    hachoir = extract_hachoir_tools(path)
    xmp_tool = scan_xmp_creator(path)

    report['editing_software'] = detect_editing_software(report['exif'], report['basic'], hachoir, xmp_tool)

    if exif:
        report['timestamps'] = compare_timestamps(path, report['exif'])
    else:
        report['timestamps'] = compare_timestamps(path, {})

    if basic and basic.get('quantization') and len(basic['quantization']) != 2:
        report['quantization_warning'] = f"Abnormal quantization tables: {len(basic['quantization'])} (expected 2)"

    return report

def main():
    parser = argparse.ArgumentParser(description="Advanced Image Forensics Toolkit")
    parser.add_argument('paths', nargs='+', help="Image file(s) or directory(ies) to process")
    parser.add_argument('-o', '--output', default='report.json', help="Path to write JSON report")
    args = parser.parse_args()

    all_reports = []
    for p in args.paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    full = os.path.join(root, f)
                    try:
                        all_reports.append(analyze_file(full))
                    except Exception as e:
                        all_reports.append({'file': full, 'error': str(e)})
        elif os.path.isfile(p):
            try:
                all_reports.append(analyze_file(p))
            except Exception as e:
                all_reports.append({'file': p, 'error': str(e)})
        else:
            all_reports.append({'file': p, 'error': 'Not found'})

    with open(args.output, 'w', encoding='utf-8') as out:
        json.dump(all_reports, out, indent=2, ensure_ascii=False)

    print(f"Analysis complete. Report saved to {args.output}")

if __name__ == '__main__':
    main()
