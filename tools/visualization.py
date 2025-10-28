from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from mmeval import COCODetection
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR.parent / 'OmniDocBench_data'
IMG_DIR = BASE_DIR.parent / 'dataset' #DATA_DIR / 'images'
GT_PATH = DATA_DIR / 'InvofoxBench.json' #'OmniDocBench.json'
PRED_PATH = DATA_DIR / 'predictions' / 'docling_granite_full.json'#'mineru_vlm_predictions_2.json'

CAT_MAP = {
    'equation_isolated': 'equation',
}


def normalize_cat(cat: str) -> str:
    return CAT_MAP.get(cat, cat)


def poly_to_bbox(poly):
    x1, y1, x2, _ = poly[0], poly[1], poly[2], poly[3]
    x3, y3 = poly[4], poly[5]
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y3), max(y1, y3)
    return [left, top, right, bottom]


def draw_dashed_rectangle(draw: ImageDraw.ImageDraw, bbox, color, width=2, dash=12):
    x1, y1, x2, y2 = bbox

    def _draw_segment(xa, ya, xb, yb):
        dx, dy = xb - xa, yb - ya
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length == 0:
            return
        step = max(int(length // dash), 1)
        for i in range(0, step, 2):
            start = i / step
            end = min((i + 1) / step, 1)
            xs, ys = xa + dx * start, ya + dy * start
            xe, ye = xa + dx * end, ya + dy * end
            draw.line([xs, ys, xe, ye], fill=color, width=width)

    _draw_segment(x1, y1, x2, y1)
    _draw_segment(x2, y1, x2, y2)
    _draw_segment(x2, y2, x1, y2)
    _draw_segment(x1, y2, x1, y1)


def bbox_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def visualize(target_image: str, prediction_path: Path, output_path: Path | None = None):
    gt_samples = json.loads(GT_PATH.read_text())
    pred_samples = {sample['page_info']['image_path']: sample for sample in json.loads(prediction_path.read_text())}

    gt_sample = next((sample for sample in gt_samples if sample['page_info']['image_path'] == target_image), None)
    if gt_sample is None:
        raise FileNotFoundError(f'{target_image} not found in ground truth JSON')

    pred_sample = pred_samples.get(target_image)
    if pred_sample is None:
        print(f'{target_image} not found in prediction JSON {prediction_path}')
        #raise FileNotFoundError(f'{target_image} not found in prediction JSON {prediction_path}')

    image_path = IMG_DIR / target_image
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    color_map: dict[str, tuple[int, int, int]] = {}

    def get_color(category: str) -> tuple[int, int, int]:
        if category not in color_map:
            color_map[category] = tuple(random.randint(0, 255) for _ in range(3))
        return color_map[category]

    categories = sorted({
        normalize_cat(det['category_type'])
        for det in gt_sample['layout_dets']
    } | {
        normalize_cat(det['category_type'])
        for det in pred_sample['layout_dets']
    })

    # Prepare COCO metric to print overall stats
    metric = COCODetection(
        dataset_meta={'CLASSES': tuple(categories)},
        iou_thrs=[0.5, 0.75],
        classwise=True,
    )

    cat_to_idx = {c: i for i, c in enumerate(categories)}

    gt_record = {
        'img_id': 0,
        'width': gt_sample['page_info']['width'],
        'height': gt_sample['page_info']['height'],
        'bboxes': [],
        'labels': [],
        'ignore_flags': [],
    }
    for det in gt_sample['layout_dets']:
        cat = normalize_cat(det['category_type'])
        if cat not in cat_to_idx:
            continue
        gt_record['bboxes'].append(poly_to_bbox(det['poly']))
        gt_record['labels'].append(cat_to_idx[cat])
        gt_record['ignore_flags'].append(False)
    gt_record['bboxes'] = np.array(gt_record['bboxes'], dtype=np.float32)
    gt_record['labels'] = np.array(gt_record['labels'], dtype=np.int64)
    gt_record['ignore_flags'] = np.array(gt_record['ignore_flags'], dtype=bool)

    pred_record = {
        'img_id': 0,
        'bboxes': [],
        'scores': [],
        'labels': [],
    }
    for det in pred_sample['layout_dets']:
        cat = normalize_cat(det['category_type'])
        if cat not in cat_to_idx:
            continue
        pred_record['bboxes'].append(poly_to_bbox(det['poly']))
        pred_record['scores'].append(det.get('score', 1.0))
        pred_record['labels'].append(cat_to_idx[cat])
    pred_record['bboxes'] = np.array(pred_record['bboxes'], dtype=np.float32)
    pred_record['scores'] = np.array(pred_record['scores'], dtype=np.float32)
    pred_record['labels'] = np.array(pred_record['labels'], dtype=np.int64)

    res = metric([pred_record], [gt_record])
    print('Metrics for', target_image)
    if 'bbox_mAP_50' in res and 'bbox_mAP_75' in res:
        print(f"  mAP@0.50: {res['bbox_mAP_50']:.4f}")
        print(f"  mAP@0.75: {res['bbox_mAP_75']:.4f}")
    else:
        print(f"  mAP: {res.get('bbox_mAP', float('nan')):.4f}")

    gt_entries = []
    for anno in gt_sample['layout_dets']:
        cat = normalize_cat(anno['category_type'])
        bbox = poly_to_bbox(anno['poly'])
        color = get_color(cat)
        draw.rectangle(bbox, outline=color, width=2)
        draw.text((bbox[2] - 50, bbox[1] - 12), cat, fill=color, font=font)
        gt_entries.append({'category': cat, 'bbox': bbox})

    gt_used = [False] * len(gt_entries)

    def find_best_match(pred_box, cat):
        best_iou, best_idx = 0.0, -1
        for idx, entry in enumerate(gt_entries):
            if gt_used[idx] or entry['category'] != cat:
                continue
            iou_val = bbox_iou(pred_box, entry['bbox'])
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = idx
        if best_idx >= 0:
            gt_used[best_idx] = True
        return best_iou

    for anno in pred_sample['layout_dets']:
        cat = normalize_cat(anno['category_type'])
        bbox = poly_to_bbox(anno['poly'])
        color = get_color(cat)
        draw_dashed_rectangle(draw, bbox, color=color, width=2, dash=12)
        iou_val = find_best_match(bbox, cat)
        draw.text((bbox[0], bbox[1] - 12), f'IoU {iou_val:.2f}', fill=color, font=font)
        if iou_val < 0.1:
            text = cat
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x = bbox[2] - 50 
            y = bbox[3]
            draw.text((x, y), text, fill=color, font=font)

    if output_path is None:
        output_path = Path(__file__).resolve().parent / f'{target_image}_overlay.png'
    img.save(output_path)
    print(f'Saved overlay visualization to {output_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize GT vs prediction overlay for OmniDocBench page')
    parser.add_argument('--image', required=True, help='Image filename (e.g., book_...png)')
    parser.add_argument('--pred', default=PRED_PATH, help='Prediction JSON path')
    parser.add_argument('--output', help='Optional output image path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pred_path = Path(args.pred)
    output = Path(args.output) if args.output else None
    visualize(args.image, pred_path, output)
