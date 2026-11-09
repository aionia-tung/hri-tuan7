"""CLI: phan vung map.pgm, gan nhan ngu nghia, xuat zones.yaml + anh nhan + anh truc quan.

    ros2 run tb4_voice segment_map --map maps/map.yaml --semantics config/zone_semantics.yaml \
        --out maps

Ket qua trong --out:
  zones.yaml         thong tin khu vuc (id, chu cai, ten, tam, diem dich, dien tich, da giac)
  zones.json         cung noi dung, dang JSON
  zone_labels.png    anh uint8 cung kich thuoc map, pixel = id khu vuc (0 = khong phai free)
  map_segmented.png  ban do phan vung co nhan (de nop / dua vao bao cao)
"""

import argparse
import json
import string
from pathlib import Path

import cv2
import numpy as np
import yaml

from tb4_voice.map_segmentation import SegmentParams, load_map, segment

# Bang mau pastel (RGB) cho toi da 12 khu vuc
PALETTE = [(141, 211, 199), (255, 237, 111), (190, 186, 218), (251, 128, 114),
           (128, 177, 211), (253, 180, 98), (179, 222, 105), (252, 205, 229),
           (188, 128, 189), (204, 235, 197), (217, 217, 217), (255, 255, 179)]


def assign_semantics(info, labels, regions, semantics):
    """Ghep vung tu dong voi muc trong zone_semantics.yaml theo seed."""
    by_label = {r.label: r for r in regions}
    assigned = {}
    for entry in semantics.get('zones', []):
        col, row = info.world_to_pixel(*entry['seed'])
        lab = int(labels[row, col]) if 0 <= row < info.height and 0 <= col < info.width else 0
        if lab == 0:
            raise ValueError(f"Seed {entry['seed']} cua {entry['letter']} khong nam trong vung nao")
        if lab in assigned:
            raise ValueError(f"{entry['letter']} va {assigned[lab]['letter']} cung roi vao vung {lab}"
                             ' - giam min_area/merge_ratio hoac doi seed')
        assigned[lab] = entry
    used = {e['letter'] for e in assigned.values()}
    spare = [c for c in string.ascii_uppercase if c not in used]
    zones = []
    for lab in sorted(by_label, key=lambda k: (assigned.get(k, {}).get('letter', '~'), k)):
        r = by_label[lab]
        entry = assigned.get(lab) or {'letter': spare.pop(0)}
        letter = entry['letter']
        zones.append({
            'id': lab,
            'letter': letter,
            'name': entry.get('name', f'Khu vực {letter}'),
            'aliases': list(entry.get('aliases', [])),
            'center': [round(r.centroid[0], 2), round(r.centroid[1], 2)],
            'goal': [round(r.goal[0], 2), round(r.goal[1], 2), float(entry.get('yaw', 0.0))],
            'clearance_m': round(r.clearance, 2),
            'area_m2': round(r.area, 1),
            'reachable': bool(r.reachable),
            'polygon': r.polygon,
        })
    return zones


def render(info, labels, zones, path, scale=2):
    """Anh phan vung: mau theo khu vuc, tuong den, chu cai + ten + diem dich."""
    base = cv2.cvtColor(info.image, cv2.COLOR_GRAY2BGR)
    color = base.copy()
    for z in zones:
        r, g, b = PALETTE[(z['id'] - 1) % len(PALETTE)]
        color[labels == z['id']] = (b, g, r)
    img = cv2.addWeighted(color, 0.75, base, 0.25, 0)
    img[info.image < 100] = (0, 0, 0)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for z in zones:
        pts = np.array([info.world_to_pixel(x, y) for x, y in z['polygon']], np.int32) * scale
        cv2.polylines(img, [pts], True, (60, 60, 60), 1, cv2.LINE_AA)
        gc, gr = info.world_to_pixel(*z['goal'][:2])
        cv2.drawMarker(img, (gc * scale, gr * scale), (0, 0, 200), cv2.MARKER_STAR, 14, 2)
        # chu cai dat ngay tren diem dich (tam hinh hoc co the roi vao vat can)
        org = (gc * scale - 10, gr * scale - 12)
        cv2.putText(img, z['letter'], org, cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 4,
                    cv2.LINE_AA)
        cv2.putText(img, z['letter'], org, cv2.FONT_HERSHEY_DUPLEX, 1.0, (20, 20, 20), 2,
                    cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--map', required=True, help='map.yaml cua map_server')
    ap.add_argument('--semantics', required=True, help='config/zone_semantics.yaml')
    ap.add_argument('--out', default='.', help='thu muc ghi ket qua')
    args = ap.parse_args(argv)

    info = load_map(args.map)
    semantics = yaml.safe_load(Path(args.semantics).read_text())
    params = SegmentParams(**semantics.get('segmentation', {}),
                           avoid=semantics.get('avoid', []))
    labels, regions, _ = segment(info, params, robot_xy=semantics.get('robot_start', (0.0, 0.0)))
    zones = assign_semantics(info, labels, regions, semantics)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {'map': str(Path(args.map).name), 'frame_id': 'map', 'resolution': info.resolution,
           'origin': list(info.origin), 'label_image': 'zone_labels.png', 'zones': zones}
    with open(out / 'zones.yaml', 'w', encoding='utf-8') as f:
        f.write('# Sinh tu dong boi segment_map - sua ten/alias trong config/zone_semantics.yaml\n')
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, default_flow_style=None)
    (out / 'zones.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
    cv2.imwrite(str(out / 'zone_labels.png'), labels.astype(np.uint8))
    render(info, labels, zones, out / 'map_segmented.png')

    for z in zones:
        warn = '' if z['reachable'] else '  (!) khong di toi duoc'
        print(f"{z['letter']}  {z['name']:<18} tam=({z['center'][0]:6.2f}, {z['center'][1]:6.2f})"
              f"  dich=({z['goal'][0]:6.2f}, {z['goal'][1]:6.2f})  {z['area_m2']:5.1f} m2{warn}")
    print(f'Da ghi {len(zones)} khu vuc vao {out}/zones.yaml')


if __name__ == '__main__':
    main()
