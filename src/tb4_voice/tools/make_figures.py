#!/usr/bin/env python3
"""Ve hinh cho bao cao vao docs/images/.

    python3 tools/make_figures.py
Can: maps/ (sau segment_map), docs/asr_eval.csv (eval_asr.py), docs/run_demo/ (record_run.py).
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from tb4_voice.map_segmentation import (SegmentParams, free_mask, load_map,  # noqa: E402
                                        segment)
from tb4_voice.segment_map import PALETTE  # noqa: E402
from tb4_voice.zones import ZoneMap  # noqa: E402

IMG = PKG / 'docs' / 'images'
INK, MUTED = '#0b0b0b', '#52514e'
BLUE, ORANGE = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': '#c9c8c2',
                     'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})


def extent(info):
    x0, y0 = info.origin[0], info.origin[1]
    return [x0, x0 + info.width * info.resolution, y0, y0 + info.height * info.resolution]


def zone_rgb(labels, zone_ids, image):
    rgb = np.full(labels.shape + (3,), 205, np.uint8)
    for zid in zone_ids:
        rgb[labels == zid] = PALETTE[(zid - 1) % len(PALETTE)]
    rgb[image < 100] = 0
    return rgb


def draw_zones(ax, info, zone_map, names=True):
    labels = zone_map.labels
    ax.imshow(zone_rgb(labels, [z.id for z in zone_map.zones], info.image), extent=extent(info),
              interpolation='nearest')
    for z in zone_map.zones:
        gx, gy = z.goal[:2]
        ax.plot(gx, gy, marker='*', ms=11, color='#c0261f', mec='white', mew=0.8, zorder=5)
        text = f'{z.letter}\n{z.name}' if names else z.letter
        ax.text(gx, gy + 0.45, text, ha='center', va='bottom', fontsize=8.5 if names else 11,
                fontweight='bold', color=INK, zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75))
    ax.set_xlim(-10.1, 10.1)
    ax.set_ylim(-10.1, 10.1)
    ax.set_xlabel('x (m) - frame map')
    ax.set_ylabel('y (m)')


def fig_pipeline(info, sem):
    params = SegmentParams(**sem['segmentation'], avoid=sem['avoid'])
    free = free_mask(info, params)
    import cv2
    dt = cv2.distanceTransform(free, cv2.DIST_L2, 5) * info.resolution
    raw_params = SegmentParams(**{**sem['segmentation'], 'merge_ratio': 99.0, 'min_area': 0.0})
    raw, raw_regions, _ = segment(info, raw_params)
    zone_map = ZoneMap.load(PKG / 'maps' / 'zones.yaml')

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    ext = extent(info)
    axes[0].imshow(info.image, cmap='gray', extent=ext, interpolation='nearest')
    axes[0].set_title('(a) Bản đồ gốc map.pgm')
    im = axes[1].imshow(np.ma.masked_where(free == 0, dt), cmap='Blues', extent=ext,
                        interpolation='nearest')
    axes[1].contour(np.flipud(dt), levels=[params.core_dist], colors=[ORANGE], linewidths=1.0,
                    extent=ext)
    axes[1].set_facecolor('#3a3a3a')
    axes[1].set_title(f'(b) Distance transform, lõi dt > {params.core_dist} m')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02, label='khoảng cách tới vật cản (m)')
    cmap = ListedColormap([np.array(PALETTE[i % len(PALETTE)]) / 255 for i in range(raw.max())])
    axes[2].imshow(np.ma.masked_where(raw == 0, raw), cmap=cmap, extent=ext,
                   interpolation='nearest')
    axes[2].imshow(np.ma.masked_where(info.image > 100, info.image), cmap='gray', extent=ext,
                   interpolation='nearest')
    axes[2].set_title(f'(c) Watershed: {len(raw_regions)} vùng')
    draw_zones(axes[3], info, zone_map, names=False)
    axes[3].set_title(f'(d) Sau khi gộp: {len(zone_map.zones)} khu vực')
    for ax in axes:
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.set_yticks([-10, -5, 0, 5, 10])
        ax.set_xlabel('')
        ax.set_ylabel('')
    fig.tight_layout()
    fig.savefig(IMG / 'seg_pipeline.png', dpi=150)
    plt.close(fig)


def fig_zones(info):
    zone_map = ZoneMap.load(PKG / 'maps' / 'zones.yaml')
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    draw_zones(ax, info, zone_map)
    ax.plot(0, 0, 'o', color=INK, ms=6)
    ax.annotate('dock (0, 0)', (0, 0), (0.3, -0.7), fontsize=8.5, color=MUTED)
    ax.set_title('Bản đồ phân vùng: 9 khu vực, ★ = điểm đích Nav2', color=INK)
    fig.tight_layout()
    fig.savefig(IMG / 'map_zones.png', dpi=150)
    plt.close(fig)


def fig_route(info):
    run = PKG / 'docs' / 'run_demo'
    if not (run / 'trajectory.csv').exists():
        return
    zone_map = ZoneMap.load(PKG / 'maps' / 'zones.yaml')
    traj = np.array([[float(r['x']), float(r['y'])] for r in
                     csv.DictReader(open(run / 'trajectory.csv'))])
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    draw_zones(ax, info, zone_map)
    if (run / 'planned_route.json').exists():
        plan = np.array(json.loads((run / 'planned_route.json').read_text()))
        ax.plot(plan[:, 0], plan[:, 1], '--', color=BLUE, lw=2, label='Đường dự kiến (planner)',
                zorder=3)
    ax.plot(traj[:, 0], traj[:, 1], '-', color=ORANGE, lw=2, label='Quỹ đạo thực tế (AMCL)',
            zorder=4)
    ax.plot(*traj[0], 'o', color=ORANGE, ms=8, mec='white', zorder=6)
    ax.annotate('xuất phát (H)', traj[0], (traj[0][0] - 2.6, traj[0][1] - 1.2), fontsize=9,
                color=INK)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_title('"đi từ phòng thí nghiệm qua sảnh chính rồi đến hành lang"\n'
                 'lộ trình H → G → E → C', color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(IMG / 'route_demo.png', dpi=150)
    plt.close(fig)


def fig_asr():
    path = PKG / 'docs' / 'asr_eval.csv'
    if not path.exists():
        return
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    conds = list(dict.fromkeys(r['condition'] for r in rows))
    acc = {(c, m): 100 * np.mean([int(r['intent_ok']) for r in rows
                                  if r['condition'] == c and r['mode'] == m])
           for c in conds for m in ('tu do', 'grammar')}
    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for k, (mode, label, color) in enumerate((('tu do', 'Vosk tự do', ORANGE),
                                              ('grammar', 'Vosk + grammar', BLUE))):
        vals = [acc[(c, mode)] for c in conds]
        bars = ax.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.0f}%', ha='center',
                    fontsize=9, color=INK)
    names = {'sach': 'Sạch'}
    ax.set_xticks(x, [names.get(c, c.replace('SNR', 'Nhiễu SNR')) for c in conds])
    ax.set_ylabel('Lệnh hiểu đúng (%)')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f'Độ chính xác lệnh trên {len(rows) // (2 * len(conds))} câu lệnh mẫu',
                 color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(IMG / 'asr_eval.png', dpi=150)
    plt.close(fig)


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    info = load_map(PKG / 'maps' / 'map.yaml')
    sem = yaml.safe_load((PKG / 'config' / 'zone_semantics.yaml').read_text(encoding='utf-8'))
    fig_pipeline(info, sem)
    fig_zones(info)
    fig_route(info)
    fig_asr()
    print('Da ghi', sorted(p.name for p in IMG.glob('*.png')))


if __name__ == '__main__':
    main()
