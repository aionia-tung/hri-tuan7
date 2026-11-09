"""Phan vung ban do Occupancy Grid 2D (map.pgm) thanh cac khu vuc (phong, hanh lang).

Pipeline (OpenCV):
  1. Threshold anh .pgm theo free_thresh/occupied_thresh cua map.yaml -> mat na free.
  2. Morphology opening bo nhieu le te, closing lap cac lo nho trong vung free.
  3. Distance transform: moi pixel free = khoang cach toi vat can gan nhat.
  4. Marker = cac thanh phan lien thong cua {dt > core_dist} (loi rong cua phong/hanh lang,
     cac "cua" hep bi cat dut) -> watershed (scikit-image) tren -dt lan marker ra toan bo
     vung free (mask = free nen nhan khong tran qua tuong).
  5. Gop vung: hai vung ke nhau duoc gop neu bien chung "rong" (dt tai yen ngua gan bang
     dt dinh cua vung nho hon -> khong co cua that), hoac vung qua nho.
  6. Moi vung: tam (centroid), diem dich hop le = pixel gan tam nhat co do thong thoang
     >= goal_clearance, nam trong phan free lien thong voi vi tri robot va khong trung cac
     vat the da biet (avoid).
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from skimage.segmentation import watershed


@dataclass
class MapInfo:
    image: np.ndarray          # anh xam goc (0 = den = vat can)
    resolution: float
    origin: tuple              # (x, y, yaw) cua goc duoi-trai anh trong frame map
    free_thresh: float
    occupied_thresh: float
    negate: int = 0

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def width(self):
        return self.image.shape[1]

    def pixel_to_world(self, col, row):
        """Tam pixel (col, row) -> (x, y) trong frame map. Hang 0 cua anh la mep TREN."""
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (self.height - row - 0.5) * self.resolution
        return float(x), float(y)

    def world_to_pixel(self, x, y):
        col = int(math.floor((x - self.origin[0]) / self.resolution))
        row = self.height - 1 - int(math.floor((y - self.origin[1]) / self.resolution))
        return col, row

    def occupancy(self):
        """Xac suat chiem cho [0, 1] giong map_server (trinary)."""
        p = self.image.astype(np.float32) / 255.0
        return p if self.negate else 1.0 - p


def load_map(yaml_path):
    yaml_path = Path(yaml_path)
    meta = yaml.safe_load(yaml_path.read_text())
    image_path = Path(meta['image'])
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'Khong doc duoc anh ban do {image_path}')
    return MapInfo(image=image, resolution=float(meta['resolution']),
                   origin=tuple(meta['origin']), free_thresh=float(meta['free_thresh']),
                   occupied_thresh=float(meta['occupied_thresh']),
                   negate=int(meta.get('negate', 0)))


@dataclass
class SegmentParams:
    core_dist: float = 1.1        # m, dt > core_dist la loi cua mot khu vuc (marker)
    merge_ratio: float = 0.75     # gop neu dt_yen_ngua >= ratio * min(dt_dinh)
    min_area: float = 9.0         # m^2, vung nho hon bi gop vao lang gieng
    goal_clearance: float = 0.5   # m, khoang cach toi thieu tu diem dich den vat can
    goal_open: float = 1.0        # m, do thong thoang mong muon cua diem dich
    open_px: int = 3              # kernel morphology
    avoid: list = field(default_factory=list)   # [[x, y, r], ...] vat the da biet (nguoi, ban)


@dataclass
class Region:
    label: int
    area: float = 0.0
    centroid: tuple = (0.0, 0.0)          # (x, y) frame map
    goal: tuple = (0.0, 0.0)              # (x, y) diem dich hop le
    clearance: float = 0.0                # m, dt tai diem dich
    polygon: list = field(default_factory=list)   # [[x, y], ...] duong bao ngoai
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)    # xmin, ymin, xmax, ymax
    reachable: bool = True                # diem dich lien thong voi vi tri robot


def free_mask(info, params):
    occ = info.occupancy()
    free = (occ < info.free_thresh).astype(np.uint8)
    k = np.ones((params.open_px, params.open_px), np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, k)
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, k)
    # closing co the lap ca pixel vat can that -> giu vat can
    free[occ > info.occupied_thresh] = 0
    return free


def _adjacency(labels):
    """Cac cap (a, b) ke nhau 4-lien thong va danh sach pixel bien cua tung cap."""
    pairs = {}
    for dr, dc in ((0, 1), (1, 0)):
        a = labels[:labels.shape[0] - dr, :labels.shape[1] - dc]
        b = labels[dr:, dc:]
        mask = (a > 0) & (b > 0) & (a != b)
        rows, cols = np.nonzero(mask)
        for r, c, la, lb in zip(rows, cols, a[mask], b[mask]):
            key = (min(la, lb), max(la, lb))
            pairs.setdefault(key, []).append((r, c))
    return pairs


def _merge_regions(labels, dt, params, px_area):
    """Gop lap tung cap vung theo tieu chi cua rong / vung nho cho den khi khong con cap nao."""
    while True:
        ids = [i for i in np.unique(labels) if i > 0]
        peak = {i: float(dt[labels == i].max()) for i in ids}
        area = {i: float((labels == i).sum()) * px_area for i in ids}
        best = None
        for (a, b), pix in _adjacency(labels).items():
            rr, cc = zip(*pix)
            saddle = float(dt[rr, cc].max())
            ratio = saddle / max(min(peak[a], peak[b]), 1e-6)
            small = min(area[a], area[b]) < params.min_area
            if ratio >= params.merge_ratio or small:
                # uu tien vung nho nhat (gop vao lang gieng thong rong nhat), sau do cap
                # co ratio lon nhat
                score = (0, min(area[a], area[b]), -ratio) if small else (1, 0.0, -ratio)
                if best is None or score < best[0]:
                    best = (score, a, b)
        if best is None:
            return labels
        _, a, b = best
        labels[labels == b] = a


def segment(info, params=None, robot_xy=(0.0, 0.0)):
    """Tra ve (labels HxW int32, list[Region], dt[m]). label 0 = khong phai free."""
    params = params or SegmentParams()
    res = info.resolution
    free = free_mask(info, params)
    dt_px = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    dt = dt_px * res

    core = (dt > params.core_dist).astype(np.uint8)
    n, markers = cv2.connectedComponents(core)
    markers = markers.astype(np.int32)
    # watershed tren anh "do cao" = -dt (day thung lung o giua phong), chi lan trong vung free
    labels = watershed(-dt, markers, mask=free.astype(bool)).astype(np.int32)

    labels = _merge_regions(labels, dt, params, res * res)

    # Phan free robot di toi duoc (da tru ban kinh an toan) lien thong voi robot
    safe = (dt >= params.goal_clearance).astype(np.uint8)
    _, comp = cv2.connectedComponents(safe)
    rc, rr = info.world_to_pixel(*robot_xy)
    reach_id = comp[rr, rc] if safe[rr, rc] else np.bincount(comp[safe > 0]).argmax()
    reachable = comp == reach_id
    # Khong dat diem dich len vat the co trong world nhung khong co trong ban do tinh
    rows_all, cols_all = np.indices(labels.shape)
    for ax, ay, ar in params.avoid:
        ac, arow = info.world_to_pixel(ax, ay)
        reachable &= (rows_all - arow) ** 2 + (cols_all - ac) ** 2 > (ar / res) ** 2

    # danh so lai 1..N
    ids = np.unique(labels)
    lut = np.zeros(ids.max() + 1, np.int32)
    lut[ids[ids > 0]] = np.arange(1, int((ids > 0).sum()) + 1)
    labels = lut[labels]

    regions = []
    for lab in range(1, labels.max() + 1):
        mask = labels == lab
        rows, cols = np.nonzero(mask)
        cy, cx = rows.mean(), cols.mean()
        # Diem dich: pixel gan tam nhat, du thong thoang (toi da goal_open m hoac 80% dinh
        # cua vung) va robot di toi duoc. Tam hinh hoc co the roi vao vat can (vung chu L).
        target = max(params.goal_clearance, min(params.goal_open, 0.8 * float(dt[mask].max())))
        cand = mask & reachable & (dt >= target)
        ok = bool(cand.any())
        if not ok:
            cand = mask & (dt == dt[mask].max())
        cr, cc = np.nonzero(cand)
        k = int(np.argmin((cr - cy) ** 2 + (cc - cx) ** 2))
        gx, gy = info.pixel_to_world(cc[k], cr[k])
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        contour = cv2.approxPolyDP(contour, 1.5, True)[:, 0, :]
        poly = [list(map(lambda v: round(v, 3), info.pixel_to_world(c, r))) for c, r in contour]
        x0, y1 = info.pixel_to_world(cols.min(), rows.min())
        x1, y0 = info.pixel_to_world(cols.max(), rows.max())
        regions.append(Region(label=lab, area=float(mask.sum()) * res * res,
                              centroid=info.pixel_to_world(cx, cy), goal=(gx, gy),
                              clearance=float(dt[cr[k], cc[k]]), polygon=poly,
                              bbox=(x0, y0, x1, y1), reachable=ok))
    return labels, regions, dt
