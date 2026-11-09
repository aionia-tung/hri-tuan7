"""Nap thong tin khu vuc (zones.yaml do segment_map sinh ra) va tra cuu diem -> khu vuc."""

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yaml


@dataclass
class Zone:
    id: int
    letter: str
    name: str
    aliases: list = field(default_factory=list)
    center: tuple = (0.0, 0.0)
    goal: tuple = (0.0, 0.0, 0.0)     # x, y, yaw[do]
    area_m2: float = 0.0
    polygon: list = field(default_factory=list)

    @property
    def label(self):
        return f'{self.letter} - {self.name}'


class ZoneMap:
    def __init__(self, zones, label_image=None, resolution=0.05, origin=(0.0, 0.0)):
        self.zones = list(zones)
        self.by_letter = {z.letter.upper(): z for z in self.zones}
        self.by_id = {z.id: z for z in self.zones}
        self.labels = label_image
        self.resolution = resolution
        self.origin = origin

    @classmethod
    def load(cls, path):
        path = Path(path)
        doc = yaml.safe_load(path.read_text(encoding='utf-8'))
        zones = [Zone(id=z['id'], letter=z['letter'], name=z['name'],
                      aliases=z.get('aliases', []), center=tuple(z['center']),
                      goal=tuple(z['goal']), area_m2=z.get('area_m2', 0.0),
                      polygon=z.get('polygon', [])) for z in doc['zones']]
        labels = None
        if doc.get('label_image'):
            labels = cv2.imread(str(path.parent / doc['label_image']), cv2.IMREAD_UNCHANGED)
        return cls(zones, labels, doc.get('resolution', 0.05), tuple(doc.get('origin', (0, 0))[:2]))

    def zone_at(self, x, y):
        """Khu vuc chua diem (x, y) frame map, None neu nam ngoai moi khu vuc."""
        if self.labels is not None:
            h, w = self.labels.shape[:2]
            col = int(math.floor((x - self.origin[0]) / self.resolution))
            row = h - 1 - int(math.floor((y - self.origin[1]) / self.resolution))
            if 0 <= row < h and 0 <= col < w:
                return self.by_id.get(int(self.labels[row, col]))
            return None
        for z in self.zones:   # khong co anh nhan -> kiem tra da giac
            if _inside(z.polygon, x, y):
                return z
        return None


def _inside(poly, x, y):
    """Ray casting point-in-polygon."""
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside
