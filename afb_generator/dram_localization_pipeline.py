from __future__ import annotations

import dataclasses
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import klayout.db as db


LAYER_CAPACITOR = (1, 0)
LAYER_WORDLINE = (2, 0)
LAYER_BITLINE = (3, 0)
LAYER_CONTACT = (4, 0)
LAYER_DEFECT = (5, 0)
LAYER_DUMMY = (6, 0)
LAYER_STREET = (7, 0)


@dataclass(frozen=True)
class DRAMParams:

    rows: int = 64
    cols: int = 85
    cell_pitch_nm: float = 80.0
    cell_pitch_bl_nm: float = 60.0
    wl_width_nm: float = 24.0
    bl_width_nm: float = 18.0
    contact_size_nm: float = 14.0
    capacitor_size_nm: float = 30.0
    dummy_rows: int = 2
    dummy_cols: int = 2

    overlay_sigma_nm: float = 3.0
    linewidth_sigma_nm: float = 2.0
    ler_amplitude_nm: float = 1.5
    ler_period_nm: float = 20.0
    contact_sigma_frac: float = 0.08
    jitter_sigma_nm: float = 1.0

    p_missing_contact: float = 0.005
    p_missing_capacitor: float = 0.003
    p_broken_wl: float = 0.02
    p_broken_bl: float = 0.02
    n_particles: int = 3
    n_scratches: int = 2
    p_cmp_dishing: float = 0.10

    output_gds: str = "dram.gds"
    output_json: str = "dram_metadata.json"
    write_metadata_json: bool = True
    seed: int = 42
    dbu_nm: float = 1.0

    output_width_px: int = 1000
    output_height_px: int = 1000
    reference_width_px: int = 100
    reference_height_px: int = 100
    zoom_ratio: int = 1
    pixels_per_nm: Optional[float] = None

    die_block_rows: int = 1
    die_block_cols: int = 1
    die_block_width_nm: float = 5120.0
    die_block_height_nm: float = 5120.0
    die_street_width_nm: float = 100.0
    die_block_variation_frac: float = 0.4

    block_presets: Optional[List[str]] = None
    boundary_bias: float = 0.0

    single_training_defect: bool = False
    training_defect_size_px: int = 10


@dataclass(frozen=True)
class RasterizationConfig:
    gds_file: str
    pixels_per_nm: float
    origin_nm: Tuple[float, float]
    width_px: int
    height_px: int
    layers: List[Tuple[int, int]]
    reference_bbox_px: List[int]
    reference_center_px: List[float]


DRAM_PRESETS: dict = {
    "dram_1x": {
        "cell_pitch_nm": 80.0, "cell_pitch_bl_nm": 60.0,
        "wl_width_nm": 24.0, "bl_width_nm": 18.0,
        "contact_size_nm": 40.0, "capacitor_size_nm": 30.0,
    },
    "dram_dense": {
        "cell_pitch_nm": 60.0, "cell_pitch_bl_nm": 45.0,
        "wl_width_nm": 18.0, "bl_width_nm": 13.5,
        "contact_size_nm": 30.0, "capacitor_size_nm": 22.5,
    },
    "dram_loose": {
        "cell_pitch_nm": 120.0, "cell_pitch_bl_nm": 90.0,
        "wl_width_nm": 36.0, "bl_width_nm": 27.0,
        "contact_size_nm": 60.0, "capacitor_size_nm": 45.0,
    },
    "dram_wide": {
        "cell_pitch_nm": 150.0, "cell_pitch_bl_nm": 112.5,
        "wl_width_nm": 45.0, "bl_width_nm": 33.75,
        "contact_size_nm": 75.0, "capacitor_size_nm": 56.25,
    },
    "dram_compact": {
        "cell_pitch_nm": 90.0, "cell_pitch_bl_nm": 67.5,
        "wl_width_nm": 27.0, "bl_width_nm": 20.25,
        "contact_size_nm": 45.0, "capacitor_size_nm": 33.75,
    },
    "dram_legacy": {
        "cell_pitch_nm": 200.0, "cell_pitch_bl_nm": 150.0,
        "wl_width_nm": 60.0, "bl_width_nm": 45.0,
        "contact_size_nm": 100.0, "capacitor_size_nm": 75.0,
    },
}


def resolve_preset(name: str) -> dict:
    if name not in DRAM_PRESETS:
        raise KeyError(f"Unknown DRAM preset {name!r}; valid presets: "
                        f"{sorted(DRAM_PRESETS.keys())}")
    return dict(DRAM_PRESETS[name])


def nm_to_pixel(x_nm: float, pixels_per_nm: float) -> float:
    return x_nm * pixels_per_nm


def pixel_to_nm(x_px: float, pixels_per_nm: float) -> float:
    return x_px / pixels_per_nm


def bbox_nm_to_pixel(bbox_nm: List[float], pixels_per_nm: float) -> List[float]:
    return [v * pixels_per_nm for v in bbox_nm]


def bbox_pixel_to_nm(bbox_px: List[float], pixels_per_nm: float) -> List[float]:
    return [v / pixels_per_nm for v in bbox_px]


class ProcessVariation:

    def __init__(self, params: DRAMParams, rng: np.random.Generator) -> None:
        self.p = params
        self.rng = rng

    def overlay_shift(self, box: db.Box) -> db.Box:
        dx = int(round(self.rng.normal(0, self.p.overlay_sigma_nm)))
        dy = int(round(self.rng.normal(0, self.p.overlay_sigma_nm)))
        return box.moved(dx, dy)

    def vary_linewidth(self, box: db.Box, axis: str) -> db.Box:
        delta = int(round(self.rng.normal(0, self.p.linewidth_sigma_nm / 2)))
        if axis == 'x':
            new_bot = box.bottom + delta
            new_top = max(new_bot + 1, box.top - delta)
            return db.Box(box.left, new_bot, box.right, new_top)
        else:
            new_left = box.left + delta
            new_right = max(new_left + 1, box.right - delta)
            return db.Box(new_left, box.bottom, new_right, box.top)

    def add_ler(self, box: db.Box, axis: str,
                n_seg: Optional[int] = None) -> List[db.Box]:
        amp = max(1, int(round(self.p.ler_amplitude_nm)))
        if axis == 'x':
            total = box.right - box.left
            n = n_seg or max(4, int(total / max(1, self.p.ler_period_nm)))
            seg = max(1, total // n)
            draws = self.rng.normal(0, amp, size=(n, 2))
            boxes: List[db.Box] = []
            for k in range(n):
                x0 = box.left + k * seg
                x1 = min(box.right, x0 + seg)
                dt = int(round(draws[k, 0]))
                db_ = int(round(draws[k, 1]))
                y0 = box.bottom + db_
                y1 = max(y0 + 1, box.top + dt)
                boxes.append(db.Box(x0, y0, x1, y1))
            return boxes
        else:
            total = box.top - box.bottom
            n = n_seg or max(4, int(total / max(1, self.p.ler_period_nm)))
            seg = max(1, total // n)
            draws = self.rng.normal(0, amp, size=(n, 2))
            boxes = []
            for k in range(n):
                y0 = box.bottom + k * seg
                y1 = min(box.top, y0 + seg)
                dl = int(round(draws[k, 0]))
                dr = int(round(draws[k, 1]))
                x0 = box.left + dl
                x1 = max(x0 + 1, box.right + dr)
                boxes.append(db.Box(x0, y0, x1, y1))
            return boxes

    def vary_contact(self, box: db.Box) -> db.Box:
        cx = (box.left + box.right) // 2
        cy = (box.bottom + box.top) // 2
        side = box.width()
        scale = float(self.rng.normal(1.0, self.p.contact_sigma_frac))
        new_half = max(1, int(round(side * scale / 2)))
        return db.Box(cx - new_half, cy - new_half,
                      cx + new_half, cy + new_half)

    def jitter_cell(self, cx: float, cy: float) -> Tuple[float, float]:
        dx = float(self.rng.normal(0, self.p.jitter_sigma_nm))
        dy = float(self.rng.normal(0, self.p.jitter_sigma_nm))
        return cx + dx, cy + dy

    def is_missing(self, prob: float) -> bool:
        return bool(self.rng.random() < prob)

    def break_line(self, box: db.Box, axis: str) -> List[db.Box]:
        if axis == 'x':
            span = box.right - box.left
            gap_start = int(self.rng.integers(span // 4, span // 2))
            gap_end   = int(self.rng.integers(span // 2, 3 * span // 4))
            piece_a = db.Box(box.left,          box.bottom,
                             box.left + gap_start, box.top)
            piece_b = db.Box(box.left + gap_end, box.bottom,
                             box.right,            box.top)
        else:
            span = box.top - box.bottom
            gap_start = int(self.rng.integers(span // 4, span // 2))
            gap_end   = int(self.rng.integers(span // 2, 3 * span // 4))
            piece_a = db.Box(box.left, box.bottom,
                             box.right, box.bottom + gap_start)
            piece_b = db.Box(box.left, box.bottom + gap_end,
                             box.right, box.top)
        return [p for p in [piece_a, piece_b] if p.width() > 0 and p.height() > 0]

    def add_particles(self, bbox: db.Box, n: int) -> List[db.Box]:
        particles = []
        for _ in range(n):
            side_x = int(self.rng.integers(5, 41))
            side_y = int(self.rng.integers(5, 41))
            x0 = int(self.rng.integers(bbox.left, max(bbox.left + 1,
                                                       bbox.right - side_x)))
            y0 = int(self.rng.integers(bbox.bottom, max(bbox.bottom + 1,
                                                         bbox.top - side_y)))
            particles.append(db.Box(x0, y0, x0 + side_x, y0 + side_y))
        return particles

    def add_scratches(self, bbox: db.Box, n: int) -> List[db.Box]:
        scratches = []
        for _ in range(n):
            if self.rng.random() < 0.5:
                length = int(self.rng.integers(50, 201))
                width  = int(self.rng.integers(2, 6))
                x0 = int(self.rng.integers(bbox.left,
                                           max(bbox.left + 1, bbox.right - length)))
                y0 = int(self.rng.integers(bbox.bottom,
                                           max(bbox.bottom + 1, bbox.top - width)))
                scratches.append(db.Box(x0, y0, x0 + length, y0 + width))
            else:
                length = int(self.rng.integers(50, 201))
                width  = int(self.rng.integers(2, 6))
                x0 = int(self.rng.integers(bbox.left,
                                           max(bbox.left + 1, bbox.right - width)))
                y0 = int(self.rng.integers(bbox.bottom,
                                           max(bbox.bottom + 1, bbox.top - length)))
                scratches.append(db.Box(x0, y0, x0 + width, y0 + length))
        return scratches

    def cmp_dishing(self, boxes: List[db.Box],
                    cx: float, cy: float, r: float) -> List[db.Box]:
        result = []
        for box in boxes:
            bcx = (box.left + box.right) / 2
            bcy = (box.bottom + box.top) / 2
            if (bcx - cx) ** 2 + (bcy - cy) ** 2 < r ** 2:
                dist = ((bcx - cx) ** 2 + (bcy - cy) ** 2) ** 0.5
                factor = 1.0 - 0.2 * (1.0 - dist / r)
                if box.width() >= box.height():
                    new_h = max(1, int(round(box.height() * factor)))
                    mid_y = (box.bottom + box.top) // 2
                    result.append(db.Box(box.left, mid_y - new_h // 2,
                                         box.right, mid_y - new_h // 2 + new_h))
                else:
                    new_w = max(1, int(round(box.width() * factor)))
                    mid_x = (box.left + box.right) // 2
                    result.append(db.Box(mid_x - new_w // 2, box.bottom,
                                         mid_x - new_w // 2 + new_w, box.top))
            else:
                result.append(box)
        return result


_BLOCK_JITTER_GEOMETRY_FIELDS = [
    "cell_pitch_nm", "cell_pitch_bl_nm", "wl_width_nm", "bl_width_nm",
    "contact_size_nm", "capacitor_size_nm",
    "overlay_sigma_nm", "linewidth_sigma_nm", "ler_amplitude_nm", "ler_period_nm",
    "contact_sigma_frac", "jitter_sigma_nm",
]
_BLOCK_JITTER_PROB_FIELDS = [
    "p_missing_contact", "p_missing_capacitor", "p_broken_wl", "p_broken_bl",
    "p_cmp_dishing",
]
_BLOCK_JITTER_COUNT_FIELDS = ["n_particles", "n_scratches"]


def _randomize_block_params(base: DRAMParams, rng: np.random.Generator,
                             frac: float) -> DRAMParams:
    overrides = {}
    for field in _BLOCK_JITTER_GEOMETRY_FIELDS:
        factor = float(rng.uniform(1.0 - frac, 1.0 + frac))
        overrides[field] = max(1e-3, getattr(base, field) * factor)
    for field in _BLOCK_JITTER_PROB_FIELDS:
        factor = float(rng.uniform(1.0 - frac, 1.0 + frac))
        overrides[field] = min(1.0, max(0.0, getattr(base, field) * factor))
    for field in _BLOCK_JITTER_COUNT_FIELDS:
        factor = float(rng.uniform(1.0 - frac, 1.0 + frac))
        overrides[field] = max(0, int(round(getattr(base, field) * factor)))
    return dataclasses.replace(base, **overrides)


def _place_training_defect(cell: db.Cell, layer: int, pixels_per_nm: float,
                            rng: np.random.Generator, bbox: db.Box,
                            size_px: int) -> dict:
    side_nm = size_px / pixels_per_nm
    half = side_nm / 2.0

    margin = half
    left_lo, left_hi = bbox.left + margin, max(bbox.left + margin + 1.0, bbox.right - margin)
    bottom_lo, bottom_hi = bbox.bottom + margin, max(bbox.bottom + margin + 1.0, bbox.top - margin)
    cx = float(rng.uniform(left_lo, left_hi))
    cy = float(rng.uniform(bottom_lo, bottom_hi))
    box = db.Box(int(round(cx - half)), int(round(cy - half)),
                 int(round(cx + half)), int(round(cy + half)))
    cell.shapes(layer).insert(box)
    return {"type": "training_scratch",
            "bbox_nm": [box.left, box.bottom, box.right, box.top]}


def _canvas_bbox_nm(pixels_per_nm: float, width_px: int, height_px: int,
                     content_top_nm: float) -> db.Box:
    canvas_w_nm = width_px / pixels_per_nm
    canvas_h_nm = height_px / pixels_per_nm
    return db.Box(0, int(round(content_top_nm - canvas_h_nm)),
                  int(round(canvas_w_nm)), int(round(content_top_nm)))


class DRAMGenerator:

    def __init__(self, params: DRAMParams) -> None:
        self.p = params

        if params.output_width_px != 1000:
            raise ValueError("output_width_px must be 1000 for benchmark compatibility")
        if params.output_height_px != 1000:
            raise ValueError("output_height_px must be 1000 for benchmark compatibility")
        if params.reference_width_px <= 0:
            raise ValueError("reference_width_px must be positive")
        if params.reference_height_px <= 0:
            raise ValueError("reference_height_px must be positive")
        if params.zoom_ratio <= 0:
            raise ValueError("zoom_ratio must be positive")

        array_w_nm = params.cols * params.cell_pitch_bl_nm
        array_h_nm = params.rows * params.cell_pitch_nm
        if params.pixels_per_nm is None:
            self._pixels_per_nm: float = min(
                params.output_width_px  / array_w_nm,
                params.output_height_px / array_h_nm,
            )
        else:
            self._pixels_per_nm = params.pixels_per_nm

        self.rng = np.random.default_rng(params.seed)
        self.pv = ProcessVariation(params, self.rng)

        self.layout = db.Layout()
        self.layout.dbu = params.dbu_nm * 1e-3
        self.cell = self.layout.create_cell("DRAM_ARRAY")

        self.L_CAP = self.layout.layer(*LAYER_CAPACITOR)
        self.L_WL  = self.layout.layer(*LAYER_WORDLINE)
        self.L_BL  = self.layout.layer(*LAYER_BITLINE)
        self.L_CON = self.layout.layer(*LAYER_CONTACT)
        self.L_DEF = self.layout.layer(*LAYER_DEFECT)
        self.L_DUM = self.layout.layer(*LAYER_DUMMY)

        self._cell_centers: List[List[float]] = []
        self._wl_coords: List[dict] = []
        self._bl_coords: List[dict] = []
        self._defect_locs: List[dict] = []

        self._wl_boxes: List[db.Box] = []
        self._bl_boxes: List[db.Box] = []

        self._last_meta: dict = {}


    def _nm_to_px(self, x_nm: float) -> float:
        return nm_to_pixel(x_nm, self._pixels_per_nm)

    def _px_to_nm(self, x_px: float) -> float:
        return pixel_to_nm(x_px, self._pixels_per_nm)

    def _cell_center(self, row: int, col: int) -> Tuple[float, float]:
        cx = col * self.p.cell_pitch_bl_nm + self.p.cell_pitch_bl_nm / 2
        cy = row * self.p.cell_pitch_nm    + self.p.cell_pitch_nm    / 2
        return cx, cy

    def _array_bbox(self) -> db.Box:
        return db.Box(0, 0,
                      int(self.p.cols * self.p.cell_pitch_bl_nm),
                      int(self.p.rows * self.p.cell_pitch_nm))

    def _full_bbox(self) -> db.Box:
        x_min = int(-self.p.dummy_cols * self.p.cell_pitch_bl_nm)
        y_min = int(-self.p.dummy_rows * self.p.cell_pitch_nm)
        x_max = int((self.p.cols + self.p.dummy_cols) * self.p.cell_pitch_bl_nm)
        y_max = int((self.p.rows + self.p.dummy_rows) * self.p.cell_pitch_nm)
        return db.Box(x_min, y_min, x_max, y_max)

    def _insert(self, layer: int, boxes: Union[db.Box, List[db.Box]]) -> None:
        shapes = self.cell.shapes(layer)
        if isinstance(boxes, db.Box):
            shapes.insert(boxes)
        else:
            for b in boxes:
                if b.width() > 0 and b.height() > 0:
                    shapes.insert(b)


    def _build_dummy_cells(self) -> None:
        p = self.p
        pitch_x = int(p.cell_pitch_bl_nm)
        pitch_y = int(p.cell_pitch_nm)
        cap_half = int(p.capacitor_size_nm / 2)

        for row in range(-p.dummy_rows, p.rows + p.dummy_rows):
            for col in range(-p.dummy_cols, p.cols + p.dummy_cols):
                if 0 <= row < p.rows and 0 <= col < p.cols:
                    continue
                cx = int(col * pitch_x + pitch_x // 2)
                cy = int(row * pitch_y + pitch_y // 2)
                box = db.Box(cx - cap_half, cy - cap_half,
                             cx + cap_half, cy + cap_half)
                self._insert(self.L_DUM, box)


    def _build_capacitors(self) -> List[List[float]]:
        p = self.p
        cap_half = int(p.capacitor_size_nm / 2)
        centers = []

        for row in range(p.rows):
            for col in range(p.cols):
                nom_cx, nom_cy = self._cell_center(row, col)
                cx, cy = self.pv.jitter_cell(nom_cx, nom_cy)
                centers.append([cx, cy])

                if not p.single_training_defect and self.pv.is_missing(p.p_missing_capacitor):
                    self._defect_locs.append({
                        "type": "missing_capacitor",
                        "bbox_nm": [int(cx) - cap_half, int(cy) - cap_half,
                                    int(cx) + cap_half, int(cy) + cap_half],
                    })
                    continue

                box = db.Box(int(cx) - cap_half, int(cy) - cap_half,
                             int(cx) + cap_half, int(cy) + cap_half)
                box = self.pv.overlay_shift(box)
                self._insert(self.L_CAP, box)

        self._cell_centers = centers
        return centers

    def _build_wordlines(self) -> List[dict]:
        p = self.p
        half_wl = int(p.wl_width_nm / 2)
        x_start = int(-p.dummy_cols * p.cell_pitch_bl_nm)
        x_end   = int((p.cols + p.dummy_cols) * p.cell_pitch_bl_nm)
        coords  = []

        for row in range(p.rows):
            y_nom = int(row * p.cell_pitch_nm + p.cell_pitch_nm / 2)
            coords.append({
                "row": row,
                "y_nm": y_nom,
                "x_start_nm": x_start,
                "x_end_nm": x_end,
            })

            box = db.Box(x_start, y_nom - half_wl, x_end, y_nom + half_wl)
            box = self.pv.vary_linewidth(box, 'x')
            box = self.pv.overlay_shift(box)

            if not p.single_training_defect and self.pv.is_missing(p.p_broken_wl):
                pieces = self.pv.break_line(box, 'x')
                self._defect_locs.append({
                    "type": "broken_wl",
                    "bbox_nm": [box.left, box.bottom, box.right, box.top],
                })
                valid_pieces = [b for b in pieces if b.width() > 0 and b.height() > 0]
                self._wl_boxes.extend(valid_pieces)
                self._insert(self.L_WL, pieces)
            else:
                ler_boxes = self.pv.add_ler(box, 'x')
                valid_ler = [b for b in ler_boxes if b.width() > 0 and b.height() > 0]
                self._wl_boxes.extend(valid_ler)
                self._insert(self.L_WL, ler_boxes)

        self._wl_coords = coords
        return coords

    def _build_bitlines(self) -> List[dict]:
        p = self.p
        half_bl = int(p.bl_width_nm / 2)
        y_start = int(-p.dummy_rows * p.cell_pitch_nm)
        y_end   = int((p.rows + p.dummy_rows) * p.cell_pitch_nm)
        coords  = []

        for col in range(p.cols):
            x_nom = int(col * p.cell_pitch_bl_nm + p.cell_pitch_bl_nm / 2)
            coords.append({
                "col": col,
                "x_nm": x_nom,
                "y_start_nm": y_start,
                "y_end_nm": y_end,
            })

            box = db.Box(x_nom - half_bl, y_start, x_nom + half_bl, y_end)
            box = self.pv.vary_linewidth(box, 'y')
            box = self.pv.overlay_shift(box)

            if not p.single_training_defect and self.pv.is_missing(p.p_broken_bl):
                pieces = self.pv.break_line(box, 'y')
                self._defect_locs.append({
                    "type": "broken_bl",
                    "bbox_nm": [box.left, box.bottom, box.right, box.top],
                })
                valid_pieces = [b for b in pieces if b.width() > 0 and b.height() > 0]
                self._bl_boxes.extend(valid_pieces)
                self._insert(self.L_BL, pieces)
            else:
                ler_boxes = self.pv.add_ler(box, 'y')
                valid_ler = [b for b in ler_boxes if b.width() > 0 and b.height() > 0]
                self._bl_boxes.extend(valid_ler)
                self._insert(self.L_BL, ler_boxes)

        self._bl_coords = coords
        return coords

    def _build_contacts(self) -> List[List[float]]:
        p = self.p
        con_half  = int(p.contact_size_nm / 2)
        x_offset  = int(p.cell_pitch_bl_nm / 4)
        contact_centers: List[List[float]] = []

        for row in range(p.rows):
            for col in range(p.cols):
                nom_cx, nom_cy = self._cell_center(row, col)
                cx = nom_cx + x_offset
                cy = nom_cy
                contact_centers.append([cx, cy])

                box = db.Box(int(cx) - con_half, int(cy) - con_half,
                             int(cx) + con_half, int(cy) + con_half)
                box = self.pv.vary_contact(box)
                box = self.pv.overlay_shift(box)

                if not p.single_training_defect and self.pv.is_missing(p.p_missing_contact):
                    self._defect_locs.append({
                        "type": "missing_contact",
                        "bbox_nm": [box.left, box.bottom, box.right, box.top],
                    })
                    continue

                self._insert(self.L_CON, box)

        return contact_centers

    def _add_defects(self) -> List[dict]:
        p = self.p
        full = self._full_bbox()
        new_defects: List[dict] = []

        if not p.single_training_defect:
            for box in self.pv.add_particles(full, p.n_particles):
                self._insert(self.L_DEF, box)
                new_defects.append({"type": "particle",
                                     "bbox_nm": [box.left, box.bottom,
                                                 box.right, box.top]})

            for box in self.pv.add_scratches(full, p.n_scratches):
                self._insert(self.L_DEF, box)
                new_defects.append({"type": "scratch",
                                     "bbox_nm": [box.left, box.bottom,
                                                 box.right, box.top]})

        if p.p_cmp_dishing > 0 and not p.single_training_defect:
            arr = self._array_bbox()
            dish_cx = float((arr.left + arr.right) / 2
                            + self.rng.uniform(-arr.width() / 4, arr.width() / 4))
            dish_cy = float((arr.bottom + arr.top) / 2
                            + self.rng.uniform(-arr.height() / 4, arr.height() / 4))
            dish_r = float(min(arr.width(), arr.height()) * p.p_cmp_dishing)
            dr = int(dish_r)

            dished_wl = self.pv.cmp_dishing(self._wl_boxes, dish_cx, dish_cy, dish_r)
            self.cell.shapes(self.L_WL).clear()
            self._insert(self.L_WL, dished_wl)

            dished_bl = self.pv.cmp_dishing(self._bl_boxes, dish_cx, dish_cy, dish_r)
            self.cell.shapes(self.L_BL).clear()
            self._insert(self.L_BL, dished_bl)

            marker = db.Box(int(dish_cx) - dr, int(dish_cy) - dr,
                            int(dish_cx) + dr, int(dish_cy) + dr)
            self._insert(self.L_DEF, marker)
            new_defects.append({"type": "cmp_dishing",
                                 "bbox_nm": [marker.left, marker.bottom,
                                             marker.right, marker.top]})

        if p.single_training_defect:
            arr = self._array_bbox()
            canvas = _canvas_bbox_nm(self._pixels_per_nm, p.output_width_px, p.output_height_px,
                                      float(arr.top - arr.bottom))
            new_defects.append(_place_training_defect(
                self.cell, self.L_DEF, self._pixels_per_nm, self.rng,
                canvas, p.training_defect_size_px))

        self._defect_locs.extend(new_defects)
        return new_defects

    def generate(self) -> dict:
        self._build_dummy_cells()
        self._build_capacitors()
        self._build_wordlines()
        self._build_bitlines()
        self._build_contacts()
        self._add_defects()

        self.layout.write(self.p.output_gds)

        p = self.p
        arr = self._array_bbox()
        array_w_nm = float(arr.right - arr.left)
        array_h_nm = float(arr.top   - arr.bottom)

        ref_w_nm = p.reference_width_px  / self._pixels_per_nm
        ref_h_nm = p.reference_height_px / self._pixels_per_nm
        rx0 = float(self.rng.uniform(0, array_w_nm - ref_w_nm))
        ry0 = float(self.rng.uniform(0, array_h_nm - ref_h_nm))
        rx1 = rx0 + ref_w_nm
        ry1 = ry0 + ref_h_nm
        ref_bbox_nm   = [rx0, ry0, rx1, ry1]
        ref_center_nm = [rx0 + ref_w_nm / 2, ry0 + ref_h_nm / 2]
        ref_bbox_px   = bbox_nm_to_pixel(ref_bbox_nm, self._pixels_per_nm)
        ref_center_px = [ref_center_nm[0] * self._pixels_per_nm,
                         ref_center_nm[1] * self._pixels_per_nm]

        meta = {
            "cell_centers":    self._cell_centers,
            "wordline_coords": self._wl_coords,
            "bitline_coords":  self._bl_coords,
            "bounding_box": {
                "x_min_nm": arr.left,
                "y_min_nm": arr.bottom,
                "x_max_nm": arr.right,
                "y_max_nm": arr.top,
            },
            "defect_locations": self._defect_locs,
            "params":    dataclasses.asdict(p),
            "gds_file":  p.output_gds,
            "metadata_file": p.output_json,
            "search_image_size_px":    [p.output_width_px, p.output_height_px],
            "reference_image_size_px": [p.reference_width_px, p.reference_height_px],
            "zoom_ratio":              p.zoom_ratio,
            "pixels_per_nm":           self._pixels_per_nm,
            "search_bbox_nm":          [0.0, 0.0, array_w_nm, array_h_nm],
            "search_bbox_px":          [0.0, 0.0,
                                        float(p.output_width_px),
                                        float(p.output_height_px)],
            "reference_bbox_nm":       ref_bbox_nm,
            "reference_center_nm":     ref_center_nm,
            "reference_bbox_px":       ref_bbox_px,
            "reference_center_px":     ref_center_px,
        }

        if p.write_metadata_json:
            Path(p.output_json).write_text(json.dumps(meta, indent=2))
        self._last_meta = meta
        return meta

    def get_rasterization_config(self) -> RasterizationConfig:
        meta = self._last_meta
        layers = [LAYER_CAPACITOR, LAYER_WORDLINE, LAYER_BITLINE,
                  LAYER_CONTACT, LAYER_DEFECT, LAYER_DUMMY]
        ref_bbox_px = [int(round(v)) for v in meta["reference_bbox_px"]]
        return RasterizationConfig(
            gds_file=self.p.output_gds,
            pixels_per_nm=self._pixels_per_nm,
            origin_nm=(0.0, 0.0),
            width_px=self.p.output_width_px,
            height_px=self.p.output_height_px,
            layers=layers,
            reference_bbox_px=ref_bbox_px,
            reference_center_px=meta["reference_center_px"],
        )


def generate_dram_layout(params: DRAMParams = DRAMParams()) -> dict:
    return DRAMGenerator(params).generate()


def square_die_block_width_nm(rows: int, cols: int, street_width_nm: float,
                               block_height_nm: float) -> float:
    return (rows * block_height_nm + (rows - cols) * street_width_nm) / cols


class DieGenerator:

    def __init__(self, params: DRAMParams) -> None:
        if params.die_block_rows < 1 or params.die_block_cols < 1:
            raise ValueError("die_block_rows and die_block_cols must be >= 1")
        self.p = params
        self.rng = np.random.default_rng(params.seed)

        self.layout = db.Layout()
        self.layout.dbu = params.dbu_nm * 1e-3
        self.cell = self.layout.create_cell("DRAM_DIE")
        self.L_CAP = self.layout.layer(*LAYER_CAPACITOR)
        self.L_WL = self.layout.layer(*LAYER_WORDLINE)
        self.L_BL = self.layout.layer(*LAYER_BITLINE)
        self.L_CON = self.layout.layer(*LAYER_CONTACT)
        self.L_DEF = self.layout.layer(*LAYER_DEFECT)
        self.L_DUM = self.layout.layer(*LAYER_DUMMY)
        self.L_STREET = self.layout.layer(*LAYER_STREET)

        array_w_nm = params.die_block_cols * params.die_block_width_nm
        array_h_nm = params.die_block_rows * params.die_block_height_nm
        if params.pixels_per_nm is None:
            self._pixels_per_nm: float = min(
                params.output_width_px / array_w_nm,
                params.output_height_px / array_h_nm,
            )
        else:
            self._pixels_per_nm = params.pixels_per_nm

        self._blocks_meta: List[dict] = []
        self._cell_centers: List[List[float]] = []
        self._wl_coords: List[dict] = []
        self._bl_coords: List[dict] = []
        self._defect_locs: List[dict] = []
        self._last_meta: dict = {}

    def _total_die_size_nm(self) -> Tuple[float, float]:
        p = self.p
        w = (p.die_block_cols * p.die_block_width_nm
             + (p.die_block_cols - 1) * p.die_street_width_nm)
        h = (p.die_block_rows * p.die_block_height_nm
             + (p.die_block_rows - 1) * p.die_street_width_nm)
        return w, h

    def _block_origin_nm(self, br: int, bc: int) -> Tuple[float, float]:
        p = self.p
        x0 = bc * (p.die_block_width_nm + p.die_street_width_nm)
        y0 = br * (p.die_block_height_nm + p.die_street_width_nm)
        return x0, y0

    def _boundary_biased_crop_origin(self, ref_br: int, ref_bc: int,
                                      ref_w_nm: float, ref_h_nm: float) -> Tuple[float, float]:
        p = self.p
        total_w_nm, total_h_nm = self._total_die_size_nm()
        block_x0, block_y0 = self._block_origin_nm(ref_br, ref_bc)

        seam_choices = []
        if p.die_block_cols > 1:
            bc = ref_bc if ref_bc < p.die_block_cols - 1 else ref_bc - 1
            seam_x = ((bc + 1) * (p.die_block_width_nm + p.die_street_width_nm)
                      - p.die_street_width_nm / 2.0)
            seam_choices.append(("x", seam_x))
        if p.die_block_rows > 1:
            br = ref_br if ref_br < p.die_block_rows - 1 else ref_br - 1
            seam_y = ((br + 1) * (p.die_block_height_nm + p.die_street_width_nm)
                      - p.die_street_width_nm / 2.0)
            seam_choices.append(("y", seam_y))

        axis, seam = seam_choices[int(self.rng.integers(0, len(seam_choices)))]
        if axis == "x":
            cx = seam
            cy = (block_y0
                  + float(self.rng.uniform(0, max(0.0, p.die_block_height_nm - ref_h_nm)))
                  + ref_h_nm / 2.0)
        else:
            cy = seam
            cx = (block_x0
                  + float(self.rng.uniform(0, max(0.0, p.die_block_width_nm - ref_w_nm)))
                  + ref_w_nm / 2.0)

        rx0 = min(max(0.0, cx - ref_w_nm / 2.0), total_w_nm - ref_w_nm)
        ry0 = min(max(0.0, cy - ref_h_nm / 2.0), total_h_nm - ref_h_nm)
        return rx0, ry0

    def _build_block(self, br: int, bc: int, tmp_dir: str) -> None:
        p = self.p
        block_seed = (p.seed * 100_003 + br * p.die_block_cols + bc) % (2 ** 32 - 1)
        block_rng = np.random.default_rng(block_seed)

        block_base = p
        if p.block_presets:
            preset_name = p.block_presets[int(block_rng.integers(0, len(p.block_presets)))]
            block_base = dataclasses.replace(block_base, **resolve_preset(preset_name))

        block_params = _randomize_block_params(block_base, block_rng, p.die_block_variation_frac)

        rows = max(1, int(p.die_block_height_nm / block_params.cell_pitch_nm))
        cols = max(1, int(p.die_block_width_nm / block_params.cell_pitch_bl_nm))

        stem = f"block_{br:02d}_{bc:02d}"
        no_defects_kw = ({"n_particles": 0, "n_scratches": 0,
                           "p_cmp_dishing": 0.0, "p_missing_capacitor": 0.0,
                           "p_broken_wl": 0.0, "p_broken_bl": 0.0,
                           "p_missing_contact": 0.0,
                           "single_training_defect": False}
                          if p.single_training_defect else {})
        block_params = dataclasses.replace(
            block_params,
            rows=rows, cols=cols,
            seed=block_seed,
            pixels_per_nm=self._pixels_per_nm,
            output_gds=str(Path(tmp_dir) / f"{stem}.gds"),
            output_json=str(Path(tmp_dir) / f"{stem}.json"),
            **no_defects_kw,
        )

        block_meta = DRAMGenerator(block_params).generate()

        x0_nm, y0_nm = self._block_origin_nm(br, bc)
        dx, dy = int(round(x0_nm)), int(round(y0_nm))

        block_layout = db.Layout()
        block_layout.read(block_params.output_gds)
        block_top = block_layout.top_cell()
        layer_map = [
            (LAYER_CAPACITOR, self.L_CAP), (LAYER_WORDLINE, self.L_WL),
            (LAYER_BITLINE, self.L_BL), (LAYER_CONTACT, self.L_CON),
            (LAYER_DEFECT, self.L_DEF), (LAYER_DUMMY, self.L_DUM),
        ]
        for (ln, dt), dest_layer in layer_map:
            src_li = block_layout.find_layer(ln, dt)
            if src_li is None:
                continue
            for shape in block_top.shapes(src_li).each():
                self.cell.shapes(dest_layer).insert(shape.bbox().moved(dx, dy))

        for f in (block_params.output_gds, block_params.output_json):
            Path(f).unlink(missing_ok=True)

        self._cell_centers.extend(
            [c[0] + x0_nm, c[1] + y0_nm] for c in block_meta["cell_centers"])
        for wl in block_meta["wordline_coords"]:
            self._wl_coords.append({
                **wl,
                "y_nm": wl["y_nm"] + y0_nm,
                "x_start_nm": wl["x_start_nm"] + x0_nm,
                "x_end_nm": wl["x_end_nm"] + x0_nm,
            })
        for bl in block_meta["bitline_coords"]:
            self._bl_coords.append({
                **bl,
                "x_nm": bl["x_nm"] + x0_nm,
                "y_start_nm": bl["y_start_nm"] + y0_nm,
                "y_end_nm": bl["y_end_nm"] + y0_nm,
            })
        for d in block_meta["defect_locations"]:
            bx0, by0, bx1, by1 = d["bbox_nm"]
            self._defect_locs.append({
                **d,
                "bbox_nm": [bx0 + x0_nm, by0 + y0_nm, bx1 + x0_nm, by1 + y0_nm],
            })

        block_bbox_nm = [x0_nm, y0_nm,
                          x0_nm + p.die_block_width_nm, y0_nm + p.die_block_height_nm]
        self._blocks_meta.append({
            "block_row": br, "block_col": bc,
            "bbox_nm": block_bbox_nm,
            "params": dataclasses.asdict(block_params),
        })

    def _build_streets(self) -> None:
        p = self.p
        if p.die_street_width_nm <= 0:
            return
        total_w_nm, total_h_nm = self._total_die_size_nm()
        sw = p.die_street_width_nm
        for bc in range(1, p.die_block_cols):
            x0 = bc * (p.die_block_width_nm + sw) - sw
            box = db.Box(int(round(x0)), 0, int(round(x0 + sw)), int(round(total_h_nm)))
            self.cell.shapes(self.L_STREET).insert(box)
        for br in range(1, p.die_block_rows):
            y0 = br * (p.die_block_height_nm + sw) - sw
            box = db.Box(0, int(round(y0)), int(round(total_w_nm)), int(round(y0 + sw)))
            self.cell.shapes(self.L_STREET).insert(box)

    def generate(self) -> dict:
        p = self.p
        if p.die_block_rows == 1 and p.die_block_cols == 1:
            meta = DRAMGenerator(p).generate()
            bb = meta["bounding_box"]
            meta["blocks"] = [{
                "block_row": 0, "block_col": 0,
                "bbox_nm": [bb["x_min_nm"], bb["y_min_nm"], bb["x_max_nm"], bb["y_max_nm"]],
                "params": dataclasses.asdict(p),
            }]
            if p.write_metadata_json:
                Path(p.output_json).write_text(json.dumps(meta, indent=2))
            self._last_meta = meta
            return meta

        with tempfile.TemporaryDirectory() as tmp_dir:
            for br in range(p.die_block_rows):
                for bc in range(p.die_block_cols):
                    self._build_block(br, bc, tmp_dir)
            self._build_streets()

        total_w_nm, total_h_nm = self._total_die_size_nm()

        if p.single_training_defect:
            canvas = _canvas_bbox_nm(self._pixels_per_nm, p.output_width_px, p.output_height_px,
                                      total_h_nm)
            self._defect_locs.append(_place_training_defect(
                self.cell, self.L_DEF, self._pixels_per_nm, self.rng,
                canvas, p.training_defect_size_px))

        self.layout.write(p.output_gds)

        ref_br = int(self.rng.integers(0, p.die_block_rows))
        ref_bc = int(self.rng.integers(0, p.die_block_cols))
        block_x0, block_y0 = self._block_origin_nm(ref_br, ref_bc)

        ref_w_nm = p.reference_width_px / self._pixels_per_nm
        ref_h_nm = p.reference_height_px / self._pixels_per_nm

        take_boundary = (p.boundary_bias > 0.0
                          and (p.die_block_cols > 1 or p.die_block_rows > 1)
                          and self.rng.random() < p.boundary_bias)
        if take_boundary:
            rx0, ry0 = self._boundary_biased_crop_origin(ref_br, ref_bc, ref_w_nm, ref_h_nm)
        else:
            rx0 = block_x0 + float(self.rng.uniform(0, max(0.0, p.die_block_width_nm - ref_w_nm)))
            ry0 = block_y0 + float(self.rng.uniform(0, max(0.0, p.die_block_height_nm - ref_h_nm)))
        rx1, ry1 = rx0 + ref_w_nm, ry0 + ref_h_nm
        ref_bbox_nm = [rx0, ry0, rx1, ry1]
        ref_center_nm = [rx0 + ref_w_nm / 2, ry0 + ref_h_nm / 2]
        ref_bbox_px = bbox_nm_to_pixel(ref_bbox_nm, self._pixels_per_nm)
        ref_center_px = [ref_center_nm[0] * self._pixels_per_nm,
                          ref_center_nm[1] * self._pixels_per_nm]

        meta = {
            "cell_centers": self._cell_centers,
            "wordline_coords": self._wl_coords,
            "bitline_coords": self._bl_coords,
            "bounding_box": {
                "x_min_nm": 0, "y_min_nm": 0,
                "x_max_nm": total_w_nm, "y_max_nm": total_h_nm,
            },
            "defect_locations": self._defect_locs,
            "blocks": self._blocks_meta,
            "params": dataclasses.asdict(p),
            "gds_file": p.output_gds,
            "metadata_file": p.output_json,
            "search_image_size_px": [p.output_width_px, p.output_height_px],
            "reference_image_size_px": [p.reference_width_px, p.reference_height_px],
            "zoom_ratio": p.zoom_ratio,
            "pixels_per_nm": self._pixels_per_nm,
            "search_bbox_nm": [0.0, 0.0, total_w_nm, total_h_nm],
            "search_bbox_px": [0.0, 0.0, float(p.output_width_px), float(p.output_height_px)],
            "reference_bbox_nm": ref_bbox_nm,
            "reference_center_nm": ref_center_nm,
            "reference_bbox_px": ref_bbox_px,
            "reference_center_px": ref_center_px,
        }
        if p.write_metadata_json:
            Path(p.output_json).write_text(json.dumps(meta, indent=2))
        self._last_meta = meta
        return meta


IMG_SIZE  = 1000
SS_FACTOR = 4
SS_SIZE   = IMG_SIZE * SS_FACTOR

SEM_LAYER_ORDER = [(6, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (7, 0)]
SEM_INTENSITY = {
    (6, 0):  40,
    (1, 0): 150,
    (2, 0): 170,
    (3, 0): 190,
    (4, 0): 255,
    (5, 0): 100,
    (7, 0):  70,
}
SEM_BG = 25

DEBUG_LAYER_STYLE = [
    ((6, 0), ( 45,  50,  60)),
    ((1, 0), (220,  80,  80)),
    ((2, 0), ( 60, 150, 255)),
    ((3, 0), ( 60, 220, 130)),
    ((4, 0), (255, 200,  80)),
    ((5, 0), (190, 100, 255)),
    ((7, 0), (140, 140, 150)),
]
DEBUG_BG  = (10, 12, 18)
DEBUG_REF = (255, 255, 0)

CIRCLE_LAYERS = {(4, 0)}

DEFECT_HALO_MARGIN_NM = 15.0
DEFECT_HALO_INTENSITY = 200
DEFECT_MARK_INTENSITY = 30
DISHING_INTENSITY     = 140

POINT_DEFECT_TYPES = {"particle", "scratch", "training_scratch"}


def _supersampled_masks_for_bbox(gds_path: str, bbox_nm: list, ppn_ss: float,
                                  layer_list: list, ss_w: int, ss_h: int,
                                  _layout: db.Layout | None = None) -> dict:
    x0_nm, y0_nm, x1_nm, y1_nm = bbox_nm

    if _layout is not None:
        layout = _layout
    else:
        layout = db.Layout()
        layout.read(gds_path)
    top = layout.top_cell()

    canvas_x1 = x0_nm + ss_w / ppn_ss
    canvas_y0 = y1_nm - ss_h / ppn_ss
    query_box = db.Box(math.floor(min(x0_nm, canvas_x1)) - 1, math.floor(min(canvas_y0, y1_nm)) - 1,
                       math.ceil(max(x0_nm, canvas_x1)) + 1, math.ceil(max(canvas_y0, y1_nm)) + 1)

    masks = {}
    for ln, dt in layer_list:
        li = layout.find_layer(ln, dt)
        mask = np.zeros((ss_h, ss_w), dtype=np.uint8)
        if li is not None:
            for shape in top.shapes(li).each_touching(query_box):
                b = shape.bbox()
                px0 = (b.left  - x0_nm) * ppn_ss
                px1 = (b.right - x0_nm) * ppn_ss
                py0 = (y1_nm - b.top)    * ppn_ss
                py1 = (y1_nm - b.bottom) * ppn_ss
                ix0, iy0 = max(0, int(round(px0))), max(0, int(round(py0)))
                ix1, iy1 = min(ss_w, int(round(px1))), min(ss_h, int(round(py1)))
                if ix1 > ix0 and iy1 > iy0:
                    if (ln, dt) in CIRCLE_LAYERS:
                        cx = (px0 + px1) / 2.0
                        cy = (py0 + py1) / 2.0
                        radius = min(px1 - px0, py1 - py0) / 2.0
                        cv2.circle(mask, (int(round(cx)), int(round(cy))),
                                   max(1, int(round(radius))), 255, -1)
                    else:
                        mask[iy0:iy1, ix0:ix1] = 255
        masks[(ln, dt)] = mask > 0
    return masks


def _supersampled_layer_masks(gds_path: str, meta: dict, layer_list: list,
                               _layout: db.Layout | None = None) -> dict:
    ppn_ss = meta["pixels_per_nm"] * SS_FACTOR
    array_w_nm = meta["search_bbox_nm"][2]
    array_h_nm = meta["search_bbox_nm"][3]
    bbox_nm = [0.0, 0.0, array_w_nm, array_h_nm]
    return _supersampled_masks_for_bbox(gds_path, bbox_nm, ppn_ss, layer_list, SS_SIZE, SS_SIZE,
                                        _layout=_layout)


def _nm_bbox_to_ss_px(bbox_nm: list, origin_bbox_nm: list, ppn_ss: float) -> tuple:
    x0_nm, y0_nm, x1_nm, y1_nm = origin_bbox_nm
    bx0, by0, bx1, by1 = bbox_nm
    px0 = (bx0 - x0_nm) * ppn_ss
    px1 = (bx1 - x0_nm) * ppn_ss
    py0 = (y1_nm - by1) * ppn_ss
    py1 = (y1_nm - by0) * ppn_ss
    return px0, py0, px1, py1


def _paint_defects(intensity: np.ndarray, defect_locations: list,
                    origin_bbox_nm: list, ppn_ss: float) -> None:
    ss_h, ss_w = intensity.shape[:2]
    for defect in defect_locations:
        if defect["type"] not in POINT_DEFECT_TYPES and defect["type"] != "cmp_dishing":
            continue

        bbox_nm = defect["bbox_nm"]

        if defect["type"] == "cmp_dishing":
            px0, py0, px1, py1 = _nm_bbox_to_ss_px(bbox_nm, origin_bbox_nm, ppn_ss)
            ix0, iy0 = max(0, int(round(px0))), max(0, int(round(py0)))
            ix1, iy1 = min(ss_w, int(round(px1))), min(ss_h, int(round(py1)))
            if ix1 > ix0 and iy1 > iy0:
                region = intensity[iy0:iy1, ix0:ix1]
                intensity[iy0:iy1, ix0:ix1] = 0.5 * region + 0.5 * DISHING_INTENSITY
            continue

        if defect["type"] == "scratch":
            px0, py0, px1, py1 = _nm_bbox_to_ss_px(bbox_nm, origin_bbox_nm, ppn_ss)
            w, h = px1 - px0, py1 - py0
            thickness = max(1, int(round(min(w, h))))
            if w >= h:
                pt1 = (px0, (py0 + py1) / 2.0)
                pt2 = (px1, (py0 + py1) / 2.0)
            else:
                pt1 = ((px0 + px1) / 2.0, py0)
                pt2 = ((px0 + px1) / 2.0, py1)
            cv2.line(intensity,
                      (int(round(pt1[0])), int(round(pt1[1]))),
                      (int(round(pt2[0])), int(round(pt2[1]))),
                      DEFECT_MARK_INTENSITY, thickness, cv2.LINE_AA)
            continue

        if defect["type"] == "training_scratch":
            px0, py0, px1, py1 = _nm_bbox_to_ss_px(bbox_nm, origin_bbox_nm, ppn_ss)
            thickness = max(1, int(round(min(px1 - px0, py1 - py0) * 0.15)))
            cv2.line(intensity,
                      (int(round(px0)), int(round(py0))),
                      (int(round(px1)), int(round(py1))),
                      DEFECT_MARK_INTENSITY, thickness, cv2.LINE_AA)
            continue

        bx0, by0, bx1, by1 = bbox_nm
        halo_nm = [bx0 - DEFECT_HALO_MARGIN_NM, by0 - DEFECT_HALO_MARGIN_NM,
                   bx1 + DEFECT_HALO_MARGIN_NM, by1 + DEFECT_HALO_MARGIN_NM]

        hpx0, hpy0, hpx1, hpy1 = _nm_bbox_to_ss_px(halo_nm, origin_bbox_nm, ppn_ss)
        ix0, iy0 = max(0, int(round(hpx0))), max(0, int(round(hpy0)))
        ix1, iy1 = min(ss_w, int(round(hpx1))), min(ss_h, int(round(hpy1)))
        if ix1 > ix0 and iy1 > iy0:
            intensity[iy0:iy1, ix0:ix1] = DEFECT_HALO_INTENSITY

        mpx0, mpy0, mpx1, mpy1 = _nm_bbox_to_ss_px(bbox_nm, origin_bbox_nm, ppn_ss)
        ix0, iy0 = max(0, int(round(mpx0))), max(0, int(round(mpy0)))
        ix1, iy1 = min(ss_w, int(round(mpx1))), min(ss_h, int(round(mpy1)))
        if ix1 > ix0 and iy1 > iy0:
            intensity[iy0:iy1, ix0:ix1] = DEFECT_MARK_INTENSITY


def _downsample(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def render_sem_image(
    gds_path: str,
    meta: dict,
    rng: np.random.Generator,
    edge_gain: float = 0.35,
    _layout: db.Layout | None = None,
) -> np.ndarray:
    masks = _supersampled_layer_masks(gds_path, meta, SEM_LAYER_ORDER, _layout=_layout)

    intensity = np.full((SS_SIZE, SS_SIZE), SEM_BG, dtype=np.float32)
    for layer in SEM_LAYER_ORDER:
        m = masks[layer]
        intensity[m] = SEM_INTENSITY[layer]

    array_w_nm = meta["search_bbox_nm"][2]
    array_h_nm = meta["search_bbox_nm"][3]
    ppn_ss = meta["pixels_per_nm"] * SS_FACTOR
    _paint_defects(intensity, meta.get("defect_locations", []),
                   [0.0, 0.0, array_w_nm, array_h_nm], ppn_ss)

    intensity /= 255.0

    gx = cv2.Sobel(intensity, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(intensity, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_mag = edge_mag / (edge_mag.max() + 1e-6)
    intensity = np.clip(intensity + edge_gain * edge_mag, 0, 1)

    img = _downsample(intensity)

    return (img.clip(0, 1) * 255).astype(np.uint8)


def render_sem_patch(
    gds_path: str,
    bbox_nm: list,
    ppn: float,
    width_px: int,
    height_px: int,
    rng: np.random.Generator,
    defect_locations: list = None,
    edge_gain: float = 0.35,
    _layout: db.Layout | None = None,
) -> np.ndarray:
    ss_w = width_px * SS_FACTOR
    ss_h = height_px * SS_FACTOR
    ppn_ss = ppn * SS_FACTOR

    masks = _supersampled_masks_for_bbox(gds_path, bbox_nm, ppn_ss, SEM_LAYER_ORDER, ss_w, ss_h,
                                         _layout=_layout)

    intensity = np.full((ss_h, ss_w), SEM_BG, dtype=np.float32)
    for layer in SEM_LAYER_ORDER:
        m = masks[layer]
        intensity[m] = SEM_INTENSITY[layer]

    _paint_defects(intensity, defect_locations or [], bbox_nm, ppn_ss)

    intensity /= 255.0

    gx = cv2.Sobel(intensity, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(intensity, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_mag = edge_mag / (edge_mag.max() + 1e-6)
    intensity = np.clip(intensity + edge_gain * edge_mag, 0, 1)

    img = cv2.resize(intensity, (width_px, height_px), interpolation=cv2.INTER_AREA)

    return (img.clip(0, 1) * 255).astype(np.uint8)


def render_debug_image(gds_path: str, meta: dict) -> np.ndarray:
    layer_list = [ln for ln, _ in DEBUG_LAYER_STYLE]
    masks = _supersampled_layer_masks(gds_path, meta, layer_list)

    canvas_ss = np.empty((SS_SIZE, SS_SIZE, 3), dtype=np.float32)
    canvas_ss[:] = DEBUG_BG
    for layer, color in DEBUG_LAYER_STYLE:
        m = masks[layer]
        canvas_ss[m] = color

    canvas = _downsample(canvas_ss).astype(np.uint8)

    x0_nm, y0_nm, x1_nm, y1_nm = meta["reference_bbox_nm"]
    ppn = meta["pixels_per_nm"]
    content_h_px = meta["search_bbox_nm"][3] * ppn

    rx0 = int(round(x0_nm * ppn))
    rx1 = int(round(x1_nm * ppn))
    ry0 = int(round(content_h_px - y1_nm * ppn))
    ry1 = int(round(content_h_px - y0_nm * ppn))
    cx = int(round(meta["reference_center_nm"][0] * ppn))
    cy = int(round(content_h_px - meta["reference_center_nm"][1] * ppn))
    cv2.rectangle(canvas, (rx0, ry0), (rx1, ry1), DEBUG_REF, 2)
    cv2.circle(canvas, (cx, cy), 4, DEBUG_REF, -1)
    return canvas


DEFAULT_PARAMS_KW = dict(
    rows=64, cols=85,
    cell_pitch_nm=80.0, cell_pitch_bl_nm=60.0,
    wl_width_nm=24.0, bl_width_nm=18.0,
    contact_size_nm=40.0, capacitor_size_nm=30.0,
    n_particles=3, n_scratches=2,
    p_broken_wl=0.02, p_broken_bl=0.02, p_cmp_dishing=0.0,
    overlay_sigma_nm=6.0, linewidth_sigma_nm=4.0, ler_amplitude_nm=3.0,
    contact_sigma_frac=0.15, jitter_sigma_nm=2.5,
)

@dataclass
class ReferenceSearchSample:
    reference_img: np.ndarray
    search_img: np.ndarray
    true_center_px: Tuple[float, float]
    zoom_ratio: float
    seed: int
    defect_bbox_px: Optional[Tuple[float, float, float, float]] = None
    training_defect_center_px: Optional[Tuple[float, float]] = None
    training_defect_bbox_px: Optional[Tuple[float, float, float, float]] = None
    training_defect_tile: Optional[Tuple[int, int]] = None


def _find_reference_defect(meta: dict) -> Optional[dict]:
    rx0, ry0, rx1, ry1 = meta["reference_bbox_nm"]
    for d in meta["defect_locations"]:
        if d["type"] not in ("particle", "scratch"):
            continue
        bx0, by0, bx1, by1 = d["bbox_nm"]
        if bx1 > rx0 and bx0 < rx1 and by1 > ry0 and by0 < ry1:
            return d
    return None


def _defect_bbox_to_search_px(bbox_nm: list, pixels_per_nm: float,
                               content_h_px: float) -> tuple:
    x0, y0, x1, y1 = bbox_nm
    xs = [x0 * pixels_per_nm, x1 * pixels_per_nm]
    ys = [content_h_px - y0 * pixels_per_nm, content_h_px - y1 * pixels_per_nm]
    return (min(xs), min(ys), max(xs), max(ys))


def generate_sample(seed: int, tmp_dir: str, **params_kw) -> ReferenceSearchSample:
    kw = {**DEFAULT_PARAMS_KW, **params_kw}
    stem = f"pair_{seed:06d}"
    gds_path  = str(Path(tmp_dir) / f"{stem}.gds")
    json_path = str(Path(tmp_dir) / f"{stem}_meta.json")

    p = DRAMParams(**kw, seed=seed, output_gds=gds_path, output_json=json_path,
                   write_metadata_json=False)
    gen = DieGenerator(p)
    meta = gen.generate()

    layout = db.Layout()
    layout.read(gds_path)

    seed_seq = np.random.SeedSequence(seed)
    search_seed, ref_seed = seed_seq.spawn(2)
    rng_search = np.random.default_rng(search_seed)
    rng_ref = np.random.default_rng(ref_seed)

    search_img = render_sem_image(gds_path, meta, rng_search, _layout=layout)

    raw_cx, raw_cy = meta["reference_center_px"]
    content_h_px = meta["search_bbox_nm"][3] * meta["pixels_per_nm"]
    true_center_px = (raw_cx, content_h_px - raw_cy)

    zoom_ratio = float(meta["zoom_ratio"])
    ref_w_px = int(round(meta["reference_image_size_px"][0] * zoom_ratio))
    ref_h_px = int(round(meta["reference_image_size_px"][1] * zoom_ratio))
    ppn_hi = meta["pixels_per_nm"] * zoom_ratio

    reference_img = render_sem_patch(gds_path, meta["reference_bbox_nm"], ppn_hi,
                                      ref_w_px, ref_h_px, rng_ref,
                                      defect_locations=meta["defect_locations"],
                                      _layout=layout)

    defect = _find_reference_defect(meta)
    defect_bbox_px = None
    if defect is not None:
        defect_bbox_px = _defect_bbox_to_search_px(
            defect["bbox_nm"], meta["pixels_per_nm"], content_h_px)

    training_defect_center_px = None
    training_defect_bbox_px = None
    training_defect_tile = None
    training_defect = next(
        (d for d in meta["defect_locations"] if d["type"] == "training_scratch"), None)
    if training_defect is not None:
        tx0, ty0, tx1, ty1 = _defect_bbox_to_search_px(
            training_defect["bbox_nm"], meta["pixels_per_nm"], content_h_px)
        training_defect_bbox_px = (tx0, ty0, tx1, ty1)
        tcx, tcy = (tx0 + tx1) / 2, (ty0 + ty1) / 2
        training_defect_center_px = (tcx, tcy)
        training_defect_tile = (min(9, max(0, int(tcy // 100))),
                                 min(9, max(0, int(tcx // 100))))

    for f in (gds_path, json_path):
        Path(f).unlink(missing_ok=True)

    return ReferenceSearchSample(
        reference_img=reference_img,
        search_img=search_img,
        true_center_px=true_center_px,
        zoom_ratio=zoom_ratio,
        seed=seed,
        defect_bbox_px=defect_bbox_px,
        training_defect_center_px=training_defect_center_px,
        training_defect_bbox_px=training_defect_bbox_px,
        training_defect_tile=training_defect_tile,
    )
