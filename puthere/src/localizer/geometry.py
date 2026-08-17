SCALE = 10
STRIDE = 4
REF_PX = 1000
SEARCH_PX = 1000
REF_DS_PX = REF_PX // SCALE
HALF = REF_DS_PX // 2
REF_FEAT = REF_DS_PX // STRIDE
SEARCH_FEAT = SEARCH_PX // STRIDE
CORR = SEARCH_FEAT - REF_FEAT + 1


def cell_to_pixel(c, align_offset: float = 0.0) -> float:
    return STRIDE * float(c) + HALF + align_offset


def pixel_to_cell(x: float, align_offset: float = 0.0) -> int:
    c = int(round((float(x) - HALF - align_offset) / STRIDE))
    return max(0, min(c, CORR - 1))


def pixel_to_cell_offset(x: float, align_offset: float = 0.0):
    c = pixel_to_cell(x, align_offset)
    return c, float(x) - cell_to_pixel(c, align_offset)
