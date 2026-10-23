"""Eight-parameter geometries. p = 0.5 is the nominal device.

Both renderers return a design variable in [0, 1], the array CMA-ES and the
length-scale check hand to the ceviche model. They do not run the density filter.
"""

from __future__ import annotations

import numpy as np


def waveguide_px(protocol: dict, device: str) -> int:
    width = int(protocol["devices"][device]["wg_width_nm"])
    resolution = int(protocol["resolution_nm"])
    if width % resolution != 0:
        raise ValueError("waveguide width is not a whole number of cells")
    return width // resolution


def render(device: str, model, params, wg_px: int) -> np.ndarray:
    vector = np.clip(np.asarray(params, dtype=float).reshape(-1), 0.0, 1.0)
    if vector.size != 8:
        raise ValueError(f"parametric dimension is 8, got {vector.size}")
    if device == "waveguide_bend":
        return bend_density(model, vector, wg_px)
    if device == "beam_splitter":
        return splitter_density(model, vector)
    raise ValueError(f"unknown device: {device}")


def _rint(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _ribbon(shape: tuple[int, int], points: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Pixels whose center lies within the local half-width of the polyline."""
    nx, ny = shape
    centers = np.stack(
        np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    start = points[:-1]
    stop = points[1:]
    segment = stop - start
    length2 = np.maximum(np.sum(segment * segment, axis=1), 1e-12)
    offset = centers[:, None, :] - start[None, :, :]
    along = np.clip(np.sum(offset * segment[None, :, :], axis=2) / length2, 0.0, 1.0)
    closest = start[None, :, :] + along[..., None] * segment[None, :, :]
    distance = np.linalg.norm(centers[:, None, :] - closest, axis=2)
    nearest = np.argmin(distance, axis=1)
    index = np.arange(centers.shape[0])
    half = 0.5 * (
        (1.0 - along[index, nearest]) * widths[:-1][nearest]
        + along[index, nearest] * widths[1:][nearest]
    )
    return (distance[index, nearest] <= half).reshape(nx, ny)


def _boundary_supports(model):
    """Waveguide pixels just outside the design region, in design-local coordinates."""
    x0, y0, x1, y1 = (int(v) for v in model.design_region_coords)
    solid = np.asarray(model.density_bg) >= 0.5
    if x0 <= 0 or y0 <= 0:
        raise RuntimeError("design region has no access-waveguide pixel to read")
    left = solid[x0 - 1, y0:y1]
    bottom = solid[x0:x1, y0 - 1]
    return left, bottom


def bend_density(model, params: np.ndarray, wg_px: int) -> np.ndarray:
    """Quarter-turn from the left port to the bottom port.

    The circle is centered on the bottom-left corner of the design region.
    Endpoints stay on the ports: the radial bow and the bulge are zero there.
    Port pads use the specification waveguide width, so the parameters cannot
    delete the ports.
    """
    nx, ny = (int(v) for v in model.design_variable_shape)
    nominal = 0.5 * float(min(nx, ny))
    radius = nominal * (0.6 + 0.8 * float(params[0]))
    width0 = float(wg_px) * (0.7 + 0.6 * float(params[1]))
    center_x = (float(params[2]) - 0.5) * 0.4 * nominal
    center_y = (float(params[3]) - 0.5) * 0.4 * nominal
    gamma = 0.4 + 1.2 * float(params[4])
    radial = (float(params[5]) - 0.5) * 0.4 * nominal
    taper = 0.6 + 0.8 * float(params[6])
    bulge = (float(params[7]) - 0.5) * 0.3

    count = max(32, 8 * max(nx, ny))
    along = np.linspace(0.0, 1.0, count)
    # gamma = 1 spreads the samples uniformly in angle.
    angle = (0.5 * np.pi) * (1.0 - along**gamma)
    curve_radius = (
        radius
        + radial * np.sin(np.pi * along)
        + radius * bulge * np.sin(2.0 * np.pi * along)
    )
    points = np.stack(
        [
            center_x + curve_radius * np.cos(angle),
            center_y + curve_radius * np.sin(angle),
        ],
        axis=1,
    )
    widths = width0 * ((1.0 - along) + taper * along)
    solid = _ribbon((nx, ny), points, widths)

    left, bottom = _boundary_supports(model)
    if left.shape != (ny,) or bottom.shape != (nx,):
        raise RuntimeError(
            f"port supports {left.shape}, {bottom.shape} do not match {(nx, ny)}"
        )
    depth = int(wg_px)
    solid[:depth, left] = True
    solid[bottom, :depth] = True
    return solid.astype(float)


def splitter_density(model, params: np.ndarray) -> np.ndarray:
    """Low-x, low-y quadrant. The model mirrors it across both axes.

    The background waveguide is copied from density_bg, because inlaying a
    design variable replaces that region. p = 0.5 reaches the mirror and has
    no hole. A hole exists only for p5 > 0.5, and it cannot erase the port guide.
    """
    nx, ny = (int(v) for v in model.design_variable_shape)
    x0, y0, _, _ = (int(v) for v in model.design_region_coords)
    background = np.asarray(model.density_bg) >= 0.5
    guide = background[x0 : x0 + nx, y0 : y0 + ny]
    if guide.shape != (nx, ny):
        raise RuntimeError(f"quadrant crop {guide.shape} != {(nx, ny)}")
    if not np.any(guide):
        raise RuntimeError("splitter quadrant contains no waveguide")

    design = np.zeros((nx, ny), dtype=float)
    design[guide] = 1.0
    guide_rows = np.where(guide.any(axis=0))[0]
    guide_top = int(guide_rows[-1]) + 1

    extra = int(np.floor(float(params[3]) * 4.0))
    if extra > 0 and guide_top < ny:
        design[:, guide_top : min(ny, guide_top + extra)] = 1.0
        guide_top = min(ny, guide_top + extra)

    length = max(1, _rint((0.2 + 0.6 * float(params[0])) * (0.55 + 0.45 * float(params[4])) * nx))
    center = _rint(float(params[2]) * (nx - 1))
    x_start = min(max(0, center - length // 2), nx - 1)
    x_stop = min(nx, max(x_start + 1, x_start + length))
    gap = ny - guide_top
    reach = _rint(min(1.0, 2.0 * float(params[1])) * gap)
    if reach > 0:
        design[x_start:x_stop, guide_top : guide_top + reach] = 1.0

    if float(params[5]) > 0.5 and reach > 0:
        radius = 1.0 + float(params[7]) * 1.5
        hole_x = x_start + float(params[6]) * max(0, x_stop - x_start - 1)
        hole_y = guide_top + 0.5 * max(reach - 1, 0)
        xs = np.arange(nx)[:, None]
        ys = np.arange(ny)[None, :]
        hole = (xs - hole_x) ** 2 + (ys - hole_y) ** 2 <= radius**2
        design[hole] = 0.0
        design[guide] = 1.0
    return design
