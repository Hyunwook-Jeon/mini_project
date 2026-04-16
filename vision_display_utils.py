from __future__ import annotations

import math

import cv2
import numpy as np


def _normalize_vec_quadrant(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return (v / n).astype(np.float64)


def quadrant_obj_frame_from_normal(
    normal: np.ndarray,
    u: float,
    v: float,
    cx: float,
    cy: float,
) -> dict:
    """표면 법선 Z와 (u,v)의 주점 대비 4분면 부호로 접평면 X,Y 단위축을 고른다."""
    z_axis = _normalize_vec_quadrant(np.asarray(normal, dtype=np.float64))
    if z_axis is None:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    t1 = np.cross(ref, z_axis)
    t1 = _normalize_vec_quadrant(t1)
    if t1 is None:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        t1 = _normalize_vec_quadrant(np.cross(ref, z_axis))
    if t1 is None:
        t1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t2 = _normalize_vec_quadrant(np.cross(z_axis, t1))
    if t2 is None:
        t2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    sx = 1.0 if u >= cx else -1.0
    sy = 1.0 if v >= cy else -1.0
    x_axis = _normalize_vec_quadrant((sx * t1) + (sy * t2))
    if x_axis is None:
        x_axis = t1
    y_axis = _normalize_vec_quadrant(np.cross(z_axis, x_axis))
    if y_axis is None:
        y_axis = t2

    return {
        'source': 'quadrant_fallback',
        'x_axis_mode': 'quadrant_sign',
        'axis_x': [float(x_axis[0]), float(x_axis[1]), float(x_axis[2])],
        'axis_y': [float(y_axis[0]), float(y_axis[1]), float(y_axis[2])],
        'axis_z': [float(z_axis[0]), float(z_axis[1]), float(z_axis[2])],
        'std_minor_m': None,
        'std_major_m': None,
    }


def centroid_cam_from_uv_depth(
    ux: float,
    uy: float,
    z_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """(u,v)+깊이 → project 카메라 좌표 3D 점 (축 원점)."""
    if not math.isfinite(z_m) or math.isnan(z_m) or z_m <= 0:
        return np.zeros(3, dtype=np.float64)
    x_opt = ((float(ux) - cx) / fx) * z_m
    y_opt = ((float(uy) - cy) / fy) * z_m
    xc, yc, zc = camera_to_project_camera_coords(x_opt, y_opt, z_m)
    return np.array([xc, yc, zc], dtype=np.float64)


def project_cam_to_pixel(
    p: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> tuple[int, int] | None:
    """카메라 좌표 (x,y,z), z 전방 -> 픽셀."""
    z = float(p[2])
    if z < 0.02:
        return None
    u = int(round(fx * float(p[0]) / z + cx))
    v = int(round(fy * float(p[1]) / z + cy))
    return u, v


def project_cam_to_pixel_display(
    p: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> tuple[int, int]:
    """축 화살표 끝 등 표시용: z가 작아도 픽셀 반환."""
    z = max(float(p[2]), 0.05)
    u = int(round(fx * float(p[0]) / z + cx))
    v = int(round(fy * float(p[1]) / z + cy))
    return u, v


def camera_to_project_camera_coords(
    x_optical: float, y_optical: float, z_optical: float
) -> tuple[float, float, float]:
    return -x_optical, y_optical, -z_optical


def to_absolute_coords(
    x_cam: float,
    y_cam: float,
    z_cam: float,
    origin_x_in_cam: float,
    origin_y_in_cam: float,
    origin_z_in_cam: float,
) -> tuple[float, float, float]:
    return (
        x_cam - origin_x_in_cam,
        y_cam - origin_y_in_cam,
        z_cam - origin_z_in_cam,
    )


def apply_calibration_offset_mm(
    x_abs: float,
    y_abs: float,
    z_abs: float,
    dx_mm: float,
    dy_mm: float,
    dz_mm: float,
) -> tuple[float, float, float]:
    return (
        x_abs + (dx_mm / 1000.0),
        y_abs + (dy_mm / 1000.0),
        z_abs + (dz_mm / 1000.0),
    )


def draw_overlay_panel(
    img: np.ndarray,
    x: int,
    y_top: int,
    lines: list[tuple[str, tuple[int, int, int], float, int]],
) -> int:
    if not lines:
        return y_top
    h, w = img.shape[:2]
    fs_max = max(t[2] for t in lines)
    line_gap = int(22 * max(0.9, fs_max / 0.5))
    pad = 6
    max_w = 0
    for txt, _, fs, _ in lines:
        max_w = max(max_w, int(len(txt) * 9 * fs))
    box_w = min(w - x - 4, max_w + 2 * pad)
    box_h = len(lines) * line_gap + 2 * pad
    x2 = min(w - 1, x + box_w)
    y2 = min(h - 1, y_top + box_h)
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y_top), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    y = y_top + pad + int(line_gap * 0.7)
    for txt, col, fs, th in lines:
        cv2.putText(
            img,
            txt,
            (x + pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            col,
            th,
            cv2.LINE_AA,
        )
        y += line_gap
    return y2 + 8


def draw_axis_arrow(
    img: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    bgr: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    h, w = img.shape[:2]
    if not (0 <= p0[0] < w and 0 <= p0[1] < h):
        return
    x1 = int(max(0, min(w - 1, p1[0])))
    y1 = int(max(0, min(h - 1, p1[1])))
    cv2.arrowedLine(
        img, p0, (x1, y1), bgr, thickness, cv2.LINE_AA, tipLength=0.18
    )
