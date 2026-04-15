from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return (v / n).astype(np.float64)


def _clip_box_to_image(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> tuple[int, int, int, int]:
    xi1 = int(max(0, min(w - 1, round(x1))))
    yi1 = int(max(0, min(h - 1, round(y1))))
    xi2 = int(max(0, min(w - 1, round(x2))))
    yi2 = int(max(0, min(h - 1, round(y2))))
    if xi2 < xi1:
        xi1, xi2 = xi2, xi1
    if yi2 < yi1:
        yi1, yi2 = yi2, yi1
    return xi1, yi1, xi2, yi2


def get_yolo_segmentation_mask_full(
    r0: Any,
    box_index: int,
    w: int,
    h: int,
) -> np.ndarray | None:
    masks = getattr(r0, "masks", None)
    boxes = getattr(r0, "boxes", None)
    if masks is None or boxes is None or len(boxes) == 0:
        return None
    if box_index >= len(boxes):
        return None
    data = getattr(masks, "data", None)
    if data is None:
        return None
    nmask = int(data.shape[0])
    if box_index >= nmask:
        return None
    m = data[box_index].detach().cpu().numpy().astype(np.float32)
    if m.ndim != 2:
        return None
    mh, mw = m.shape
    if mh != h or mw != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    return m > 0.5


def build_depth_object_mask(
    depth_m: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cx_box: float,
    cy_box: float,
    w: int,
    h: int,
    z_ref_m: float,
    z_band_m: float,
    ellipse_scale: float,
    inner_margin: float = 0.12,
) -> np.ndarray | None:
    if not math.isfinite(z_ref_m):
        return None
    bw = x2 - x1
    bh = y2 - y1
    if bw < 4 or bh < 4:
        return None
    dx = bw * inner_margin * 0.5
    dy = bh * inner_margin * 0.5
    xa, ya = x1 + dx, y1 + dy
    xb, yb = x2 - dx, y2 - dy
    if xb <= xa or yb <= ya:
        xa, ya, xb, yb = x1, y1, x2, y2
    xi1, yi1, xi2, yi2 = _clip_box_to_image(xa, ya, xb, yb, w, h)
    zlo = z_ref_m - z_band_m
    zhi = z_ref_m + z_band_m
    roi = depth_m[yi1 : yi2 + 1, xi1 : xi2 + 1]
    m_roi = (
        np.isfinite(roi)
        & (roi > 0.05)
        & (roi < 10.0)
        & (roi >= zlo)
        & (roi <= zhi)
    )
    ell = np.zeros((h, w), dtype=np.uint8)
    ax = max(3, int(ellipse_scale * 0.5 * bw))
    ay = max(3, int(ellipse_scale * 0.5 * bh))
    cv2.ellipse(
        ell,
        (int(round(cx_box)), int(round(cy_box))),
        (ax, ay),
        0,
        0,
        360,
        255,
        thickness=-1,
    )
    full = np.zeros((h, w), dtype=bool)
    full[yi1 : yi2 + 1, xi1 : xi2 + 1] = m_roi
    return full & (ell.astype(bool))


def box_xyxy_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    ae = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    be = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    union = ae + be - inter
    return float(inter / union)


def apply_depth_band_to_mask(
    mask: np.ndarray,
    depth_m: np.ndarray,
    z_ref_m: float,
    z_band_m: float,
) -> np.ndarray:
    if not math.isfinite(z_ref_m):
        return mask
    zlo = z_ref_m - z_band_m
    zhi = z_ref_m + z_band_m
    depth_ok = (
        np.isfinite(depth_m)
        & (depth_m > 0.05)
        & (depth_m < 10.0)
        & (depth_m >= zlo)
        & (depth_m <= zhi)
    )
    mt = mask & depth_ok
    return mt if np.count_nonzero(mt) >= 12 else mask


def mask_from_fastsam(
    fastsam_model: Any,
    frame_bgr: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    device: str | int,
    h: int,
    w: int,
) -> np.ndarray | None:
    try:
        b = [float(x1), float(y1), float(x2), float(y2)]
        res = fastsam_model.predict(
            source=frame_bgr,
            bboxes=[b],
            device=device,
            verbose=False,
            retina_masks=True,
        )
    except Exception:
        return None
    r0 = res[0]
    masks = getattr(r0, "masks", None)
    if masks is None:
        return None
    data = getattr(masks, "data", None)
    if data is None or data.shape[0] == 0:
        return None
    acc = np.zeros((h, w), dtype=bool)
    for k in range(int(data.shape[0])):
        m = data[k].detach().cpu().numpy().astype(np.float32)
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        acc |= m > 0.5
    return acc if np.count_nonzero(acc) >= 20 else None


def mask_from_seg_aux_crop(
    seg_model: Any,
    frame_bgr: np.ndarray,
    det_xyxy: np.ndarray,
    device: str | int,
    imgsz: int,
    crop_pad: float,
) -> np.ndarray | None:
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(det_xyxy[i]) for i in range(4)]
    bw = x2 - x1
    bh = y2 - y1
    pad = crop_pad * max(bw, bh, 8.0)
    xa = max(0, int(math.floor(x1 - pad)))
    ya = max(0, int(math.floor(y1 - pad)))
    xb = min(W, int(math.ceil(x2 + pad)))
    yb = min(H, int(math.ceil(y2 + pad)))
    if xb <= xa + 2 or yb <= ya + 2:
        return None
    crop = frame_bgr[ya:yb, xa:xb]
    try:
        res = seg_model.predict(
            crop,
            imgsz=imgsz,
            conf=0.25,
            device=device,
            verbose=False,
            retina_masks=True,
        )
    except Exception:
        return None
    r0 = res[0]
    masks = getattr(r0, "masks", None)
    if masks is None:
        return None
    data = getattr(masks, "data", None)
    if data is None or data.shape[0] == 0:
        return None
    gt = det_xyxy.astype(np.float32)
    cw = xb - xa
    ch = yb - ya
    best_iou = 0.0
    best_full: np.ndarray | None = None
    for k in range(int(data.shape[0])):
        m = data[k].detach().cpu().numpy().astype(np.float32)
        if m.shape[0] != ch or m.shape[1] != cw:
            m = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_LINEAR)
        mbin = m > 0.5
        ys, xs = np.nonzero(mbin)
        if ys.size < 10:
            continue
        bx1 = float(xs.min() + xa)
        bx2 = float(xs.max() + xa)
        by1 = float(ys.min() + ya)
        by2 = float(ys.max() + ya)
        bb = np.array([bx1, by1, bx2, by2], dtype=np.float32)
        iou = box_xyxy_iou(gt, bb)
        if iou > best_iou:
            best_iou = iou
            full = np.zeros((H, W), dtype=bool)
            full[ya:yb, xa:xb] = mbin
            best_full = full
    if best_full is None or best_iou < 0.08:
        return None
    return best_full


def draw_mask_contour_style(
    bgr: np.ndarray,
    mask_bool: np.ndarray,
    style: str,
    thickness: int = 2,
) -> None:
    m = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    cnt = max(contours, key=cv2.contourArea)
    col_line: tuple[int, int, int] = (0, 255, 128)
    col_ell: tuple[int, int, int] = (255, 200, 0)
    if style in ("outline", "both"):
        cv2.drawContours(bgr, [cnt], -1, col_line, thickness, cv2.LINE_AA)
    if style in ("ellipse", "both"):
        if len(cnt) >= 5:
            ell = cv2.fitEllipse(cnt)
            c = col_ell if style == "both" else col_line
            cv2.ellipse(bgr, ell, c, thickness, cv2.LINE_AA)
        elif style == "ellipse":
            cv2.drawContours(bgr, [cnt], -1, col_line, thickness, cv2.LINE_AA)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask
    m8 = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if num <= 1:
        return mask
    best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best_label


def collect_points_cam_from_mask(
    depth_m: np.ndarray,
    mask_bool: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    step: int = 2,
    max_points: int = 8000,
) -> np.ndarray | None:
    ys, xs = np.nonzero(mask_bool)
    if ys.size < 12:
        return None
    st = max(1, int(step))
    idx = np.arange(0, ys.size, st, dtype=np.int32)
    if idx.size > max_points:
        idx = np.linspace(0, idx.size - 1, max_points).astype(np.int32)
    pts: list[list[float]] = []
    for j in idx:
        v = int(ys[j])
        u = int(xs[j])
        z = float(depth_m[v, u])
        if not math.isfinite(z) or z < 0.05 or z > 10.0:
            continue
        x = (float(u) - cx) / fx * z
        y = (float(v) - cy) / fy * z
        pts.append([x, y, z])
    if len(pts) < 12:
        return None
    return np.asarray(pts, dtype=np.float64)


def build_object_mask_for_pca(
    r0: Any,
    box_index: int,
    depth_m: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cx_box: float,
    cy_box: float,
    w: int,
    h: int,
    z_ref_m: float,
    z_band_m: float,
    ellipse_scale: float,
    use_depth_fallback: bool,
    frame_bgr: np.ndarray | None = None,
    fastsam_model: Any = None,
    seg_aux_model: Any = None,
    device: str | int = "cpu",
    seg_aux_imgsz: int = 640,
    seg_crop_pad: float = 0.15,
) -> tuple[np.ndarray | None, str | None]:
    det_box = np.array([x1, y1, x2, y2], dtype=np.float32)
    if fastsam_model is not None and frame_bgr is not None:
        m = mask_from_fastsam(
            fastsam_model, frame_bgr, x1, y1, x2, y2, device, h, w
        )
        if m is not None:
            if use_depth_fallback:
                m = apply_depth_band_to_mask(m, depth_m, z_ref_m, z_band_m)
            m = largest_connected_component(m)
            if np.count_nonzero(m) >= 12:
                return m, "fastsam"
    if seg_aux_model is not None and frame_bgr is not None:
        m = mask_from_seg_aux_crop(
            seg_aux_model,
            frame_bgr,
            det_box,
            device,
            seg_aux_imgsz,
            seg_crop_pad,
        )
        if m is not None:
            if use_depth_fallback:
                m = apply_depth_band_to_mask(m, depth_m, z_ref_m, z_band_m)
            m = largest_connected_component(m)
            if np.count_nonzero(m) >= 12:
                return m, "yolo_seg_aux"
    seg = get_yolo_segmentation_mask_full(r0, box_index, w, h)
    if seg is not None and np.count_nonzero(seg) >= 20:
        m = seg.copy()
        if use_depth_fallback:
            m = apply_depth_band_to_mask(m, depth_m, z_ref_m, z_band_m)
        m = largest_connected_component(m)
        if np.count_nonzero(m) >= 12:
            return m, "yolo_seg"
    if not use_depth_fallback or not math.isfinite(z_ref_m):
        return None, None
    dm = build_depth_object_mask(
        depth_m, x1, y1, x2, y2, cx_box, cy_box, w, h, z_ref_m, z_band_m, ellipse_scale
    )
    if dm is None or not np.any(dm):
        return None, None
    dm = largest_connected_component(dm)
    if np.count_nonzero(dm) < 12:
        return None, None
    return dm, "depth_z_ellipse"


def object_frame_pca_short_x_long_y(
    points_cam: np.ndarray,
    z_table: np.ndarray,
) -> dict | None:
    z_axis = _normalize_vec(z_table)
    centroid = np.mean(points_cam, axis=0)
    rel = points_cam - centroid
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, z_axis))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    basis_u = _normalize_vec(np.cross(z_axis, ref))
    basis_v = _normalize_vec(np.cross(z_axis, basis_u))
    u1 = rel @ basis_u
    u2 = rel @ basis_v
    coords = np.column_stack((u1, u2))
    if coords.shape[0] < 8:
        return None
    cov = np.cov(coords, rowvar=False)
    if cov.shape != (2, 2):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    i_min = int(np.argmin(eigvals))
    i_max = int(np.argmax(eigvals))
    v_minor_2 = eigvecs[:, i_min]
    v_major_2 = eigvecs[:, i_max]
    x_dir = _normalize_vec(v_minor_2[0] * basis_u + v_minor_2[1] * basis_v)
    y_dir = _normalize_vec(np.cross(z_axis, x_dir))
    major_3d = _normalize_vec(v_major_2[0] * basis_u + v_major_2[1] * basis_v)
    if float(np.dot(major_3d, y_dir)) < 0.0:
        x_dir = -x_dir
        y_dir = _normalize_vec(np.cross(z_axis, x_dir))
    std_minor = float(math.sqrt(max(eigvals[i_min], 0.0)))
    std_major = float(math.sqrt(max(eigvals[i_max], 0.0)))
    return {
        "axis_x": x_dir,
        "axis_y": y_dir,
        "axis_z": z_axis,
        "std_minor_m": std_minor,
        "std_major_m": std_major,
        "eig_minor": float(eigvals[i_min]),
        "eig_major": float(eigvals[i_max]),
        "centroid_cam": centroid,
    }
