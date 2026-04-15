#!/usr/bin/env python3
"""
학습한 best.pt로 실시간 검출 + 무게중심·거리·법선(단위벡터).

거리 측정 방식 (택일):

  A) --realsense  Intel RealSense 깊이 센서 (권장) — 박스 ROI 안 깊이 중앙값 (m).

  B) 기본 OpenCV 웹캠 — 바운딩 박스 높이 + 물체 추정 높이로 핀홀 근사 (불확실).

무게중심: 검출 박스 중심 (cx, cy) 픽셀.

법선: 깊이 없을 때는 광선 역방향 근사. RealSense 사용 시 깊이 맵에서
      centroid 인근 소패치의 gradient로 표면 법선을 추정 (완만한 표면에 유효).

객체 좌표계 (RealSense + 물체 윤곽에 맞는 점만 PCA):
  - Z = 테이블(표면) 법선(고정).
  - 윤곽(마스크) 우선순위:
      1) --fastsam-weights …  FastSAM 박스 프롬프트 (일반 물체 윤곽에 강함)
      2) --seg-weights …      보조 YOLO 세그: 검출 박스 크롭 후 마스크, IoU로 매칭
      3) 메인 가중치가 세그 모델이면 그 마스크
      4) 깊이 밴드 + 타원 + 최대 연결요소 (폴백)
  - 마스크 안 깊이 점 → 평면 PCA → 단축=X, Y=Z×X.
  - --show-contour + --contour-style 로 윤곽선/피팅 타원/둘 다 표시 가능.
  - 화면 기본: YOLO 검출 박스 + 절대좌표(패널) + 객체 XYZ 축(RealSense+마스크).

RealSense 실행 예:

  pip install pyrealsense2
  python yolo_live_cam_3d_metrics.py --realsense
    # 가중치 기본: mini_project_main/runs/weights/weights/best.pt

OpenCV 웹캠 / 장치 번호:

  python yolo_live_cam_3d_metrics.py --list-cameras
  python yolo_live_cam_3d_metrics.py --camera 0

종료: 창 포커스 후 q
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

try:
    from ultralytics import FastSAM
except ImportError:  # pragma: no cover
    FastSAM = None  # type: ignore[misc, assignment]


def repo_root() -> Path:
    return Path(__file__).resolve().parent


# 학습 산출 best.pt (저장소 기준 고정 경로)
DEFAULT_WEIGHTS_PATH = repo_root() / "runs" / "weights" / "weights" / "best.pt"


def find_best_pt(search_under: Path) -> Path | None:
    cands = list(search_under.rglob("best.pt"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def open_capture(index: int) -> cv2.VideoCapture:
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def list_cameras(max_index: int = 8) -> None:
    """0~max_index 중 프레임을 읽을 수 있는 장치만 출력 (USB 웹캠 포함)."""
    print("사용 가능한 카메라 인덱스 (프레임 한 장 읽기 성공만 표시):\n")
    any_ok = False
    for i in range(max_index + 1):
        cap = open_capture(i)
        if not cap.isOpened():
            continue
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"  [{i}] 열리지만 프레임 없음")
            continue
        fh, fw = frame.shape[:2]
        print(f"  [{i}] OK  약 {fw}x{fh}")
        any_ok = True
    if not any_ok:
        print("  (없음) — 권한·케이블·다른 앱 점유를 확인하세요.")
    else:
        print(
            "\n실행: python yolo_live_cam_3d_metrics.py --camera <위 번호>\n"
            "RealSense 거리 측정: --realsense"
        )


def list_realsense_devices() -> None:
    try:
        import pyrealsense2 as rs
    except ImportError:
        raise SystemExit(
            "pyrealsense2 가 없습니다. 설치: pip install pyrealsense2"
        ) from None
    ctx = rs.context()
    devs = ctx.query_devices()
    if devs.size() == 0:
        print("연결된 RealSense 장치가 없습니다.")
        return
    print("RealSense 장치:\n")
    for i in range(devs.size()):
        d = devs[i]
        serial = d.get_info(rs.camera_info.serial_number)
        name = d.get_info(rs.camera_info.name)
        print(f"  [{i}] {name}  serial={serial}")
    print("\n실행 예: python yolo_live_cam_3d_metrics.py --realsense --rs-serial <serial>")


def load_class_heights(path: Path | None, data_yaml: Path | None) -> dict[int, float]:
    """클래스 id -> 실제 대표 높이(m). 없으면 빈 dict."""
    out: dict[int, float] = {}
    if path and path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        for k, v in raw.items():
            out[int(k)] = float(v)
        return out
    if data_yaml and data_yaml.is_file():
        d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        ch = d.get("class_heights_m")
        if isinstance(ch, dict):
            for k, v in ch.items():
                out[int(k)] = float(v)
    return out


def intrinsics_from_fov(
    w: int, h: int, fov_h_deg: float
) -> tuple[float, float, float, float]:
    fh = math.radians(fov_h_deg)
    fx = (0.5 * w) / math.tan(0.5 * fh)
    fy = fx
    cx = 0.5 * w
    cy = 0.5 * h
    return fx, fy, cx, cy


def ray_unit_opencv(
    u: float, v: float, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0
    r = np.array([x, y, z], dtype=np.float64)
    return r / (np.linalg.norm(r) + 1e-12)


def surface_normal_toward_camera(ray_to_point: np.ndarray) -> np.ndarray:
    return -ray_to_point


def estimate_depth_m(
    bbox_h_px: float,
    fy: float,
    object_height_m: float,
) -> float:
    if bbox_h_px < 1.0:
        return float("nan")
    return float(fy * object_height_m / bbox_h_px)


def clip_box_to_image(
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


def median_depth_in_roi(
    depth_m: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    w: int,
    h: int,
    margin: float = 0.08,
) -> float:
    """
    검출 박스 안에서 유효한 깊이(mm 스케일이 아닌 m) 픽셀의 중앙값.
    가장자리 노이즈 완화를 위해 안쪽으로 margin 만큼 줄인 ROI 사용.
    """
    bw = x2 - x1
    bh = y2 - y1
    if bw < 4 or bh < 4:
        return float("nan")
    dx = bw * margin * 0.5
    dy = bh * margin * 0.5
    xa, ya = x1 + dx, y1 + dy
    xb, yb = x2 - dx, y2 - dy
    if xb <= xa or yb <= ya:
        xa, ya, xb, yb = x1, y1, x2, y2
    xi1, yi1, xi2, yi2 = clip_box_to_image(xa, ya, xb, yb, w, h)
    roi = depth_m[yi1 : yi2 + 1, xi1 : xi2 + 1]
    valid = roi[np.isfinite(roi) & (roi > 0.05) & (roi < 10.0)]
    if valid.size < 3:
        return float("nan")
    return float(np.median(valid))


def mask_centroid_uv(mask_bool: np.ndarray) -> tuple[float, float] | None:
    """마스크 True 픽셀의 (u, v) 평균 = 윤곽 기준 2D 무게중심."""
    ys, xs = np.nonzero(mask_bool)
    if ys.size < 3:
        return None
    return float(xs.mean()), float(ys.mean())


def median_depth_on_mask(
    depth_m: np.ndarray, mask_bool: np.ndarray
) -> float:
    """마스크 안 깊이 중앙값 (m)."""
    vals = depth_m[mask_bool]
    vals = vals[np.isfinite(vals) & (vals > 0.05) & (vals < 10.0)]
    if vals.size < 3:
        return float("nan")
    return float(np.median(vals))


def normal_from_depth_patch(
    depth_m: np.ndarray,
    u: float,
    v: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    patch: int = 9,
) -> np.ndarray | None:
    """
    깊이 맵 중심차분으로 카메라 좌표계 표면 법선(카메라를 향하도록 점 근처).
    유효 깊이가 너무 적으면 None.
    """
    h, w = depth_m.shape
    ui = int(round(u))
    vi = int(round(v))
    r = patch // 2
    x1 = max(0, ui - r)
    y1 = max(0, vi - r)
    x2 = min(w - 1, ui + r)
    y2 = min(h - 1, vi + r)
    patch_d = depth_m[y1 : y2 + 1, x1 : x2 + 1].astype(np.float64)
    m = np.isfinite(patch_d) & (patch_d > 0.05) & (patch_d < 10.0)
    if np.count_nonzero(m) < patch * 2:
        return None
    # 그리드 인덱스 -> 3D 점
    ys, xs = np.indices(patch_d.shape)
    glob_x = (x1 + xs).astype(np.float64)
    glob_y = (y1 + ys).astype(np.float64)
    z = patch_d
    x3 = (glob_x - cx) / fx * z
    y3 = (glob_y - cy) / fy * z
    valid = m
    pts = np.stack([x3[valid], y3[valid], z[valid]], axis=1)
    if pts.shape[0] < 8:
        return None
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    n = vh[-1, :]
    n = n / (np.linalg.norm(n) + 1e-12)
    # 카메라 쪽을 보도록 부호 정리 (원점 방향과 내적이 양수가 되게)
    toward_cam = -centroid
    toward_cam = toward_cam / (np.linalg.norm(toward_cam) + 1e-12)
    if np.dot(n, toward_cam) < 0:
        n = -n
    return n.astype(np.float64)


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return (v / n).astype(np.float64)


def get_yolo_segmentation_mask_full(
    r0,
    box_index: int,
    w: int,
    h: int,
) -> np.ndarray | None:
    """YOLO 세그 모델이면 해당 검출의 전체 해상도 bool 마스크, 아니면 None."""
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
    """
    박스 전체가 아니라 '물체에 가깝다'고 보는 픽셀만 True.
    - z_ref_m 주변 깊이 밴드 (전경 물체 vs 배경 테이블 분리에 유효).
    - 타원 ROI로 AABB 코너(배경) 제거.
    """
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
    xi1, yi1, xi2, yi2 = clip_box_to_image(xa, ya, xb, yb, w, h)
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
    """a,b = [x1,y1,x2,y2] 픽셀."""
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
    fastsam_model,
    frame_bgr: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    device: str | int,
    h: int,
    w: int,
) -> np.ndarray | None:
    """FastSAM 박스 프롬프트 → 전체 이미지 크기 bool 마스크."""
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
    seg_model: YOLO,
    frame_bgr: np.ndarray,
    det_xyxy: np.ndarray,
    device: str | int,
    imgsz: int,
    crop_pad: float,
) -> np.ndarray | None:
    """보조 세그 YOLO를 박스 크롭에만 돌려, 검출 박스와 IoU가 가장 큰 마스크 선택."""
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


def draw_mask_contour_overlay(
    bgr: np.ndarray,
    mask_bool: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 128),
    thickness: int = 2,
) -> None:
    m = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(bgr, contours, -1, color, thickness, cv2.LINE_AA)


def draw_mask_contour_style(
    bgr: np.ndarray,
    mask_bool: np.ndarray,
    style: str,
    thickness: int = 2,
) -> None:
    """
    style:
      outline — 마스크 외곽 폴리라인 (기존과 동일)
      ellipse — 외곽 점에 최소자승 맞춤 타원 (박스 아님, 회전 가능)
      both    — 윤곽선 + 타원(색 구분)
    """
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
    """True 픽셀 중 면적 최대 연결요소만 남김."""
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
    """마스크가 True인 픽셀만 카메라 3D로 옮겨 점군 (z 전방)."""
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
    r0,
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
    fastsam_model=None,
    seg_aux_model: YOLO | None = None,
    device: str | int = "cpu",
    seg_aux_imgsz: int = 640,
    seg_crop_pad: float = 0.15,
) -> tuple[np.ndarray | None, str | None]:
    """
    물체 윤곽 마스크: FastSAM → 보조 YOLO 세그(크롭) → 메인 YOLO 세그 → 깊이 폴백.
    """
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
        depth_m,
        x1,
        y1,
        x2,
        y2,
        cx_box,
        cy_box,
        w,
        h,
        z_ref_m,
        z_band_m,
        ellipse_scale,
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
    """
    테이블 법선 Z(단위)를 고정하고, 평면 위 점군 PCA로 단축=X, 장축=Y(=Z×X) 오른손 좌표계.

    반환: axis_x, axis_y, axis_z, std_minor_m, std_major_m (표준편차 스케일 근사),
          eig_minor, eig_major (분산 고유값).
    """
    # Z는 "점군이 놓인 평면(테이블/물체 상면)"에 수직이 되도록,
    # 3D PCA의 최소분산 축(평면 법선)을 사용한다. 부호는 z_table 힌트와 맞춘다.
    z_hint = _normalize_vec(z_table)
    centroid = np.mean(points_cam, axis=0)
    rel = points_cam - centroid
    if rel.shape[0] < 8:
        return None

    cov3 = np.cov(rel, rowvar=False)
    if cov3.shape != (3, 3):
        return None
    eigvals3, eigvecs3 = np.linalg.eigh(cov3)
    i_min3 = int(np.argmin(eigvals3))
    z_axis = _normalize_vec(eigvecs3[:, i_min3])
    if float(np.dot(z_axis, z_hint)) < 0.0:
        z_axis = -z_axis

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

    x_dir = _normalize_vec(
        v_minor_2[0] * basis_u + v_minor_2[1] * basis_v
    )
    y_dir = _normalize_vec(np.cross(z_axis, x_dir))
    major_3d = _normalize_vec(
        v_major_2[0] * basis_u + v_major_2[1] * basis_v
    )
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


def project_cam_to_pixel(
    p: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> tuple[int, int] | None:
    """카메라 좌표 (x,y,z), z 전방 → 픽셀."""
    z = float(p[2])
    if z < 0.02:
        return None
    u = int(round(fx * float(p[0]) / z + cx))
    v = int(round(fy * float(p[1]) / z + cy))
    return u, v


def project_cam_to_pixel_display(
    p: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> tuple[int, int]:
    """축 화살표 끝 등 표시용: z가 너무 작으면 잘라서라도 픽셀 반환."""
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


def setup_realsense(
    serial: str,
    width: int,
    height: int,
    fps: int,
):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    if serial.strip():
        cfg.enable_device(serial.strip())
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(cfg)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    align = rs.align(rs.stream.color)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()

    return pipeline, align, depth_scale, intr


def main() -> None:
    root = repo_root()
    default_yaml = root / "datasets" / "yolo_final" / "data.yaml"

    ap = argparse.ArgumentParser(
        description="YOLO + 거리·무게중심·법선 (RealSense 깊이 또는 웹캠 근사)"
    )
    ap.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS_PATH),
        help=f"YOLO 가중치 .pt (기본: …/runs/weights/weights/)",
    )
    ap.add_argument("--project", type=Path, default=root / "runs", help="best.pt 검색 루트")
    ap.add_argument(
        "--realsense",
        action="store_true",
        help="Intel RealSense (pyrealsense2) 깊이로 박스 ROI 거리 측정",
    )
    ap.add_argument(
        "--rs-serial",
        type=str,
        default="",
        help="RealSense 여러 대일 때 시리얼 (rs-enumerate-devices 로 확인)",
    )
    ap.add_argument(
        "--rs-width",
        type=int,
        default=640,
        help="RealSense depth/color 너비 (둘 동일 해상도)",
    )
    ap.add_argument(
        "--rs-height",
        type=int,
        default=480,
        help="RealSense depth/color 높이",
    )
    ap.add_argument("--rs-fps", type=int, default=30, help="프레임레이트")
    ap.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV VideoCapture 인덱스 (--realsense 아닐 때만)",
    )
    ap.add_argument(
        "--list-cameras",
        action="store_true",
        help="OpenCV로 0~8번 확인 후 종료",
    )
    ap.add_argument(
        "--list-realsense",
        action="store_true",
        help="연결된 RealSense 이름·시리얼 출력 후 종료",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--no-half", action="store_true")
    ap.add_argument(
        "--fov-h-deg",
        type=float,
        default=60.0,
        help="웹캠 모드에서만 FOV 근사 (RealSense는 내부 캘리브 fx,fy 사용)",
    )
    ap.add_argument("--fx", type=float, default=0.0)
    ap.add_argument("--fy", type=float, default=0.0)
    ap.add_argument("--cx", type=float, default=0.0)
    ap.add_argument("--cy", type=float, default=0.0)
    ap.add_argument(
        "--default-object-height-m",
        type=float,
        default=0.12,
        help="웹캠 모드: 클래스별 높이 불명 시",
    )
    ap.add_argument("--class-heights", type=Path, default=None)
    ap.add_argument("--data-yaml", type=Path, default=default_yaml)
    ap.add_argument(
        "--origin-x",
        type=float,
        default=-0.80,
        help="절대좌표 원점: 카메라 프레임 X (m)",
    )
    ap.add_argument(
        "--origin-y",
        type=float,
        default=0.0,
        help="절대좌표 원점: 카메라 프레임 Y (m)",
    )
    ap.add_argument(
        "--origin-z",
        type=float,
        default=-0.96,
        help="절대좌표 원점: 카메라 프레임 Z (m)",
    )
    ap.add_argument("--calib-dx-mm", type=float, default=-20.0, help="절대좌표 보정 X (mm)")
    ap.add_argument("--calib-dy-mm", type=float, default=-20.0, help="절대좌표 보정 Y (mm)")
    ap.add_argument("--calib-dz-mm", type=float, default=140.0, help="절대좌표 보정 Z (mm)")
    ap.add_argument(
        "--pca-step",
        type=int,
        default=4,
        help="박스 내 깊이 샘플 간격(픽셀). 작을수록 점 많음.",
    )
    ap.add_argument(
        "--pca-arrow-m",
        type=float,
        default=0.08,
        help="화면에 그리는 X/Y/Z 축 화살표 길이(m).",
    )
    ap.add_argument(
        "--pca-z-band-m",
        type=float,
        default=0.045,
        help="세그 없을 때: ROI 깊이 중앙값±이 값(m) 안만 물체로 간주.",
    )
    ap.add_argument(
        "--pca-ellipse-scale",
        type=float,
        default=0.38,
        help="세그 없을 때: 박스 대비 타원 반축 비율(작을수록 코너 배경 제거).",
    )
    ap.add_argument(
        "--no-depth-pca-fallback",
        action="store_true",
        help="세그 마스크만 쓰고 깊이+타원 폴백 끄기(세그 전용 가중치 권장).",
    )
    ap.add_argument(
        "--fastsam-weights",
        type=str,
        default="",
        help="FastSAM 가중치(예: FastSAM-s.pt). 박스 기반 정밀 윤곽. 첫 실행 시 자동 다운로드.",
    )
    ap.add_argument(
        "--seg-weights",
        type=str,
        default="",
        help="보조 YOLO 세그 가중치(*-seg.pt 등). 검출 박스 크롭 안에서 마스크를 뽑아 윤곽에 사용.",
    )
    ap.add_argument(
        "--seg-aux-imgsz",
        type=int,
        default=640,
        help="보조 세그 모델 predict imgsz.",
    )
    ap.add_argument(
        "--seg-crop-pad",
        type=float,
        default=0.15,
        help="보조 세그용 박스 크롭 패딩 비율.",
    )
    ap.add_argument(
        "--show-contour",
        action="store_true",
        help="선택된 물체 마스크를 영상에 오버레이.",
    )
    ap.add_argument(
        "--contour-style",
        type=str,
        choices=("outline", "ellipse", "both"),
        default="outline",
        help="outline=폴리라인 윤곽, ellipse=윤곽에 맞춘 피팅 타원(회전·비율), both=둘 다.",
    )
    ap.add_argument(
        "--no-bboxes",
        action="store_true",
        help="검출 박스를 그리지 않음 (기본: YOLO처럼 박스·라벨·신뢰도 표시).",
    )
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return
    if args.list_realsense:
        list_realsense_devices()
        return

    wpath = Path(args.weights).expanduser().resolve()
    if not wpath.is_file():
        found = find_best_pt(Path(args.project).resolve())
        if found is None:
            raise SystemExit(
                f"가중치 없음: {wpath}\n"
                f"  기본 경로({DEFAULT_WEIGHTS_PATH})에 파일이 없고, "
                f"--project({args.project}) 아래에서도 best.pt 를 찾지 못했습니다.\n"
                "  --weights 로 best.pt 전체 경로를 지정하세요."
            )
        print(f"기본 가중치 없음 → 대체 사용: {found}")
        wpath = found

    class_heights = load_class_heights(args.class_heights, args.data_yaml)
    dev = args.device or (0 if torch.cuda.is_available() else "cpu")
    use_half = (not args.no_half) and dev != "cpu" and torch.cuda.is_available()

    model = YOLO(str(wpath))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    fastsam_model = None
    fw = str(args.fastsam_weights).strip()
    if fw:
        if FastSAM is None:
            print("FastSAM 을 불러올 수 없습니다. ultralytics 를 최신으로 올려 보세요.")
        else:
            try:
                fastsam_model = FastSAM(fw)
                print(f"FastSAM 로드: {fw} (박스 프롬프트 윤곽)")
            except Exception as e:
                print(f"FastSAM 로드 실패: {e}")

    seg_aux_model = None
    sw = str(args.seg_weights).strip()
    if sw:
        sp = Path(sw).expanduser().resolve()
        if not sp.is_file():
            print(f"보조 세그 가중치 파일 없음: {sp}")
        else:
            try:
                seg_aux_model = YOLO(str(sp))
                print(f"보조 세그 YOLO 로드: {sp}")
            except Exception as e:
                print(f"보조 세그 YOLO 로드 실패: {e}")

    if (
        getattr(model, "task", "") != "segment"
        and fastsam_model is None
        and seg_aux_model is None
    ):
        print(
            "\n[팁] 물체 윤곽을 더 정확히: "
            "`--fastsam-weights FastSAM-s.pt` 또는 학습한 세그 가중치를 "
            "`--seg-weights .../best.pt` 로 지정하세요. "
            "메인 가중치를 YOLO-seg 로 바꿔도 됩니다.\n"
        )

    use_rs = args.realsense
    if use_rs:
        try:
            pipeline, align, depth_scale, rs_intr = setup_realsense(
                args.rs_serial, args.rs_width, args.rs_height, args.rs_fps
            )
        except ImportError:
            raise SystemExit(
                "pyrealsense2 를 설치하세요: pip install pyrealsense2\n"
                "또는 Intel RealSense SDK 가 있는 환경인지 확인하세요."
            ) from None
        fx = float(rs_intr.fx)
        fy = float(rs_intr.fy)
        cx = float(rs_intr.ppx)
        cy = float(rs_intr.ppy)
        cap = None
        print(
            f"RealSense 깊이 정렬(color), scale={depth_scale:.6f} m/unit, "
            f"fx={fx:.2f} fy={fy:.2f} cx={cx:.1f} cy={cy:.1f}"
        )
    else:
        pipeline = None
        align = None
        depth_scale = 0.0
        cap = open_capture(args.camera)
        if not cap.isOpened():
            raise SystemExit(
                f"카메라를 열 수 없습니다 (index={args.camera}).\n"
                "  --list-cameras 로 확인하거나 RealSense면 `--realsense` 를 쓰세요."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window = "YOLO metrics — q 종료"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # 창을 바로 띄워서(검은 화면 방지) 사용자가 화면을 볼 수 있게 함
    if os.environ.get("DISPLAY") is None and sys.platform.startswith("linux"):
        print(
            "\n[경고] DISPLAY 가 비어 있습니다. 로컬 데스크톱 터미널에서 실행하거나 "
            "SSH면 X11 포워딩(-X) / Wayland 환경을 확인하세요.\n"
        )
    preview = None
    if use_rs:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        cf = aligned.get_color_frame()
        if cf is not None:
            preview = np.asanyarray(cf.get_data()).copy()
    else:
        ok_pv, preview = cap.read()
        if not ok_pv or preview is None:
            preview = None
    if preview is not None:
        cv2.putText(
            preview,
            "Camera OK | Starting YOLO...  (q to quit)",
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, preview)
        ph, pw = preview.shape[:2]
        cv2.resizeWindow(window, min(pw, 1920), min(ph, 1080))
        cv2.waitKey(1)
    else:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            blank,
            "No camera frame — check USB / RealSense",
            (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, blank)
        cv2.resizeWindow(window, 640, 480)
        cv2.waitKey(1)

    mode = "RealSense depth" if use_rs else "pinhole approx"
    print(f"가중치: {wpath}")
    print(f"device={dev}, 거리={mode}, class_heights={class_heights or '(웹캠만 해당)'}")
    print(
        "윤곽: FastSAM>보조세그>메인세그>깊이폴백 | "
        f"박스={'off' if args.no_bboxes else 'on'} | step={args.pca_step} | "
        f"contour={args.show_contour} style={args.contour_style}"
    )
    print(
        f"절대좌표 원점(cam m): ({args.origin_x:.3f},{args.origin_y:.3f},{args.origin_z:.3f}), "
        f"보정(mm): ({args.calib_dx_mm:+.1f},{args.calib_dy_mm:+.1f},{args.calib_dz_mm:+.1f})"
    )
    print("종료: q")

    try:
        while True:
            if use_rs:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())
                raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
                depth_m = raw * float(depth_scale)
            else:
                ok, frame = cap.read()
                if not ok:
                    print("프레임 읽기 실패.")
                    break
                h, w = frame.shape[:2]
                if args.fx > 0 and args.fy > 0:
                    fx, fy = args.fx, args.fy
                    cx = args.cx if args.cx > 0 else w * 0.5
                    cy = args.cy if args.cy > 0 else h * 0.5
                else:
                    fx, fy, cx, cy = intrinsics_from_fov(w, h, args.fov_h_deg)
                depth_m = None

            h, w = frame.shape[:2]

            t0 = time.perf_counter()
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=dev,
                half=use_half,
                verbose=False,
                retina_masks=True,
            )
            t1 = time.perf_counter()
            r0 = results[0]
            out = r0.plot(
                boxes=not args.no_bboxes,
                labels=True,
                conf=True,
                line_width=2,
                masks=False,
            )

            fps = 1.0 / max(t1 - t0, 1e-6)
            hint = (
                f"FPS ~{fps:.1f}  RealSense Z=ROI median (m)"
                if use_rs
                else f"FPS ~{fps:.1f}  Z~fy*H/h (근사)"
            )
            cv2.putText(
                out,
                hint,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                "Obj: X=green Y=magenta Z=yellow | ABS XYZ = project frame (m)",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 220, 255),
                1,
                cv2.LINE_AA,
            )

            if r0.boxes is None or len(r0.boxes) == 0:
                cv2.imshow(window, out)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue

            boxes = r0.boxes.xyxy.cpu().numpy()
            clss = r0.boxes.cls.cpu().numpy().astype(int)

            line_y = 72
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = b
                bh = max(y2 - y1, 1.0)
                cx_box = 0.5 * (x1 + x2)
                cy_box = 0.5 * (y1 + y2)
                cid = clss[i] if i < len(clss) else 0
                label = names.get(cid, str(cid))
                H_m = class_heights.get(cid, args.default_object_height_m)

                if use_rs and depth_m is not None:
                    z_m = median_depth_in_roi(depth_m, x1, y1, x2, y2, w, h)
                    n_est = normal_from_depth_patch(
                        depth_m, cx_box, cy_box, fx, fy, cx, cy, patch=9
                    )
                    if n_est is not None:
                        normal = n_est
                    else:
                        ray = ray_unit_opencv(cx_box, cy_box, fx, fy, cx, cy)
                        normal = surface_normal_toward_camera(ray)
                else:
                    z_m = estimate_depth_m(bh, fy, H_m)
                    ray = ray_unit_opencv(cx_box, cy_box, fx, fy, cx, cy)
                    normal = surface_normal_toward_camera(ray)

                pca_info = None
                mask_obj = None
                pca_src: str | None = None
                if use_rs and depth_m is not None:
                    mask_obj, pca_src = build_object_mask_for_pca(
                        r0,
                        i,
                        depth_m,
                        x1,
                        y1,
                        x2,
                        y2,
                        cx_box,
                        cy_box,
                        w,
                        h,
                        z_m,
                        float(args.pca_z_band_m),
                        float(args.pca_ellipse_scale),
                        use_depth_fallback=not args.no_depth_pca_fallback,
                        frame_bgr=frame,
                        fastsam_model=fastsam_model,
                        seg_aux_model=seg_aux_model,
                        device=dev,
                        seg_aux_imgsz=int(args.seg_aux_imgsz),
                        seg_crop_pad=float(args.seg_crop_pad),
                    )
                    if args.show_contour and mask_obj is not None:
                        draw_mask_contour_style(
                            out, mask_obj, args.contour_style, thickness=2
                        )

                    if mask_obj is not None:
                        uv_m = mask_centroid_uv(mask_obj)
                        if uv_m is not None:
                            u_m, v_m = uv_m
                            z_m_mask = median_depth_on_mask(depth_m, mask_obj)
                            if math.isfinite(z_m_mask):
                                z_m = z_m_mask
                            n_est2 = normal_from_depth_patch(
                                depth_m, u_m, v_m, fx, fy, cx, cy, patch=11
                            )
                            if n_est2 is not None:
                                normal = n_est2
                            else:
                                ray = ray_unit_opencv(u_m, v_m, fx, fy, cx, cy)
                                normal = surface_normal_toward_camera(ray)

                        pts_cam = collect_points_cam_from_mask(
                            depth_m,
                            mask_obj,
                            fx,
                            fy,
                            cx,
                            cy,
                            step=max(1, int(args.pca_step)),
                        )
                        if pts_cam is not None:
                            pca_info = object_frame_pca_short_x_long_y(
                                pts_cam, normal
                            )
                            if pca_info is not None and pca_src is not None:
                                pca_info["source"] = pca_src

                alen = float(args.pca_arrow_m)
                if pca_info is not None:
                    c0 = pca_info["centroid_cam"]
                    ax = pca_info["axis_x"]
                    ay = pca_info["axis_y"]
                    az = pca_info["axis_z"]
                    p0 = project_cam_to_pixel(c0, fx, fy, cx, cy)
                    if p0 is not None:
                        # ---- 사분면 분포 기반 X축 방향 선택 ----
                        # 마스크 픽셀의 분포를 (p0 기준) 사분면으로 나눠,
                        # Q1 또는 Q3에 많이 몰리면 X를 Q2(좌상향),
                        # Q2 또는 Q4에 많이 몰리면 X를 Q1(우상향)으로 고정.
                        #
                        # (픽셀 좌표계: u 오른쪽+, v 아래+. 사분면은 y=위쪽+로 보기 위해 dy=-(v-v0))
                        desired = "Q1"  # 기본: 우상향
                        if mask_obj is not None and np.any(mask_obj):
                            ys, xs = np.nonzero(mask_obj)
                            if ys.size >= 20:
                                step = max(1, int(ys.size // 1800))
                                xs_s = xs[::step]
                                ys_s = ys[::step]
                                dx = xs_s.astype(np.int32) - int(p0[0])
                                dy = -(ys_s.astype(np.int32) - int(p0[1]))
                                q1 = int(np.count_nonzero((dx >= 0) & (dy >= 0)))
                                q2 = int(np.count_nonzero((dx < 0) & (dy >= 0)))
                                q3 = int(np.count_nonzero((dx < 0) & (dy < 0)))
                                q4 = int(np.count_nonzero((dx >= 0) & (dy < 0)))
                                if (q1 + q3) >= (q2 + q4):
                                    desired = "Q2"  # 좌상향
                                else:
                                    desired = "Q1"  # 우상향

                        def _ensure_y_up_and_x_quadrant(
                            ax_in: np.ndarray, az_in: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray, str]:
                            ax2 = _normalize_vec(ax_in)
                            az2 = _normalize_vec(az_in)
                            ay2 = _normalize_vec(np.cross(az2, ax2))

                            # Y가 화면에서 위쪽(-v)으로 가게 부호 정리
                            y_tip_cam2 = np.asarray(c0 + alen * ay2, dtype=np.float64)
                            p_y2 = project_cam_to_pixel(y_tip_cam2, fx, fy, cx, cy)
                            if p_y2 is None:
                                p_y2 = project_cam_to_pixel_display(
                                    y_tip_cam2, fx, fy, cx, cy
                                )
                            if p_y2[1] > p0[1]:
                                ax2 = -ax2
                                ay2 = -ay2

                            # X가 원하는 사분면(Q1=우상, Q2=좌상)으로 가게 부호 정리
                            x_tip_cam2 = np.asarray(c0 + alen * ax2, dtype=np.float64)
                            p_x2 = project_cam_to_pixel(x_tip_cam2, fx, fy, cx, cy)
                            if p_x2 is None:
                                p_x2 = project_cam_to_pixel_display(
                                    x_tip_cam2, fx, fy, cx, cy
                                )
                            dx2 = p_x2[0] - p0[0]
                            dy2 = -(p_x2[1] - p0[1])
                            cur = "Q2" if (dx2 < 0 and dy2 >= 0) else "Q1"
                            if desired == "Q2":
                                ok = dx2 < 0 and dy2 >= 0
                            else:
                                ok = dx2 >= 0 and dy2 >= 0
                            if not ok:
                                ax2 = -ax2
                                ay2 = -ay2
                                cur = "Q2" if desired == "Q2" else "Q1"
                            return ax2, ay2, cur

                        ax, ay, cur_q = _ensure_y_up_and_x_quadrant(ax, az)
                        pca_info["axis_x"] = ax
                        pca_info["axis_y"] = ay
                        pca_info["x_quadrant"] = cur_q
                        pca_info["x_rule"] = f"{desired}_by_mask_quadrants"

                        cv2.circle(out, p0, 6, (0, 255, 255), -1, cv2.LINE_AA)
                        for end_arr, col, aname in (
                            (c0 + alen * ax, (0, 255, 0), "X"),
                            (c0 + alen * ay, (255, 0, 255), "Y"),
                            (c0 + alen * az, (255, 255, 0), "Z"),
                        ):
                            end = np.asarray(end_arr, dtype=np.float64)
                            p1 = project_cam_to_pixel(end, fx, fy, cx, cy)
                            if p1 is None:
                                p1 = project_cam_to_pixel_display(end, fx, fy, cx, cy)
                            draw_axis_arrow(out, p0, p1, col, 2)
                            tx = int(max(0, min(w - 1, p1[0] + 4)))
                            ty = int(max(0, min(h - 1, p1[1] - 4)))
                            cv2.putText(
                                out,
                                aname,
                                (tx, ty),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                col,
                                2,
                                cv2.LINE_AA,
                            )
                        cv2.putText(
                            out,
                            f"Xrule={desired}->{cur_q}",
                            (p0[0] + 8, p0[1] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (200, 255, 200),
                            1,
                            cv2.LINE_AA,
                        )
                else:
                    if mask_obj is not None:
                        um = mask_centroid_uv(mask_obj)
                        if um is not None:
                            pt = (int(round(um[0])), int(round(um[1])))
                        else:
                            pt = (int(round(cx_box)), int(round(cy_box)))
                    else:
                        pt = (int(round(cx_box)), int(round(cy_box)))
                    cv2.circle(out, pt, 6, (0, 255, 255), -1, cv2.LINE_AA)
                    norm_tip = np.linalg.norm([cx - pt[0], cy - pt[1]]) + 1e-6
                    tip = (
                        int(round(cx + 40 * (cx - pt[0]) / norm_tip)),
                        int(round(cy + 40 * (cy - pt[1]) / norm_tip)),
                    )
                    cv2.arrowedLine(
                        out, pt, tip, (255, 180, 0), 2, cv2.LINE_AA, tipLength=0.25
                    )

                ux_show, uy_show = cx_box, cy_box
                if mask_obj is not None:
                    um2 = mask_centroid_uv(mask_obj)
                    if um2 is not None:
                        ux_show, uy_show = um2

                z_show = f"{z_m:.3f}m" if not math.isnan(z_m) else "nan"
                if math.isfinite(z_m) and not math.isnan(z_m):
                    x_opt = ((ux_show - cx) / fx) * z_m
                    y_opt = ((uy_show - cy) / fy) * z_m
                    z_opt = z_m
                    x_cam, y_cam, z_cam = camera_to_project_camera_coords(
                        x_opt, y_opt, z_opt
                    )
                    x_abs, y_abs, z_abs = to_absolute_coords(
                        x_cam,
                        y_cam,
                        z_cam,
                        args.origin_x,
                        args.origin_y,
                        args.origin_z,
                    )
                    x_abs, y_abs, z_abs = apply_calibration_offset_mm(
                        x_abs,
                        y_abs,
                        z_abs,
                        args.calib_dx_mm,
                        args.calib_dy_mm,
                        args.calib_dz_mm,
                    )
                else:
                    x_abs = y_abs = z_abs = float("nan")

                def _fmt_m(v: float) -> str:
                    return f"{v:+.4f}" if math.isfinite(v) else "  nan"

                ntxt = (
                    f"{label}  uv=({ux_show:.0f},{uy_show:.0f})px  Z={z_show}  "
                    f"ABS=({_fmt_m(x_abs)},{_fmt_m(y_abs)},{_fmt_m(z_abs)})m"
                )
                if pca_info is not None:
                    src = pca_info.get("source", "?")
                    ntxt += (
                        f"  [{src}] sx={pca_info['std_minor_m']:.3f} "
                        f"sy={pca_info['std_major_m']:.3f}m"
                    )
                cv2.putText(
                    out,
                    ntxt,
                    (10, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (220, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
                line_y += 20
                abs_panel: list[tuple[str, tuple[int, int, int], float, int]] = [
                    (f"{label}  absolute XYZ [m]", (0, 255, 255), 0.55, 2),
                    (f"  X = {_fmt_m(x_abs)}", (235, 235, 240), 0.52, 1),
                    (f"  Y = {_fmt_m(y_abs)}", (235, 235, 240), 0.52, 1),
                    (f"  Z = {_fmt_m(z_abs)}", (235, 235, 240), 0.52, 1),
                ]
                line_y = draw_overlay_panel(out, 10, line_y, abs_panel)

                if use_rs:
                    line = (
                        f"[{label}] centroid_uv=({ux_show:.1f},{uy_show:.1f}) "
                        f"(mask/3D) dist_m={z_m}  "
                        f"ABS_m=({_fmt_m(x_abs)},{_fmt_m(y_abs)},{_fmt_m(z_abs)})  "
                        f"normal_cam={normal[0]:+.4f},{normal[1]:+.4f},{normal[2]:+.4f}"
                    )
                    if pca_info is not None:
                        ax = pca_info["axis_x"]
                        ay = pca_info["axis_y"]
                        az = pca_info["axis_z"]
                        src = pca_info.get("source", "?")
                        line += (
                            f"\n  obj_frame[{src}]: "
                            f"X=[{ax[0]:+.3f},{ax[1]:+.3f},{ax[2]:+.3f}] "
                            f"Y=[{ay[0]:+.3f},{ay[1]:+.3f},{ay[2]:+.3f}] "
                            f"Z=[{az[0]:+.3f},{az[1]:+.3f},{az[2]:+.3f}] "
                            f"sigma_minor={pca_info['std_minor_m']:.4f}m "
                            f"sigma_major={pca_info['std_major_m']:.4f}m"
                        )
                    print(line)
                else:
                    print(
                        f"[{label}] centroid_px=({cx_box:.1f},{cy_box:.1f}) "
                        f"dist_m~(pinhole)={z_m:.3f} (H_ref={H_m}m, h_px={bh:.1f}) "
                        f"ABS_m=({_fmt_m(x_abs)},{_fmt_m(y_abs)},{_fmt_m(z_abs)})  "
                        f"normal_cam={normal[0]:+.4f},{normal[1]:+.4f},{normal[2]:+.4f} "
                        "(PCA·객체축은 RealSense+마스크 권장)"
                    )

            cv2.imshow(window, out)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        if cap is not None:
            cap.release()
        if pipeline is not None:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
