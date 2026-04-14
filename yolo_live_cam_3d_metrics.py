#!/usr/bin/env python3
"""
학습한 best.pt로 실시간 검출 + 무게중심·거리·법선(단위벡터).

거리 측정 방식 (택일):

  A) --realsense  Intel RealSense 깊이 센서 (권장) — 박스 ROI 안 깊이 중앙값 (m).

  B) 기본 OpenCV 웹캠 — 바운딩 박스 높이 + 물체 추정 높이로 핀홀 근사 (불확실).

무게중심: 검출 박스 중심 (cx, cy) 픽셀.

법선: 깊이 없을 때는 광선 역방향 근사. RealSense 사용 시 깊이 맵에서
      centroid 인근 소패치의 gradient로 표면 법선을 추정 (완만한 표면에 유효).

RealSense 실행 예:

  pip install pyrealsense2
  python yolo_live_cam_3d_metrics.py --realsense --weights runs/weights/weights/best.pt

OpenCV 웹캠 / 장치 번호:

  python yolo_live_cam_3d_metrics.py --list-cameras
  python yolo_live_cam_3d_metrics.py --camera 0

종료: 창 포커스 후 q
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


def repo_root() -> Path:
    return Path(__file__).resolve().parent


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
    ap.add_argument("--weights", type=str, default="", help="비우면 runs/ 최신 best.pt")
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
    args = ap.parse_args()

    if args.list_cameras:
        list_cameras()
        return
    if args.list_realsense:
        list_realsense_devices()
        return

    if args.weights:
        wpath = Path(args.weights).resolve()
    else:
        found = find_best_pt(Path(args.project).resolve())
        if found is None:
            raise SystemExit(
                "best.pt 를 찾을 수 없습니다. --weights 로 경로를 지정하세요."
            )
        wpath = found

    if not wpath.is_file():
        raise SystemExit(f"가중치 없음: {wpath}")

    class_heights = load_class_heights(args.class_heights, args.data_yaml)
    dev = args.device or (0 if torch.cuda.is_available() else "cpu")
    use_half = (not args.no_half) and dev != "cpu" and torch.cuda.is_available()

    model = YOLO(str(wpath))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

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

    mode = "RealSense depth" if use_rs else "pinhole approx"
    print(f"가중치: {wpath}")
    print(f"device={dev}, 거리={mode}, class_heights={class_heights or '(웹캠만 해당)'}")
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
            )
            t1 = time.perf_counter()
            r0 = results[0]
            out = r0.plot()

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

            if r0.boxes is None or len(r0.boxes) == 0:
                cv2.imshow(window, out)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue

            boxes = r0.boxes.xyxy.cpu().numpy()
            clss = r0.boxes.cls.cpu().numpy().astype(int)

            line_y = 52
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

                pt = (int(round(cx_box)), int(round(cy_box)))
                cv2.circle(out, pt, 6, (0, 255, 255), -1, cv2.LINE_AA)

                norm_tip = np.linalg.norm([cx - cx_box, cy - cy_box]) + 1e-6
                tip = (
                    int(round(cx + 40 * (cx - cx_box) / norm_tip)),
                    int(round(cy + 40 * (cy - cy_box) / norm_tip)),
                )
                cv2.arrowedLine(out, pt, tip, (255, 180, 0), 2, cv2.LINE_AA, tipLength=0.25)

                z_show = f"{z_m:.3f}m" if not math.isnan(z_m) else "nan"
                ntxt = (
                    f"{label}  c=({cx_box:.0f},{cy_box:.0f})px  "
                    f"Z={z_show}  N=[{normal[0]:+.2f},{normal[1]:+.2f},{normal[2]:+.2f}]"
                )
                cv2.putText(
                    out,
                    ntxt,
                    (10, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (220, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
                line_y += 18

                if use_rs:
                    print(
                        f"[{label}] centroid_px=({cx_box:.1f},{cy_box:.1f}) "
                        f"dist_m(RealSense median)={z_m}  "
                        f"normal_cam={normal[0]:+.4f},{normal[1]:+.4f},{normal[2]:+.4f}"
                    )
                else:
                    print(
                        f"[{label}] centroid_px=({cx_box:.1f},{cy_box:.1f}) "
                        f"dist_m~(pinhole)={z_m:.3f} (H_ref={H_m}m, h_px={bh:.1f}) "
                        f"normal_cam={normal[0]:+.4f},{normal[1]:+.4f},{normal[2]:+.4f}"
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
