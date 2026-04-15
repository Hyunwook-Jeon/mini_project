#!/usr/bin/env python3
"""
호환 래퍼 파일.

요청한 파일명 `yolo_live_cam_3d_metricx.py`로 실행해도
`yolo_live_cam_3d_metrics.py`의 최신 구현(사분면 기반 X축 생성 포함)을 사용한다.
"""

from yolo_live_cam_3d_metrics import main


if __name__ == "__main__":
    main()
