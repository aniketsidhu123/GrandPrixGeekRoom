import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("VisionAggregator")


class VisionAggregator:
    """
    Central Hub — Aggregates outputs from all YOLO models, Depth model, and VLM.
    Maps pixel coordinates to Simulation Grid coordinates using Homography.
    Restructures and compiles all data into a unified intelligence report.
    """

    def __init__(self, grid_width: float, grid_height: float):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.homography_matrix = None
        self.is_calibrated = False

        # Cached VLM analysis (persists between frames)
        self.latest_vlm_analysis: Optional[dict] = None

        # Detection history for trend analysis
        self._detection_counts_history = []
        self._max_history = 30  # Keep last 30 frames of counts

    def auto_calibrate(self, frame, reference_points_img=None, reference_points_grid=None):
        """
        Calculates the Homography matrix for perspective transform.
        Requires at least 4 point pairs: image coords → grid coords.
        """
        if reference_points_img is not None and reference_points_grid is not None:
            pts_src = np.array(reference_points_img, dtype=np.float32)
            pts_dst = np.array(reference_points_grid, dtype=np.float32)
            h, status = cv2.findHomography(pts_src, pts_dst)
            self.homography_matrix = h
            self.is_calibrated = True
            logger.info("Homography matrix calibrated successfully.")
        else:
            logger.warning("No reference points provided — calibration skipped.")
            self.is_calibrated = False

    def map_to_grid(self, pixel_x: float, pixel_y: float, depth_value: float = None):
        """
        Converts pixel (x, y) to grid coordinates using homography.
        Falls back to simple linear scaling if not calibrated.
        """
        if not self.is_calibrated or self.homography_matrix is None:
            return (pixel_x / 1920.0) * self.grid_width, (pixel_y / 1080.0) * self.grid_height

        pts = np.array([[[pixel_x, pixel_y]]], dtype="float32")
        dst = cv2.perspectiveTransform(pts, self.homography_matrix)
        grid_x, grid_y = dst[0][0][0], dst[0][0][1]

        # Clamp to grid bounds
        grid_x = max(0.0, min(float(grid_x), self.grid_width - 0.1))
        grid_y = max(0.0, min(float(grid_y), self.grid_height - 0.1))

        return grid_x, grid_y

    def aggregate(self, frame_id: int, yolo_results: list, depth_map: np.ndarray,
                  vlm_analysis: dict = None):
        """
        The Central Hub function.
        Takes raw detections from ALL YOLO models + depth map + VLM analysis,
        restructures and compiles them into unified real-agent data.
        """
        # ━━━ 1. Update VLM analysis if new one arrived ━━━
        if vlm_analysis is not None:
            self.latest_vlm_analysis = vlm_analysis

        # ━━━ 2. Process YOLO detections ━━━
        real_agents = []
        detection_counts = {}

        for det in yolo_results:
            task = det["task"]
            detection_counts[task] = detection_counts.get(task, 0) + 1

            px, py = det["center"]

            # Sample depth map at the detection point
            d_val = None
            if depth_map is not None:
                dh, dw = depth_map.shape
                s_x = min(max(int(px), 0), dw - 1)
                s_y = min(max(int(py), 0), dh - 1)
                d_val = float(depth_map[s_y, s_x])

            # Map to grid coordinates
            gx, gy = self.map_to_grid(px, py, d_val)

            real_agents.append({
                "id": f"real_{frame_id}_{len(real_agents)}",
                "x": round(gx, 2),
                "y": round(gy, 2),
                "type": task,
                "class_id": det["class"],
                "class_name": det.get("class_name", ""),
                "conf": det["conf"],
                "depth": d_val,
            })

        # ━━━ 3. Track detection trends ━━━
        self._detection_counts_history.append(detection_counts)
        if len(self._detection_counts_history) > self._max_history:
            self._detection_counts_history.pop(0)

        return real_agents
