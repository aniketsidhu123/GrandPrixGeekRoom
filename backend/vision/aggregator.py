import cv2
import numpy as np

class VisionAggregator:
    def __init__(self, grid_width: float, grid_height: float):
        """
        Aggregates outputs from multiple YOLO models and a Depth model.
        Maps pixel coordinates to Simulation Grid coordinates using Homography.
        """
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.homography_matrix = None
        self.is_calibrated = False

    def auto_calibrate(self, frame: np.ndarray, reference_points_img=None, reference_points_grid=None):
        """
        Calculates the Homography matrix. 
        In a real scenario, you'd auto-detect court lines or use predefined reference points.
        For now, we accept manual points or fallback to a default mapping.
        """
        if reference_points_img is not None and reference_points_grid is not None:
            # Need at least 4 points to compute homography
            pts_src = np.array(reference_points_img, dtype=float)
            pts_dst = np.array(reference_points_grid, dtype=float)
            h, status = cv2.findHomography(pts_src, pts_dst)
            self.homography_matrix = h
            self.is_calibrated = True
            print("VisionAggregator: Homography matrix calibrated.")
        else:
            print("VisionAggregator: No points provided, calibration failed.")
            self.is_calibrated = False

    def map_to_grid(self, pixel_x: float, pixel_y: float, depth_value: float = None):
        """
        Converts (x, y) pixel coordinates (bottom center of bounding box) to grid coordinates.
        If depth_value is provided, it can be used to refine the 3D projection.
        """
        if not self.is_calibrated or self.homography_matrix is None:
            # Fallback: simple linear scaling if not calibrated (highly inaccurate for perspective)
            # Assuming frame is 1920x1080 for fallback
            return (pixel_x / 1920.0) * self.grid_width, (pixel_y / 1080.0) * self.grid_height

        pts = np.array([[[pixel_x, pixel_y]]], dtype="float32")
        dst = cv2.perspectiveTransform(pts, self.homography_matrix)
        grid_x, grid_y = dst[0][0][0], dst[0][0][1]
        
        # Ensure it's within grid bounds
        grid_x = max(0.0, min(float(grid_x), self.grid_width))
        grid_y = max(0.0, min(float(grid_y), self.grid_height))
        
        return grid_x, grid_y

    def aggregate(self, frame_id: int, yolo_results: list, depth_map: np.ndarray):
        """
        Takes raw YOLO detections and depth map, and produces the unified real-agent state.
        yolo_results format: List of detections from all models.
        """
        real_agents = []
        for det in yolo_results:
            px, py = det["center"]
            
            # Sample depth map at the feet
            d_val = None
            if depth_map is not None:
                # bounds check
                dh, dw = depth_map.shape
                s_x = min(max(int(px), 0), dw - 1)
                s_y = min(max(int(py), 0), dh - 1)
                d_val = float(depth_map[s_y, s_x])
            
            # Map to grid
            gx, gy = self.map_to_grid(px, py, d_val)
            
            real_agents.append({
                "id": f"real_{frame_id}_{len(real_agents)}", # Simple temporary ID. Use DeepSORT for real tracking
                "x": gx,
                "y": gy,
                "type": det["task"],
                "class_id": det["class"],
                "conf": det["conf"],
                "depth": d_val
            })
            
        return real_agents
