import cv2
import time
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from .models import YOLOWorker, DepthWorker, VLMWorker
from .aggregator import VisionAggregator
from .model_manager import model_manager

logger = logging.getLogger("VisionPipeline")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WEIGHTS DIRECTORY (relative to project root)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "weights")


def _w(filename: str) -> str:
    """Resolve a weight filename to its full path."""
    return os.path.join(WEIGHTS_DIR, filename)


class VisionPipeline:
    """
    Multi-model vision pipeline for real-time and recorded video analysis.

    Runs 4 YOLO models + 1 Depth model + 1 VLM concurrently per frame.
    The VLM only runs on keyframes (every N frames) to avoid bottlenecking.
    """

    def __init__(self, source=0, grid_width=100.0, grid_height=100.0):
        self.source = source
        self.cap = None
        self.running = False
        self.mode = "idle"  # idle | live | video | batch

        self.aggregator = VisionAggregator(grid_width, grid_height)

        # ━━━ YOLO Workers (using correct downloaded weights) ━━━
        self.workers = {
            "crowd": YOLOWorker(_w("yolov8n.pt"), "crowd", classes=[0]),
            "traffic": YOLOWorker(_w("yolov8s.pt"), "traffic", classes=[2, 3, 5, 7]),
            "anomaly": YOLOWorker(_w("yolov8m.pt"), "anomaly"),
            "flow": YOLOWorker(_w("yolov8n-seg.pt"), "flow", classes=[0]),
        }

        # ━━━ Depth Worker ━━━
        self.depth_worker = DepthWorker(_w("midas_v21_small.pt"))

        # ━━━ VLM Worker (SmolVLM2 2.2B Q4 on GPU) ━━━
        self.vlm_worker = VLMWorker(
            model_path=_w("SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"),
            mmproj_path=_w("mmproj-SmolVLM2-2.2B-Instruct-f16.gguf"),
            n_gpu_layers=-1,
        )
        self.vlm_keyframe_interval = 30  # Run VLM every 30 frames

        # Thread pool for concurrent inference
        self.executor = ThreadPoolExecutor(max_workers=7)

        # Callbacks
        self.on_agents_detected: Optional[Callable] = None
        self.on_vision_update: Optional[Callable] = None  # For dashboard metrics

        # Metrics
        self.fps = 0.0
        self.frame_count = 0
        self.total_frames = 0  # For video files
        self.last_vlm_analysis = None

        # Batch queue
        self._video_queue: list = []
        self._current_video_name = ""

    def initialize(self):
        """Loads models into memory. Failures are skipped, not fatal —
        the pipeline runs with whatever loaded successfully."""
        logger.info("━━━ Initializing Vision Pipeline ━━━")
        loaded, failed = [], []

        candidates = list(self.workers.items())
        candidates.append(("depth", self.depth_worker))
        candidates.append(("vlm", self.vlm_worker))

        for name, worker in candidates:
            if worker.loaded:
                loaded.append(name)
                continue
            try:
                worker.load()
                loaded.append(name)
            except Exception as e:
                logger.warning(f"Skipping model '{name}': {e}")
                failed.append(name)

        logger.info(f"━━━ Loaded: {loaded or 'none'} | Skipped: {failed or 'none'} ━━━")
        return {"loaded": loaded, "failed": failed}

    def set_source(self, source):
        """Change the video source (0=webcam, or file path string)."""
        self.source = source

    async def start(self):
        """Start processing the current video source."""
        if self.running:
            return

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return

        self.running = True
        is_file = isinstance(self.source, str) and os.path.isfile(self.source)
        self.mode = "video" if is_file else "live"
        self._current_video_name = os.path.basename(self.source) if is_file else "Webcam"

        if is_file:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.total_frames = 0

        logger.info(f"Vision Pipeline started — mode={self.mode}, source={self.source}")

        # Auto-calibrate homography (simple scaling)
        frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        self.aggregator.auto_calibrate(
            frame=None,
            reference_points_img=[[0, 0], [frame_w, 0], [frame_w, frame_h], [0, frame_h]],
            reference_points_grid=[
                [0, 0],
                [self.aggregator.grid_width, 0],
                [self.aggregator.grid_width, self.aggregator.grid_height],
                [0, self.aggregator.grid_height],
            ],
        )

        self.frame_count = 0
        fps_timer = time.time()
        fps_frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.mode == "video":
                    logger.info(f"Video '{self._current_video_name}' analysis complete.")
                    self.running = False
                    break
                else:
                    await asyncio.sleep(0.5)
                    continue

            self.frame_count += 1
            fps_frame_count += 1
            start_t = time.time()

            # ━━━ Dispatch YOLO + Depth concurrently ━━━
            loop = asyncio.get_running_loop()
            yolo_tasks = [
                loop.run_in_executor(self.executor, worker.predict, frame)
                for worker in self.workers.values()
            ]
            depth_task = loop.run_in_executor(self.executor, self.depth_worker.predict, frame)

            # ━━━ VLM on keyframes only ━━━
            vlm_task = None
            if self.frame_count % self.vlm_keyframe_interval == 0:
                vlm_task = loop.run_in_executor(self.executor, self.vlm_worker.predict, frame)

            # Wait for YOLO + Depth
            yolo_results_raw = await asyncio.gather(*yolo_tasks)
            depth_map = await depth_task

            # Wait for VLM if it was dispatched
            vlm_analysis = None
            if vlm_task is not None:
                vlm_analysis = await vlm_task
                if vlm_analysis:
                    self.last_vlm_analysis = vlm_analysis

            # Flatten YOLO results
            all_detections = []
            for r in yolo_results_raw:
                all_detections.extend(r)

            # Aggregate
            real_agents = self.aggregator.aggregate(
                self.frame_count, all_detections, depth_map, vlm_analysis
            )

            # Inject into simulation
            if self.on_agents_detected:
                self.on_agents_detected(real_agents)

            # Calculate FPS
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self.fps = round(fps_frame_count / elapsed, 1)
                fps_frame_count = 0
                fps_timer = time.time()

            inference_ms = round((time.time() - start_t) * 1000, 1)

            # Build detection summary for dashboard
            det_summary = {}
            for det in all_detections:
                task = det["task"]
                det_summary[task] = det_summary.get(task, 0) + 1

            # Push vision update to frontend via callback
            if self.on_vision_update:
                self.on_vision_update({
                    "fps": self.fps,
                    "latency_ms": inference_ms,
                    "frame": self.frame_count,
                    "total_frames": self.total_frames,
                    "mode": self.mode,
                    "detections": det_summary,
                    "total_detections": len(all_detections),
                    "vlm_analysis": self.last_vlm_analysis,
                    "model_manager": model_manager.get_status(),
                })

            # Yield control
            await asyncio.sleep(0.005)

        # Cleanup
        if self.cap:
            self.cap.release()
            self.cap = None
        self.mode = "idle"

    async def scan_videos(self, paths: list):
        """
        Batch scan multiple video files sequentially.
        `paths` can contain file paths and/or folder paths.
        """
        self._video_queue = []
        for p in paths:
            if os.path.isdir(p):
                for f in sorted(os.listdir(p)):
                    if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm')):
                        self._video_queue.append(os.path.join(p, f))
            elif os.path.isfile(p):
                self._video_queue.append(p)

        logger.info(f"Batch scan: {len(self._video_queue)} videos queued.")
        self.mode = "batch"

        for i, video_path in enumerate(self._video_queue):
            logger.info(f"Processing video {i + 1}/{len(self._video_queue)}: {video_path}")
            self.set_source(video_path)
            await self.start()
            # Small pause between videos
            await asyncio.sleep(0.1)

        self.mode = "idle"
        logger.info("Batch scan complete.")

    def stop(self):
        """Stop the pipeline."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.mode = "idle"
        logger.info("Vision Pipeline stopped.")
