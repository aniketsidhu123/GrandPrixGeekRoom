import cv2
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from .models import YOLOWorker, DepthWorker
from .aggregator import VisionAggregator

logger = logging.getLogger("VisionPipeline")

class VisionPipeline:
    def __init__(self, source=0, grid_width=100.0, grid_height=100.0):
        self.source = source
        self.cap = None
        self.running = False
        
        self.aggregator = VisionAggregator(grid_width, grid_height)
        
        # Define models
        # Note: In a real environment, replace these with actual paths to trained weights (.pt/.engine)
        self.workers = {
            "crowd": YOLOWorker("yolov8n.pt", "crowd", classes=[0]), # 0 is usually 'person' in COCO
            "traffic": YOLOWorker("yolov8n.pt", "traffic", classes=[2, 3, 5, 7]), # cars, motorcycles, bus, truck
            "anomaly": YOLOWorker("yolov8n.pt", "anomaly"), # Placeholder for specialized model
            "flow": YOLOWorker("yolov8n.pt", "flow", classes=[0]) # Flow tracker placeholder
        }
        self.depth_worker = DepthWorker("MiDaS_small")
        
        # Thread pool for concurrent inference (GIL might bottleneck if using purely CPU, but OK for GPU/TensorRT)
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Callback for when new real agents are processed
        self.on_agents_detected = None

    def initialize(self):
        """Loads models into memory (VRAM). This takes time."""
        logger.info("Initializing Vision Pipeline models...")
        for name, worker in self.workers.items():
            worker.load()
        self.depth_worker.load()
        logger.info("Models initialized.")

    async def start(self):
        if self.running:
            return
            
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source {self.source}")
            return
            
        self.running = True
        logger.info("Vision Pipeline started.")
        
        # Dummy calibration for testing (mapping corners of 1080p to a 100x100 grid)
        self.aggregator.auto_calibrate(
            frame=None,
            reference_points_img=[[0,0], [1920,0], [1920,1080], [0,1080]],
            reference_points_grid=[[0,0], [100,0], [100,100], [0,100]]
        )

        frame_id = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Video stream ended or dropped frame.")
                await asyncio.sleep(1) # Wait before retry (if live stream)
                continue
                
            frame_id += 1
            start_t = time.time()
            
            # Dispatch to workers concurrently
            loop = asyncio.get_running_loop()
            tasks = []
            
            # 1. Run all YOLO models
            for name, worker in self.workers.items():
                tasks.append(loop.run_in_executor(self.executor, worker.predict, frame))
            
            # 2. Run Depth model
            depth_task = loop.run_in_executor(self.executor, self.depth_worker.predict, frame)
            
            # Wait for all inference to complete for this frame
            yolo_results_raw = await asyncio.gather(*tasks)
            depth_map = await depth_task
            
            # Flatten YOLO results
            all_detections = []
            for r in yolo_results_raw:
                all_detections.extend(r)
                
            # Aggregate and map to grid
            real_agents = self.aggregator.aggregate(frame_id, all_detections, depth_map)
            
            # Call the callback to inject into simulation
            if self.on_agents_detected:
                self.on_agents_detected(real_agents)
                
            inference_time = time.time() - start_t
            # logger.debug(f"Frame {frame_id} processed in {inference_time:.3f}s. Real agents: {len(real_agents)}")
            
            # Yield control (prevent choking FastAPI)
            await asyncio.sleep(0.01)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info("Vision Pipeline stopped.")
