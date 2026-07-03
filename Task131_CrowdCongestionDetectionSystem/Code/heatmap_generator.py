import os
import cv2
import numpy as np

class HeatmapGenerator:
    def __init__(self, output_dir="Outputs", colormap="COLORMAP_JET", alpha=0.6, blur_radius=25):
        """
        Accumulates person centroids and generates density heatmaps overlayed on video frames.
        
        Args:
            output_dir (str): Directory where heatmap images will be saved.
            colormap (str): OpenCV colormap name (e.g., COLORMAP_JET).
            alpha (float): Transparency blending factor for the overlay (0.0 to 1.0).
            blur_radius (int): Gaussian blur size for smoothing the accumulated points.
        """
        self.output_dir = output_dir
        self.alpha = alpha
        self.blur_radius = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
        
        # Resolve colormap attribute
        if hasattr(cv2, colormap):
            self.colormap_id = getattr(cv2, colormap)
        else:
            self.colormap_id = cv2.COLORMAP_JET
            
        self.accumulator = None
        self.reference_frame = None
        os.makedirs(self.output_dir, exist_ok=True)
        
    def update(self, tracked_persons, frame):
        """
        Accumulates person centroids in this frame.
        
        Args:
            tracked_persons (dict): Active tracks from the tracker.
            frame (numpy.ndarray): Current video frame to store as reference.
        """
        if self.accumulator is None:
            h, w = frame.shape[:2]
            self.accumulator = np.zeros((h, w), dtype=np.float32)
            
        # Store reference frame (usually the first frame or background)
        if self.reference_frame is None:
            self.reference_frame = frame.copy()
            
        # Accumulate bottom-middle coordinate of each tracked person
        for tid, track in tracked_persons.items():
            bbox = track["bbox"]
            cx = int((bbox[0] + bbox[2]) / 2.0)
            cy = int(bbox[3])
            
            # Bound within frame dimensions
            h, w = self.accumulator.shape
            cx = min(max(0, cx), w - 1)
            cy = min(max(0, cy), h - 1)
            
            # Accumulate a small dot. To make it smooth, we can increment pixels
            # in a small region, or just increment the single pixel and rely on 
            # Gaussian blur later. To make the heatmap look great even for short videos,
            # we draw a small solid circle of radius 5 in each update.
            cv2.circle(self.accumulator, (cx, cy), 8, 1.0, -1)
            
    def generate_heatmap(self, base_image=None):
        """
        Renders the accumulated density as a colormapped heatmap overlay.
        
        Args:
            base_image (numpy.ndarray): Base image to overlay the heatmap on. 
                                        If None, uses the stored reference frame.
                                        
        Returns:
            numpy.ndarray: Colormapped overlay image.
        """
        if self.accumulator is None or np.max(self.accumulator) == 0:
            # If no accumulation occurred, return base image
            return base_image if base_image is not None else self.reference_frame
            
        if base_image is None:
            base_image = self.reference_frame.copy()
            
        # Apply Gaussian blur to smooth the discrete accumulation points
        blurred = cv2.GaussianBlur(self.accumulator, (self.blur_radius, self.blur_radius), 0)
        
        # Normalize to 0-255 range
        max_val = np.max(blurred)
        if max_val > 0:
            normalized = (blurred / max_val * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(blurred, dtype=np.uint8)
            
        # Apply the colormap
        colored_heatmap = cv2.applyColorMap(normalized, self.colormap_id)
        
        # Create a transparency mask so zero-density areas are not colored blue/cool
        # Mask is based on the normalized values (0 is no density)
        mask = normalized > 5
        
        overlay = base_image.copy()
        # Blend the colored heatmap with the base image in regions where mask is True
        overlay[mask] = cv2.addWeighted(base_image, 1.0 - self.alpha, colored_heatmap, self.alpha, 0)[mask]
        
        return overlay
        
    def save_snapshot(self, filename, base_image=None):
        """
        Generates and saves a heatmap snapshot to the output directory.
        
        Args:
            filename (str): Name of the file to save.
            base_image (numpy.ndarray): Frame to overlay heatmap on.
            
        Returns:
            str: Path of the saved image.
        """
        heatmap_img = self.generate_heatmap(base_image)
        save_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(save_path, heatmap_img)
        return save_path
