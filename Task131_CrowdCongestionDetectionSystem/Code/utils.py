import os
import cv2
import glob
import numpy as np

# Color definitions for density levels
DENSITY_COLORS = {
    "LOW": (0, 255, 0),       # Green
    "MODERATE": (0, 255, 255), # Yellow
    "HIGH": (0, 165, 255),     # Orange
    "CRITICAL": (0, 0, 255)    # Red
}

def video_frame_generator(video_path):
    """
    Opens a video file or a directory containing sequential images, and yields frames 
    sequentially along with sequence metadata.
    
    Args:
        video_path (str): Path to the input video file or image sequence directory.
        
    Yields:
        tuple: (frame, frame_idx, fps, frame_count, width, height)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input path not found at: {video_path}")
        
    if os.path.isdir(video_path):
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(video_path, ext)))
            image_files.extend(glob.glob(os.path.join(video_path, ext.upper())))
            
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            raise FileNotFoundError(f"No image files found in directory: {video_path}")
            
        frame_count = len(image_files)
        first_frame = cv2.imread(image_files[0])
        if first_frame is None:
            raise IOError(f"Could not read the first image frame: {image_files[0]}")
        height, width = first_frame.shape[:2]
        fps = 30.0  # Default fallback FPS for image sequences
        
        for frame_idx, img_path in enumerate(image_files):
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"⚠️ Warning: Could not read frame image: {img_path}")
                continue
            yield frame, frame_idx, fps, frame_count, width, height
            
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"OpenCV was unable to open the video file at: {video_path}")
            
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps is None:
                fps = 10.0  # Fallback FPS
                
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame, frame_idx, fps, frame_count, width, height
                frame_idx += 1
        finally:
            cap.release()

def calculate_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1 (list or tuple): [x1, y1, x2, y2] bounding box coordinates.
        box2 (list or tuple): [x1, y1, x2, y2] bounding box coordinates.
        
    Returns:
        float: IoU value between 0.0 and 1.0.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
        
    intersection_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0.0:
        return 0.0
        
    return intersection_area / union_area

def is_point_in_polygon(point, polygon):
    """
    Checks if a point (x, y) is inside a polygon using OpenCV's pointPolygonTest.
    
    Args:
        point (tuple or list): (x, y) coordinates of the point.
        polygon (list of list/tuple): Vertices of the polygon [[x1, y1], [x2, y2], ...].
        
    Returns:
        bool: True if point is inside or on the boundary of the polygon, False otherwise.
    """
    poly_array = np.array(polygon, dtype=np.int32)
    # cv2.pointPolygonTest expects a contour of shape (N, 1, 2) or (N, 2)
    # returns positive value if inside, negative value if outside, zero if on edge
    dist = cv2.pointPolygonTest(poly_array, (float(point[0]), float(point[1])), False)
    return dist >= 0
