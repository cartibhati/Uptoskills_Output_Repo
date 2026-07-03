from utils import calculate_iou

class SimpleIoUTracker:
    def __init__(self, iou_threshold=0.3, max_lost_frames=30):
        """
        A lightweight IoU-based tracker for matching detections across frames.
        
        Args:
            iou_threshold (float): Minimum IoU overlap to consider a match.
            max_lost_frames (int): Number of consecutive frames a track can be missing before deletion.
        """
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_id = 1
        self.tracked_persons = {}
        
    def update(self, detections, frame_idx):
        """
        Updates the active tracks with new detections from the current frame.
        
        Args:
            detections (list): List of detections, each is {"bbox": [x1, y1, x2, y2], "confidence": float}
            frame_idx (int): Current frame index.
            
        Returns:
            dict: Currently active tracks visible in this frame.
        """
        active_ids = list(self.tracked_persons.keys())
        matches = []
        
        # Calculate IoU between all current detections and existing tracked persons
        for det_idx, det in enumerate(detections):
            det_bbox = det["bbox"]
            for track_id in active_ids:
                track_bbox = self.tracked_persons[track_id]["bbox"]
                iou = calculate_iou(det_bbox, track_bbox)
                if iou >= self.iou_threshold:
                    matches.append((iou, det_idx, track_id))
                    
        # Sort matches by IoU in descending order
        matches.sort(key=lambda x: x[0], reverse=True)
        
        matched_det_indices = set()
        matched_track_ids = set()
        
        for iou, det_idx, track_id in matches:
            if det_idx in matched_det_indices or track_id in matched_track_ids:
                continue
                
            matched_det_indices.add(det_idx)
            matched_track_ids.add(track_id)
            
            # Update tracked person details
            det = detections[det_idx]
            track_data = self.tracked_persons[track_id]
            track_data["bbox"] = det["bbox"]
            track_data["confidence"] = det["confidence"]
            track_data["lost_frames"] = 0
            track_data["last_seen_frame"] = frame_idx
            
            centroid_x = (det["bbox"][0] + det["bbox"][2]) / 2.0
            centroid_y = (det["bbox"][1] + det["bbox"][3]) / 2.0
            track_data["centroid_history"].append((centroid_x, centroid_y, frame_idx))
            
            if len(track_data["centroid_history"]) > 100:
                track_data["centroid_history"].pop(0)
                
        # Register new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_indices:
                centroid_x = (det["bbox"][0] + det["bbox"][2]) / 2.0
                centroid_y = (det["bbox"][1] + det["bbox"][3]) / 2.0
                
                self.tracked_persons[self.next_id] = {
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "centroid_history": [(centroid_x, centroid_y, frame_idx)],
                    "lost_frames": 0,
                    "last_seen_frame": frame_idx
                }
                self.next_id += 1
                
        # Handle lost tracks
        dead_tracks = []
        for track_id in active_ids:
            if track_id not in matched_track_ids:
                self.tracked_persons[track_id]["lost_frames"] += 1
                if self.tracked_persons[track_id]["lost_frames"] > self.max_lost_frames:
                    dead_tracks.append(track_id)
                    
        # Remove dead tracks
        for track_id in dead_tracks:
            del self.tracked_persons[track_id]
            
        # Return tracks present in the current frame
        return {
            tid: data for tid, data in self.tracked_persons.items() 
            if data["last_seen_frame"] == frame_idx
        }
