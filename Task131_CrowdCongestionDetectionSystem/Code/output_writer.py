import os
import cv2
import pandas as pd
import numpy as np
from utils import DENSITY_COLORS

class OutputWriter:
    def __init__(self, video_output_path, csv_output_path, fps, width, height):
        """
        Manages the writing of the annotated output video and the CSV telemetry logs.
        
        Args:
            video_output_path (str): File path for the output MP4 video.
            csv_output_path (str): File path for the output CSV telemetry.
            fps (float): Frame rate of the video.
            width (int): Frame width.
            height (int): Frame height.
        """
        self.video_output_path = video_output_path
        self.csv_output_path = csv_output_path
        self.fps = fps
        self.width = width
        self.height = height
        
        # Create output directories if needed
        os.makedirs(os.path.dirname(os.path.abspath(self.video_output_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.csv_output_path)), exist_ok=True)
        
        # Initialize video writer (try avc1/H.264 first, fallback to mp4v)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.out_writer = cv2.VideoWriter(self.video_output_path, fourcc, self.fps, (self.width, self.height))
        if not self.out_writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out_writer = cv2.VideoWriter(self.video_output_path, fourcc, self.fps, (self.width, self.height))
        
        # Initialize CSV log file with header
        with open(self.csv_output_path, 'w') as f:
            f.write("timestamp,frame_number,zone,crowd_count,congestion_level,is_congested\n")
            
    def write_telemetry(self, timestamp_str, frame_idx, zone_states):
        """
        Appends crowd telemetry to the CSV file.
        """
        with open(self.csv_output_path, 'a') as f:
            for name, state in zone_states.items():
                is_congested_val = 1 if state["state"] == "CONGESTED" else 0
                f.write(f"{timestamp_str},{frame_idx},{name},{state['current_count']},{state['current_density']},{is_congested_val}\n")
                
    def annotate_frame(self, frame, tracked_persons, zone_states, zones):
        """
        Draws premium annotations on the frame:
        - Bounding boxes with IDs.
        - Semi-transparent filled zone polygons.
        - HUD overlay displaying status for all zones.
        - System alert banner if any zone is congested.
        
        Args:
            frame (numpy.ndarray): The raw BGR frame.
            tracked_persons (dict): Active tracked persons in the current frame.
            zone_states (dict): Current states of all zones.
            zones (list): List of zone geometries.
            
        Returns:
            numpy.ndarray: Annotated frame.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # 1. Draw Semi-Transparent Zone Polygons
        overlay = annotated.copy()
        for zone in zones:
            name = zone["name"]
            polygon = np.array(zone["polygon"], dtype=np.int32)
            state = zone_states[name]
            density = state["current_density"]
            color = DENSITY_COLORS.get(density, (0, 255, 0))
            
            # Fill polygon semi-transparently
            cv2.fillPoly(overlay, [polygon], color)
            
            # Draw boundary line
            thickness = 3 if state["state"] == "CONGESTED" else 2
            cv2.polylines(annotated, [polygon], True, color, thickness)
            
            # Add a small label in the center/centroid of the zone for readability
            moments = cv2.moments(polygon)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                label = f"{name}"
                cv2.putText(annotated, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                cv2.putText(annotated, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
        # Blend the filled polygon overlay
        alpha = 0.15  # subtle transparency
        cv2.addWeighted(overlay, alpha, annotated, 1.0 - alpha, 0, annotated)
        
        # 2. Draw Person Bounding Boxes
        for tid, track in tracked_persons.items():
            bbox = track["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bbox
            box_color = (255, 200, 0) # Sleek blue/cyan
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            
            # Centroid dot
            cx = (x1 + x2) // 2
            cy = y2
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
            
            # Bbox label (ID)
            label = f"ID {tid}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated, (x1, y1 - lh - 6), (x1 + lw + 6, y1), box_color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            
        # 3. HUD Overlay (Top-Right Panel)
        hud_w, hud_h = 240, 30 + len(zones) * 30
        hud_x, hud_y = w - hud_w - 15, 15
        
        # Semi-transparent background
        hud_bg = annotated.copy()
        cv2.rectangle(hud_bg, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (30, 30, 30), -1)
        cv2.addWeighted(hud_bg, 0.8, annotated, 0.2, 0, annotated)
        cv2.rectangle(annotated, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (100, 100, 100), 1)
        
        # HUD Header
        cv2.putText(annotated, "ZONE LIVE MONITOR", (hud_x + 10, hud_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(annotated, (hud_x + 10, hud_y + 25), (hud_x + hud_w - 10, hud_y + 25), (100, 100, 100), 1)
        
        # HUD Rows
        for i, zone in enumerate(zones):
            name = zone["name"]
            state = zone_states[name]
            count = state["current_count"]
            density = state["current_density"]
            color = DENSITY_COLORS.get(density, (255, 255, 255))
            
            # Shorten name if too long
            short_name = name[:15]
            row_y = hud_y + 45 + i * 25
            
            # Draw indicator circle
            cv2.circle(annotated, (hud_x + 20, row_y - 4), 6, color, -1)
            
            # Zone info text
            txt = f"{short_name}: {count} ({density})"
            cv2.putText(annotated, txt, (hud_x + 35, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
            
            # Highlight if congested
            if state["state"] == "CONGESTED":
                cv2.putText(annotated, "⚠️ CONGESTED", (hud_x + 155, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
                
        # 4. Global Alert Banner (Top-Center) if any zone is congested
        any_congested = any(s["state"] == "CONGESTED" for s in zone_states.values())
        if any_congested:
            banner_h = 35
            banner_bg = annotated.copy()
            cv2.rectangle(banner_bg, (0, 0), (w, banner_h), (0, 0, 220), -1) # Sleek Red
            cv2.addWeighted(banner_bg, 0.85, annotated, 0.15, 0, annotated)
            
            alert_text = "⚠️ CRITICAL SYSTEM WARNING: ACTIVE CROWD CONGESTION DETECTED"
            (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.putText(annotated, alert_text, ((w - tw) // 2, (banner_h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
        return annotated
        
    def write_frame(self, frame):
        """Writes the annotated frame to the output video."""
        self.out_writer.write(frame)
        
    def close(self):
        """Releases the video writer resource."""
        if self.out_writer is not None:
            self.out_writer.release()
            self.out_writer = None
