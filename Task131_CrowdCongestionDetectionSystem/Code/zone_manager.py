import datetime
from utils import is_point_in_polygon

class ZoneManager:
    def __init__(self, zone_config, min_duration_seconds=5.0):
        """
        Manages spatial zones, person-to-zone assignment, density levels, 
        and congestion hotspot state transitions.
        
        Args:
            zone_config (list): List of zone dicts from config yaml.
            min_duration_seconds (float): Minimum consecutive seconds to trigger congestion.
        """
        self.zones = zone_config
        self.min_duration_seconds = min_duration_seconds
        
        # State tracking per zone
        self.zone_states = {}
        for zone in self.zones:
            name = zone["name"]
            self.zone_states[name] = {
                "state": "NORMAL",                    # "NORMAL" or "CONGESTED"
                "consec_congested_frames": 0,
                "start_frame_idx": None,
                "start_timestamp": None,
                "peak_count": 0,
                "current_count": 0,
                "current_density": "LOW"
            }
            
        # Callbacks that can be registered by the orchestrator
        self.on_congestion_start_callback = None
        self.on_congestion_end_callback = None
        
        # Complete historical list of congestion events
        self.congestion_history = []
        
    def register_callbacks(self, on_start, on_end):
        """Registers external callback functions for state transitions."""
        self.on_congestion_start_callback = on_start
        self.on_congestion_end_callback = on_end
        
    def update(self, tracked_persons, frame_idx, fps):
        """
        Assigns people to zones, calculates densities, and updates the congestion state machine.
        
        Args:
            tracked_persons (dict): Active tracks from the tracker.
            frame_idx (int): Current video frame index.
            fps (float): Video frames per second.
            
        Returns:
            dict: Current zone states with count, density, and congestion state.
        """
        required_frames = max(1, int(fps * self.min_duration_seconds))
        
        # Reset counts for the current frame
        zone_counts = {zone["name"]: 0 for zone in self.zones}
        
        # Map each tracked person to a zone using bottom-middle point
        for tid, track in tracked_persons.items():
            bbox = track["bbox"]
            # Bottom-middle point of the bounding box
            centroid_x = (bbox[0] + bbox[2]) / 2.0
            centroid_y = bbox[3]
            point = (centroid_x, centroid_y)
            
            for zone in self.zones:
                if is_point_in_polygon(point, zone["polygon"]):
                    zone_counts[zone["name"]] += 1
                    break  # Assign to the first matching zone
                    
        # Update density and state machine for each zone
        for zone in self.zones:
            name = zone["name"]
            count = zone_counts[name]
            thresholds = zone["thresholds"]
            
            # Determine density level
            if count >= thresholds.get("critical", 8):
                density_level = "CRITICAL"
            elif count >= thresholds.get("high", 5):
                density_level = "HIGH"
            elif count >= thresholds.get("moderate", 2):
                density_level = "MODERATE"
            else:
                density_level = "LOW"
                
            state_data = self.zone_states[name]
            state_data["current_count"] = count
            state_data["current_density"] = density_level
            
            is_high_density = (density_level in ["HIGH", "CRITICAL"])
            
            if is_high_density:
                state_data["consec_congested_frames"] += 1
                
                # Check for state transition: NORMAL -> CONGESTED
                if state_data["state"] == "NORMAL":
                    if state_data["consec_congested_frames"] >= required_frames:
                        state_data["state"] = "CONGESTED"
                        # Start frame is when the high density period began
                        state_data["start_frame_idx"] = frame_idx - required_frames + 1
                        state_data["start_timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        state_data["peak_count"] = count
                        
                        if self.on_congestion_start_callback:
                            self.on_congestion_start_callback(
                                name, 
                                state_data["start_timestamp"], 
                                count, 
                                density_level
                            )
                else:
                    # Keep track of peak count during the event
                    if count > state_data["peak_count"]:
                        state_data["peak_count"] = count
            else:
                # Check for state transition: CONGESTED -> NORMAL
                if state_data["state"] == "CONGESTED":
                    end_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    start_frame = state_data["start_frame_idx"]
                    duration = (frame_idx - start_frame) / fps
                    peak = state_data["peak_count"]
                    
                    event = {
                        "zone": name,
                        "start_timestamp": state_data["start_timestamp"],
                        "end_timestamp": end_timestamp,
                        "duration_seconds": float(round(duration, 2)),
                        "peak_count": int(peak)
                    }
                    self.congestion_history.append(event)
                    
                    if self.on_congestion_end_callback:
                        self.on_congestion_end_callback(
                            name, 
                            state_data["start_timestamp"], 
                            end_timestamp, 
                            duration, 
                            peak
                        )
                        
                # Reset counters
                state_data["state"] = "NORMAL"
                state_data["consec_congested_frames"] = 0
                state_data["start_frame_idx"] = None
                state_data["start_timestamp"] = None
                state_data["peak_count"] = 0
                
        return self.zone_states
