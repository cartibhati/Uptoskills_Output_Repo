import os
import datetime

class AlertManager:
    def __init__(self, log_path="Outputs/alerts.log"):
        """
        Manages alert generation, logging, and external notification routing.
        
        Args:
            log_path (str): File path where alerts will be written.
        """
        self.log_path = log_path
        # Create output directories if needed
        log_dir = os.path.dirname(os.path.abspath(log_path))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        # Initialize the log file with a header if it doesn't exist
        if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
            with open(self.log_path, 'w') as f:
                f.write("timestamp,zone,event_type,count,duration_seconds\n")
                
    def handle_congestion_start(self, zone_name, timestamp, count, density):
        """
        Handles the start of a congestion event.
        """
        message = f"🚨 [ALERT START] Zone '{zone_name}' is CONGESTED! Count: {count}, Density Level: {density}"
        print(f"\n{message}")
        
        # Log to file
        with open(self.log_path, 'a') as f:
            f.write(f"{timestamp},{zone_name},START,{count},N/A\n")
            
        # Pluggable delivery
        self._send_external_alert(zone_name, "START", count, density=density)
        
    def handle_congestion_end(self, zone_name, start_timestamp, end_timestamp, duration, peak_count):
        """
        Handles the end of a congestion event.
        """
        message = f"✅ [ALERT END] Zone '{zone_name}' congestion resolved. Duration: {duration:.2f}s, Peak Count: {peak_count}"
        print(f"\n{message}")
        
        # Log to file
        with open(self.log_path, 'a') as f:
            f.write(f"{end_timestamp},{zone_name},END,{peak_count},{duration:.2f}\n")
            
        # Pluggable delivery
        self._send_external_alert(zone_name, "END", peak_count, duration=duration)
        
    def _send_external_alert(self, zone_name, event_type, count, density=None, duration=None):
        """
        Pluggable hook for external alert delivery (e.g., Telegram, Slack, Webhook).
        This can be wired to active API endpoints.
        """
        # Hook demonstration output (Console Stub)
        if event_type == "START":
            # For developers to easily wire up later:
            # send_telegram_message(f"ALERT: {zone_name} congested with {count} people.")
            print(f"🔗 [Webhook Hook] Dispatching START alert payload for '{zone_name}' to external notification endpoint...")
        elif event_type == "END":
            # send_slack_message(f"RESOLVED: {zone_name} clear. Dwell: {duration}s.")
            print(f"🔗 [Webhook Hook] Dispatching END alert payload for '{zone_name}' to external notification endpoint...")
