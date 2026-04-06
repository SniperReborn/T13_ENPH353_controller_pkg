import numpy as np

class LidarMotionDetector:
    def __init__(self, check_distance=10.0, min_diff=0.20, min_rays=3):
        self.check_distance = check_distance
        self.min_diff = min_diff
        self.min_rays = min_rays 
        self.baseline_ranges = None

    def reset_baseline(self):
        self.baseline_ranges = None

    def detect_movement(self, scan_data):
        current_ranges = np.array(scan_data.ranges)
        
        # Clean bad data
        bad_data = np.isinf(current_ranges) | np.isnan(current_ranges) | (current_ranges == 0.0)
        current_ranges[bad_data] = scan_data.range_max

        if self.baseline_ranges is None:
            self.baseline_ranges = current_ranges
            return False

        if len(current_ranges) != len(self.baseline_ranges):
            return False
            
        diff = np.abs(current_ranges - self.baseline_ranges)
        in_range = current_ranges < self.check_distance
        significant_change = diff > self.min_diff

        # Return True if enough rays detect movement
        return np.sum(in_range & significant_change) >= self.min_rays