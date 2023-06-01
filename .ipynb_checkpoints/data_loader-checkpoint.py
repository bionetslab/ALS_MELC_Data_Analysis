import os
import cv2 

class ALS_data_loader:
    def __init__(self, data_path) -> None:
        self._data_path = data_path
        self.field_of_views = sorted(os.listdir(data_path))
        
    def get_cd45(self, field_of_view):
        fov_dir = self.get_fov_dir(field_of_view)
        cd45_path = self.get_channel_path(fov_dir, "cd45-")
        return cv2.imread(cd45_path, cv2.IMREAD_GRAYSCALE)

    
    def get_prop_iodide(self, field_of_view):
        fov_dir = self.get_fov_dir(field_of_view)
        bleach_dir = self.get_bleach_dir(fov_dir)
        prop_iodide_path = self.get_channel_path(bleach_dir, "propidium")
        return cv2.imread(prop_iodide_path, cv2.IMREAD_GRAYSCALE)
    
    
    def get_fov_dir(self, field_of_view):
        fov_dir = os.path.join(self._data_path, field_of_view)
        assert os.path.isdir(fov_dir), f"Field of view {field_of_view} does not exist!"
        return fov_dir
    
    
    def get_bleach_dir(self, fov_dir):
        bleach_dir = os.path.join(fov_dir, "bleach")
        assert os.path.isdir(bleach_dir), f"Field of view {fov_dir} does not contain a bleach directory!"
        return bleach_dir
    
    
    def get_channel_path(self, img_dir, channel):
        channels = [c for c in os.listdir(img_dir) if channel.lower() in c.lower()]
        assert len(channels) == 1, f"Path for field of view {img_dir} does not contain exactly one image of the desired channel!"
        channel_path = os.path.join(img_dir, channels[0])
        return channel_path