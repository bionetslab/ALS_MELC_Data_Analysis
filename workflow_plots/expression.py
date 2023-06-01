from segmentation import MELC_Segmentation
import os 

class MELC_expression(MELC_Segmentation):
    def __init__(self, data_path, membrane_marker="cd45") -> None:
        super().__init__(data_path, membrane_marker)
        self._markers = None
        
    @property
    def markers(self):
        if self._markers is None:
            self._markers = [m.split("_")[1] for m in os.listdir(self.get_fov_dir()) if m.endswith(".tif") and "phase" not in m]    
        return self._markers
    
    @markers.setter
    def markers(self, markers):
        print("Please specify a field of fiew to automatically retrieve available markers. Markers cannot be set manually.")
        
        
        
    
        
    
    