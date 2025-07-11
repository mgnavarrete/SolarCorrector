# utils/config.py

import os

class Config:
    MODEL_PATH = os.path.join("utils", "PANEL-SEG-v1.pt")

    @staticmethod
    def get_project_paths(path_root):
        # Agrega la barra final si no está incluida
        path_root = f"{path_root}PP"
    
        return {
            "PP": path_root,
            "original_images": f"{path_root}/original_img",
            "cvat_images": f"{path_root}/cvat",
            "lines_images": f"{path_root}/lines",
            "metadata_lines": f"{path_root}/metadata_lines",
            "geonp": f"{path_root}/georef_numpy",
            "metadata": f"{path_root}/metadata",
            "masks": f"{path_root}/masks",
            "segmented_images": f"{path_root}/segmented_images",
            "json_path": f"{path_root}/corrector_data.json",
            "flights_json": f"{path_root}/list_flights.json",
            
        }


