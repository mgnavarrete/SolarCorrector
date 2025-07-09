from solar_corrector import SolarCorrector
from core.folder_manager import FolderManager
from core.metadata_manager import MetadataManager
from collections import Counter
import numpy as np

if __name__ == "__main__":

    folders_list = FolderManager.select_directories()
    
    for folder in folders_list:
        SC = SolarCorrector(folder)
        #SC.reset_metadata(var='all')    
        #SC.save_geo_matrix()
        #SC.findFlights(4, save_kml=False)
        #SC.get_seg_paneles(save_masks=True)
        SC.correct_E()
      