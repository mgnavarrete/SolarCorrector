from solar_corrector import SolarCorrector
from core.folder_manager import FolderManager
from core.metadata_manager import MetadataManager

if __name__ == "__main__":

    folders_list = FolderManager.select_directories()
    
    for folder in folders_list:
        SC = SolarCorrector(folder)
        SC.reset_metadata(var='all')    
        SC.save_geo_matrix()
        SC.findFlights(3)
        SC.get_seg_paneles(save_masks=True)
    