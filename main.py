from solar_corrector import SolarCorrector
from utils.folder_utils import select_directories


if __name__ == "__main__":

    folders_list = select_directories()
    
    for folder in folders_list:
        SC = SolarCorrector(folder, cvat_images=False)
        SC.init_from_json()
        SC.reset_metadata(var='all')    
        SC.findFlights(4, save_kml=True)
        SC.get_seg_paneles(save_masks=True)
        SC.correct_yaw(save_images=True, save_kml=False)
        SC.correct_E(save_images=False, save_kml=True)
      