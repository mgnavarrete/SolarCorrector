from solar_corrector import SolarCorrector
from core.file_manager import FileManager

if __name__ == "__main__":

    folders_list = FileManager().select_directories()
    
    for folder in folders_list:
        SC = SolarCorrector(folder)
        SC.findFlights(3)
        SC.get_seg_paneles(save_masks=True)
    