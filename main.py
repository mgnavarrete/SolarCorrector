from solar_corrector import SolarCorrector
from core.file_manager import FileManager

if __name__ == "__main__":

    folders_list = FileManager().select_directories()
    
    for folder in folders_list:
        solarCorrection = SolarCorrection(folder)
        solarCorrection.findFlights()
        solarCorrection.get_seg_paneles()
    