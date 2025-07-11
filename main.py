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
        puntos = [
    (-70.7951126851, -33.0957588996, 0),
    (-70.7950894205, -33.0957592347, 0),
    (-70.7950732533, -33.0949640250, 0),
    (-70.7950965177, -33.0949636899, 0),
    (-70.7951126851, -33.0957588996, 0)
]
        SC.correct_yaw(save_images=True, puntos=puntos)
      