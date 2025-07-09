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
        SC.findFlights(4, save_kml=False)
        SC.get_seg_paneles(save_masks=True)
        
        areas = SC.list_areas
        
        # Sacar Media de areas
        media_areas = sum(areas) / len(areas)
        print(f"Media de areas: {media_areas}")
        
        # Sacar Moda de areas
        moda_areas = max(set(areas), key=areas.count)
        print(f"Moda de areas: {moda_areas}")

        # Paso 1: Obtener el mínimo y máximo
        minimo = min(areas)
        maximo = max(areas)
        
        # Paso 2: Crear los bordes de los intervalos
        bordes = np.arange(minimo, maximo + 0.5, 0.5)
        
        # Paso 3: Agrupar valores en intervalos
        intervalos = np.digitize(areas, bordes, right=False)
        
        # Paso 4: Contar la frecuencia de cada intervalo
        frecuencias = Counter(intervalos)
        
        # Paso 5: Encontrar el intervalo más frecuente
        intervalo_mas_frecuente = max(frecuencias, key=frecuencias.get)
        
        # Paso 6: Determinar los bordes del intervalo moda
        moda_inferior = bordes[intervalo_mas_frecuente - 1]
        moda_superior = bordes[intervalo_mas_frecuente]

        print(f"Moda aproximada está en el intervalo: [{moda_inferior}, {moda_superior})")
        print(f"Frecuencia en ese intervalo: {frecuencias[intervalo_mas_frecuente]}")
            
        
    