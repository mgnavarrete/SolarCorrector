import os
from tqdm import tqdm
import json


class MetadataManager:

    def reset_all_metadata(self, list_images, metadata_path, var: str = 'all'):
        # Resetea los metadatos de las imágenes
        # var: all, E, yaw, H
        # all: resetea todos los metadatos
        
        if var not in ['all', 'E', 'yaw', 'H']:
            raise ValueError(f"El parámetro {var} no es válido, los parámetros válidos son: all, E, yaw, H")
        
        else:
            for image_path in tqdm(list_images, desc="Reset Metadata:"):
                try:
                    # Carga la metadata de la imagen
                    metadata_file = f'{metadata_path}/{image_path[:-4]}.txt'
                    
                    if not os.path.exists(metadata_file):
                        print(f"Advertencia: Archivo de metadatos no encontrado: {metadata_file}")
                        continue
                        
                    with open(metadata_file, 'r') as archivo:
                        data = json.load(archivo)
                        
                    if var == 'all':
                        data['offset_E'] = 0
                        data['offset_altura'] = 0
                        data['offset_E_tot'] = 0
                        data['offset_yaw'] = 0
                        
                    elif var == 'E':
                        data['offset_E'] = 0
                        data['offset_E_tot'] = 0
                        
                    elif var == 'yaw':
                        data['offset_yaw'] = 0
                        
                    elif var == 'H':
                        data['offset_altura'] = 0
                        
                    with open(metadata_file, 'w') as f:
                        json.dump(data, f, indent=4)
                        
                except FileNotFoundError:
                    print(f"Error: No se pudo encontrar el archivo de metadatos: {metadata_file}")
                except json.JSONDecodeError as e:
                    print(f"Error: JSON inválido en el archivo {metadata_file}: {e}")
                except PermissionError:
                    print(f"Error: No hay permisos para acceder al archivo: {metadata_file}")
                except Exception as e:
                    print(f"Error inesperado al procesar {image_path}: {e}")
                
    def adjust_metadata(self, list_images, metadata_path, param: str, value: float):
        # Ajusta los metadatos de las imágenes
        # param: offset_E, offset_altura, offset_yaw
        # value: float
        
        if param not in ['offset_E', 'offset_altura', 'offset_yaw']:
            raise ValueError(f"El parámetro {param} no es válido, los parámetros válidos son: offset_E, offset_altura, offset_yaw")
        
        
        else:
            for image_path in tqdm(list_images, desc="Adjusting Metadata:"):
                try:
                    # Carga la metadata de la imagen
                    metadata_file = f'{metadata_path}/{image_path[:-4]}.txt'
                    
                    if not os.path.exists(metadata_file):
                        print(f"Advertencia: Archivo de metadatos no encontrado: {metadata_file}")
                        continue
                        
                    with open(metadata_file, 'r') as archivo:
                        data = json.load(archivo)
                        
                    data[param] = value
                    if param == 'offset_E': 
                        data['offset_E_tot'] = value  
                        
                    with open(metadata_file, 'w') as f:
                        json.dump(data, f, indent=4)
                        
                except FileNotFoundError:
                    print(f"Error: No se pudo encontrar el archivo de metadatos: {metadata_file}")
                except json.JSONDecodeError as e:
                    print(f"Error: JSON inválido en el archivo {metadata_file}: {e}")
                except PermissionError:
                    print(f"Error: No hay permisos para acceder al archivo: {metadata_file}")
                except KeyError as e:
                    print(f"Error: Clave no encontrada en el archivo {metadata_file}: {e}")
                except Exception as e:
                    print(f"Error inesperado al procesar {image_path}: {e}")
        
    def get_metadata(self, image_path):
        # Carga la metadata de la imagen
        try:
            metadata_file = f'{image_path[:-4]}.txt'
            
            if not os.path.exists(metadata_file):
                raise FileNotFoundError(f"Archivo de metadatos no encontrado: {metadata_file}")
                
            with open(metadata_file, 'r') as archivo:
                data = json.load(archivo)
            return data
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: JSON inválido en el archivo {metadata_file}: {e}")
            return None
        except PermissionError:
            print(f"Error: No hay permisos para acceder al archivo: {metadata_file}")
            return None
        except Exception as e:
            print(f"Error inesperado al cargar metadatos de {image_path}: {e}")
            return None
