import os
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
from pyproj import CRS, Transformer
import cv2
import shutil
from core.image_processor import ImageHandler
from core.metadata_manager import MetadataManager
from core.geo_processor import GeoProcessor
from core.polygon_processor import PolygonProcessor
import json
from scipy.optimize import minimize
from utils.config import Config



class SolarCorrector:
    
    def __init__(self, path_root: str, kml_path: str = None, cvat_images: bool = True):
        paths = Config.get_project_paths(path_root)
        
        self.path_PP = paths["PP"]
        self.cvat_images = cvat_images
        self.kml_path = kml_path
        self.original_images_path = paths["original_images"]
        self.cvat_images_path = paths["cvat_images"]
        self.lines_images_path = paths["lines_images"]
        self.metadata_lines_path = paths["metadata_lines"]
        self.geonp_path = paths["geonp"]
        self.metadata_path = paths["metadata"]
        self.masks_path = paths["masks"]
        self.segmented_images_path = paths["segmented_images"]
        self.json_path = paths["json_path"]
        self.flights_json = paths["flights_json"]

        # Modelo YOLO con ruta y parámetros centralizados
        self.model = YOLO(Config.MODEL_PATH)

        self.list_images = os.listdir(self.cvat_images_path)
        self.list_flights = []
        self.zone_number = 19
        self.zone_letter = 'S'

        self.utm_crs = CRS(f"+proj=utm +zone={self.zone_number} +{'+south' if self.zone_letter > 'N' else ''} +ellps=WGS84")
        self.latlon_crs = CRS("EPSG:4326")
        self.transformer = Transformer.from_crs(self.utm_crs, self.latlon_crs, always_xy=True)

        
        self.panels_data = {}
        for image_path in self.list_images:
            self.panels_data[image_path] = {"polygons":[]}
        
        # Crear las carpetas si no existen
        os.makedirs(self.lines_images_path, exist_ok=True)
        os.makedirs(self.metadata_lines_path, exist_ok=True)
        os.makedirs(self.masks_path, exist_ok=True)
        os.makedirs(self.segmented_images_path, exist_ok=True)
        
    def init_from_json(self):
        print(f"Cargando datos desde JSON")
        with open(self.json_path, 'r') as f:
            self.panels_data = json.load(f)
        
        # Cargar la lista de vuelos desde JSON
        with open(f"{self.path_PP}/list_flights.json", 'r') as f:
            self.list_flights = json.load(f)
                 


        print(f"Datos cargados de exitosamente")
        
    def reset_metadata(self, var: str = 'all'):
        MetadataManager().reset_all_metadata(self.list_images, self.metadata_path, var)
       
        MetadataManager().reset_all_metadata(self.list_images, self.metadata_lines_path, var)
        
    def save_geo_matrix(self):
        GeoProcessor().save_georef_matriz(self.list_images, self.metadata_path, self.geonp_path)
        
    def findFlights(self, min_line: int = 4, save_kml: bool = False):
                
        print(f"Buscando vuelos en {self.path_PP}")
        
        # min_line: minimo de imagenes por vuelo, si no hay suficientes, se elimina el vuelo y las imagenes, si es 0 no se elimina nada.
    
        listCords = []
        for image_path in tqdm(self.list_images, desc="Calculando lineas"):
            try:
                # Cargar imagen con control de errores
                try:
                    img = cv2.imread(self.cvat_images_path + "/" + image_path)
                    if img is None:
                        print(f"Error: No se pudo cargar la imagen {image_path}")
                        continue
                except Exception as e:
                    print(f"Error al cargar la imagen {image_path}: {e}")
                    continue
                
                H, W, _ = img.shape

                # coordenada centro de la imagen
                xc = W // 2
                yc = H // 2
                
                # Cargar datos geográficos con control de errores
                try:
                    metadata = MetadataManager().get_metadata(f"{self.metadata_path}/{image_path[:-4]}.txt")
                    geo_data = GeoProcessor().get_georef_matriz(metadata, metadata['offset_E_tot'], metadata['offset_N_tot'], metadata['offset_yaw'], metadata['offset_altura'])
                    
                except FileNotFoundError:
                    print(f"Error: No se encontró la matrix de georeferenciada para {image_path}")
                    continue
                except Exception as e:
                    print(f"Error al cargar la matrix de georeferenciada para {image_path}: {e}")
                    continue
                
                try:
                    xc_utm, yc_utm = geo_data[yc][xc][0], geo_data[yc][xc][1]
                    lon_c, lat_c = self.transformer.transform(xc_utm, yc_utm)
                    listCords.append((lat_c, lon_c))
                except IndexError:
                    print(f"Error: Índices fuera de rango para {image_path}")
                    continue
                except Exception as e:
                    print(f"Error en transformación de coordenadas para {image_path}: {e}")
                    continue
                    
            except Exception as e:
                print(f"Error general procesando {image_path}: {e}")
                continue
        
        if not listCords:
            print("Error: No se pudieron procesar coordenadas de ninguna imagen")
            return
            
        latitudes, longitudes = zip(*listCords)

        flights_list = []
        idx_flight = 0
        # Calcular los cambios de dirección
        flights_list.append([self.list_images[0]])
        
        for i in range(1, len(self.list_images)):
            try:
                dy = latitudes[i] - latitudes[i-1]
                dx = longitudes[i] - longitudes[i-1]
                angle = abs(np.arctan2(dy, dx) * (180 / np.pi))  # Convertir a grados
                                    
                if  100 > angle >= 80:
                    flights_list[idx_flight].append(self.list_images[i])
                else:
                    idx_flight += 1
                    flights_list.append([self.list_images[i]])
            except Exception as e:
                print(f"Error calculando ángulo para imagen {i}: {e}")
                continue
                        

        new_flights = flights_list.copy()
        for line in new_flights:
            if len(line) < min_line:
                new_flights.remove(line)
        
        print("Numero de vuelos: ", len(new_flights))

        print(f"Copiando imagenes a la carpeta {self.lines_images_path}")
        for flight in new_flights:
            for image_path in flight:
                try:
                    shutil.copy(self.cvat_images_path + "/" + image_path, self.lines_images_path + "/" + image_path)
                    shutil.copy(self.metadata_path + "/" + image_path[:-4] + '.txt', self.metadata_lines_path + "/" + image_path[:-4] + '.txt')
  
                except FileNotFoundError as e:
                    print(f"Error: No se encontró el archivo {image_path} o su metadata")
                    
                except Exception as e:
                    print(f"Error copiando archivos para {image_path}: {e}")

        self.list_flights = new_flights
        
        if save_kml:
            GeoProcessor().save_kml_vuelos(self.path_PP, self.lines_images_path, self.metadata_lines_path, self.list_flights, name="Flights")
        
        with open(self.json_path, 'w') as f:
            json.dump(self.panels_data, f)
            
        # guardar como lista usando JSON (mejor para listas de listas con strings)
        with open(f"{self.path_PP}/list_flights.json", 'w') as f:
            json.dump(self.list_flights, f)
        
        
    def get_seg_paneles(self, save_masks: bool = False, epsilon_factor: float = 0.015, area_min: float = 4500):
        print(f"Detectando paneles en {self.path_PP}")
        
        if self.list_flights == []:
            print("No hay vuelos para procesar")
            return
        for flight in tqdm(self.list_flights, desc="Detectando paneles"):
            for image_path in flight:
                try:
                    # Cargar datos de la imagen con control de errores
                    try:
                        if self.cvat_images:
                            data_image = ImageHandler().get_image_data(self.cvat_images_path + "/" + image_path)
                        else:
                            data_image = ImageHandler().get_image_data(self.original_images_path + "/" + image_path)
                    except FileNotFoundError:
                        print(f"Error: No se encontró la imagen {image_path}")
                        continue
                    except Exception as e:
                        print(f"Error al cargar la imagen {image_path}: {e}")
                        continue
                    
                    try:
                        if self.cvat_images:
                            data_processed_image, H, W = ImageHandler().process_image(self.cvat_images_path + "/" + image_path, self.cvat_images)
                        else:
                            data_processed_image, H, W = ImageHandler().process_image(self.original_images_path + "/" + image_path, self.cvat_images)
                    except Exception as e:
                        print(f"Error al procesar la imagen con filtros de color {image_path}: {e}")
                        continue
                    
                    try:
                        results = self.model(source=data_processed_image, verbose=False)
                    except Exception as e:
                        print(f"Error en la detección YOLO para {image_path}: {e}")
                        continue

                    polygons_list = []
      
                    for result in results:
                        if result.masks is not None:
                            for j, mask in enumerate(result.masks.data):
                                try:
                                    mask = mask.cpu().numpy() * 255
                                    mask = cv2.resize(mask, (W, H))
                                    data_processed_image = cv2.resize(data_processed_image, (W, H))
                                    
                                    # Convertir la máscara a una imagen binaria
                                    _, thresholded = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)

                                    # Encontrar contornos
                                    contours, _ = cv2.findContours(thresholded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    
                                    if save_masks:
                                        try:
                                            if not os.path.exists(f"{self.masks_path}/{image_path}"):
                                                cv2.imwrite(f'{self.masks_path}/{image_path}', mask)
                                            else:
                                                mask_saved = cv2.imread(f'{self.masks_path}/{image_path}')
                                                # pasar de (W,H,3) a (W,H)
                                                mask_saved = cv2.cvtColor(mask_saved, cv2.COLOR_BGR2GRAY)
                                                mask_total  = mask_saved + mask
                              
                                                cv2.imwrite(f'{self.masks_path}/{image_path}', mask_total)
                                        except Exception as e:
                                            print(f"Error al guardar máscara para {image_path}: {e}")
                                            
                                    if contours:
                                        # Encuentra el contorno más grande
                                        largest_contour = max(contours, key=cv2.contourArea)

                                        # Aproximación del polígono
                                        epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
                                        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
                                        approx_polygon = sorted(approx_polygon, key=lambda x: x[0][0])
                                        approx_polygon = np.array(approx_polygon, dtype=int)

                                    
                                        if len(approx_polygon) > 3:
                                            
                                      
                                            x1 = approx_polygon[0][0][0]
                                            y1 = approx_polygon[0][0][1]
                                            x2 = approx_polygon[1][0][0]
                                            y2 = approx_polygon[1][0][1]
                                            x3 = approx_polygon[2][0][0]
                                            y3 = approx_polygon[2][0][1]
                                            x4 = approx_polygon[3][0][0]
                                            y4 = approx_polygon[3][0][1]


                                            points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                                            points_ordered = ImageHandler().order_points(points)
                                            
                                            x1, y1 = points_ordered[0]
                                            x2, y2 = points_ordered[1]
                                            x3, y3 = points_ordered[2]
                                            x4, y4 = points_ordered[3]
                                                                                                                     
                                            polygons_list.append([(int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3)), (int(x4), int(y4))])
                                            
                                            if save_masks:
                                                try:
                                                    if not os.path.exists(f"{self.segmented_images_path}/{image_path}"):
                                                        draw_image = ImageHandler().draw_segmented_image(self.cvat_images_path, image_path, points_ordered)
                                                    else:
                                                        draw_image = ImageHandler().draw_segmented_image(self.segmented_images_path, image_path, points_ordered)
                                                        
                                                    cv2.imwrite(f"{self.segmented_images_path}/{image_path}", draw_image)
                                                except Exception as e:
                                                    print(f"Error al guardar la imagen segmentada para {image_path}: {e}")
                                            
                                    
                                except Exception as e:
                                    print(f"Error procesando la máscara {j} de {image_path}: {e}")
                                    continue
                    self.panels_data[image_path]["polygons"] = polygons_list
                
                    
                except Exception as e:
                    print(f"Error general procesando la imagen {image_path}: {e}")
                    continue
        
        with open(self.json_path, 'w') as f:
            json.dump(self.panels_data, f)

        print(f"Paneles detectados: {len(self.panels_data)}")
        
        
    def correct_yaw_img(self, save_images: bool = False, puntos: list = None):
        
        if self.list_flights == []:
            self.init_from_json()         
        for flight in tqdm(self.list_flights, desc="Calculando desplazamientos de las lineas"):
            for e, image_path in enumerate(flight):
                if e+1 < len(flight):
                    
                    next_image_path = flight[e+1]
                    
                    start_point, end_point = PolygonProcessor().get_middle_line(self.segmented_images_path, 
                                                                                self.cvat_images_path, image_path, 
                                                                                self.panels_data, save_images)
                    
                    start_point_next, end_point_next = PolygonProcessor().get_middle_line(self.segmented_images_path,
                                                                                          self.cvat_images_path, next_image_path, 
                                                                                          self.panels_data, save_images)
                        
                    desp_yaw = PolygonProcessor().get_desp_line_yaw([start_point, end_point, start_point_next, end_point_next], 
                                                                       [MetadataManager().get_metadata(f"{self.metadata_lines_path}/{image_path[:-4]}.txt"),
                                                                        MetadataManager().get_metadata(f"{self.metadata_lines_path}/{next_image_path[:-4]}.txt")])
                    
                    MetadataManager().adjust_metadata(f"{self.metadata_lines_path}/{next_image_path[:-4]}.txt", 'offset_yaw', desp_yaw)
                    
                    MetadataManager().adjust_metadata(f"{self.metadata_path}/{image_path[:-4]}.txt", 'offset_yaw', desp_yaw)
           
        GeoProcessor().save_kml_vuelos(self.path_PP, self.segmented_images_path, self.metadata_lines_path, self.list_flights, name="Y_line")
        GeoProcessor().save_kml_vuelos(self.path_PP, self.lines_images_path, self.metadata_lines_path, self.list_flights, name="Y")
                
    def correct_yaw(self, save_images: bool = False, puntos: list = None):
        if self.list_flights == []:
            self.init_from_json()         
            
        for flight in tqdm(self.list_flights, desc="Calculando desplazamientos de las lineas"):
            for e, image_path in enumerate(flight):
        
                    start_point, end_point = PolygonProcessor().get_middle_line(self.segmented_images_path, 
                                                                                self.cvat_images_path, image_path, 
                                                                                self.panels_data, save_images)
            
                    desp_yaw = PolygonProcessor().get_desp_yaw_image(self.transformer, puntos, start_point, end_point, 
                                                                     MetadataManager().get_metadata(f"{self.metadata_path}/{image_path[:-4]}.txt"))
                    MetadataManager().adjust_metadata(f"{self.metadata_path}/{image_path[:-4]}.txt", 'offset_yaw', desp_yaw)
                    MetadataManager().adjust_metadata(f"{self.metadata_lines_path}/{image_path[:-4]}.txt", 'offset_yaw', desp_yaw)
           
        GeoProcessor().save_kml_vuelos(self.path_PP, self.segmented_images_path, self.metadata_lines_path, self.list_flights, name="Y_line")
        GeoProcessor().save_kml_vuelos(self.path_PP, self.lines_images_path, self.metadata_lines_path, self.list_flights, name="Y")
        
        
        
        
        
        
        
    def correct_H(self):
        pass

 


            