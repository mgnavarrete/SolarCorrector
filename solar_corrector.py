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
import json

class SolarCorrector:
    def __init__(self, path_root: str, kml_path: str = None):
        
        self.path_PP = f"{path_root}PP"                                             # Path de la carpeta PP
        self.kml_path = kml_path                                                    # Path de la tabla kml
        self.original_images_path = os.path.join(self.path_PP, 'original_img')      # Path de las imágenes originales
        self.cvat_images_path = os.path.join(self.path_PP, 'cvat')                  # Path de las imágenes cvat
        self.lines_images_path = os.path.join(self.path_PP, 'lines')                # Path de las imágenes de las lineas
        self.metadata_lines_path = os.path.join(self.path_PP, 'metadata_lines')     # Path de los archivos JSON de metadatos de las lineas
        self.geonp_path = os.path.join(self.path_PP, 'georef_numpy')                # Path de los archivos numpy georeferenciados
        self.metadata_path = os.path.join(self.path_PP, 'metadata')                 # Path de los archivos JSON de metadatos
        self.model = YOLO('utils\PANEL-SEG-v1.pt')                                  # Modelo de YOLO
        self.list_images = os.listdir(self.cvat_images_path)                        # Lista de imágenes cvat
        self.list_flights = []
        self.zone_number = 19
        self.zone_letter = 'S'
        self.utm_crs = CRS(f"+proj=utm +zone={self.zone_number} +{'+south' if self.zone_letter > 'N' else ''} +ellps=WGS84")
        self.latlon_crs = CRS("EPSG:4326")
        self.transformer = Transformer.from_crs(self.utm_crs, self.latlon_crs, always_xy=True)
        self.masks_path = os.path.join(self.path_PP, 'masks')
        self.segmented_images_path = os.path.join(self.path_PP, 'segmented_images')
        self.list_areas = []
        self.json_path = os.path.join(self.path_PP, 'corrector_data.json')
        
        self.panels_data = {}
        for image_path in self.list_images:
            self.panels_data[image_path] = {"polygons":[], "geo_polygons":[], "isFlight":False, "area":0}
        
        with open(self.json_path, 'w') as f:
            json.dump(self.panels_data, f)
        
        if kml_path is not None:
            self.df = pd.read_csv(kml_path)
            for col in ['polyP1', 'polyP2', 'polyP3', 'polyP4']:
                self.df[col] = self.df[col].apply(lambda x: tuple(map(float, x.split(','))))

            self.yaw_mean = self.df['yaw'].mean()                                  # Yaw medio de las lineas
            self.ancho_mean = self.df['ancho'].mean()    
            # Ancho medio de las lineas
            
        os.makedirs(self.lines_images_path, exist_ok=True)
        os.makedirs(self.metadata_lines_path, exist_ok=True)
        os.makedirs(self.masks_path, exist_ok=True)    
        os.makedirs(self.segmented_images_path, exist_ok=True)
        
    def reset_metadata(self, var: str = 'all'):
        MetadataManager().reset_all_metadata(self.list_images, self.metadata_path, var)
        
    def save_geo_matrix(self):
        GeoProcessor().get_geoMatrix(self.list_images, self.metadata_path, self.geonp_path)
        
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
                    geo_data = np.load(f"{self.geonp_path}/{image_path[:-4]}.npy")
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
        # for vuelo in vueloList:
        if len(flights_list) < min_line:
            new_flights.remove(flights_list)
        
        print("Numero de vuelos: ", len(new_flights))

        print(f"Copiando imagenes a la carpeta {self.lines_images_path}")
        for flight in new_flights:
            for image_path in flight:
                try:
                    shutil.copy(self.cvat_images_path + "/" + image_path, self.lines_images_path + "/" + image_path)
                    shutil.copy(self.metadata_path + "/" + image_path[:-4] + '.txt', self.metadata_lines_path + "/" + image_path[:-4] + '.txt')
                    self.panels_data[image_path]["isFlight"] = True
                    
                except FileNotFoundError as e:
                    print(f"Error: No se encontró el archivo {image_path} o su metadata")
                    
                except Exception as e:
                    print(f"Error copiando archivos para {image_path}: {e}")

        self.list_flights = new_flights
        
        if save_kml:
            GeoProcessor().save_kml_vuelos(self.path_PP, self.metadata_lines_path, self.list_flights, name="Flights")
        
        with open(self.json_path, 'w') as f:
            json.dump(self.panels_data, f)
        
        
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
                        data_image = ImageHandler().get_image_data(self.cvat_images_path + "/" + image_path)
                    except FileNotFoundError:
                        print(f"Error: No se encontró la imagen {image_path}")
                        continue
                    except Exception as e:
                        print(f"Error al cargar la imagen {image_path}: {e}")
                        continue
                    
                    try:
                        data_processed_image, H, W = ImageHandler().process_image(self.cvat_images_path + "/" + image_path)
                    except Exception as e:
                        print(f"Error al procesar la imagen con filtros de color {image_path}: {e}")
                        continue
                    
                    try:
                        results = self.model(source=data_processed_image, verbose=False)
                    except Exception as e:
                        print(f"Error en la detección YOLO para {image_path}: {e}")
                        continue

                        
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
                                                mask_saved = cv2.add(mask_saved, mask)
                                                cv2.imwrite(f'{self.masks_path}/{image_path}', mask_saved)
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
                                            
                                            try:
                                                geo_data = np.load(f"{self.geonp_path}/{image_path[:-4]}.npy")
                                            except FileNotFoundError:
                                                print(f"Error: No se encontró la matrix de georeferenciada para {image_path}")
                                                continue
                                            except Exception as e:
                                                print(f"Error al cargar la matrix de georeferenciada para {image_path}: {e}")
                                                continue

                                            x1_utm, y1_utm = geo_data[y1][x1][0], geo_data[y1][x1][1]
                                            x2_utm, y2_utm = geo_data[y2][x2][0], geo_data[y2][x2][1]
                                            x3_utm, y3_utm = geo_data[y3][x3][0], geo_data[y3][x3][1]
                                            x4_utm, y4_utm = geo_data[y4][x4][0], geo_data[y4][x4][1]
                                           
                                       
                                            lon1, lat1 = self.transformer.transform(x1_utm, y1_utm)
                                            lon2, lat2 = self.transformer.transform(x2_utm, y2_utm)
                                            lon3, lat3 = self.transformer.transform(x3_utm, y3_utm)
                                            lon4, lat4 = self.transformer.transform(x4_utm, y4_utm)    
                                          

                                            area = ImageHandler().get_area_polygon(points_ordered)
                                            self.list_areas.append(area)
                                            self.panels_data[image_path]["area"] = area
                                            
                                                                                
                                            self.panels_data[image_path]["polygons"].append(points_ordered)
                                            self.panels_data[image_path]["geo_polygons"].append([(lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4)])

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
                                    
                except Exception as e:
                    print(f"Error general procesando la imagen {image_path}: {e}")
                    continue
        
        with open(self.json_path, 'w') as f:
            json.dump(self.panels_data, f)
        
        print(f"Paneles detectados: {len(self.panels_data)}")
        
    def correct_E(self):
        
        for flight in self.list_flights:
            for e, image_path in enumerate(flight):
                data_image = cv2.imread(self.cvat_images_path + "/" + image_path)
                W, H, _ = data_image.shape
                
                next_image_path = flight[e+1]
                
                polygons_image = self.panels_data[image_path]["polygons"]
                polygons_next_image = self.panels_data[next_image_path]["polygons"]
                
                middle_polygon, e_middle_polygon = ImageHandler().find_middle_polygon(polygons_image, W, H)
                middle_polygon_next, e_middle_polygon_next = ImageHandler().find_middle_polygon(polygons_next_image, W, H)
                
                draw_image = ImageHandler().draw_segmented_image(self.cvat_images_path, image_path, middle_polygon)
                cv2.imwrite(f"{image_path[:-4]}_middle.png", draw_image)
                                
                geo_polygons_image = self.panels_data[image_path]["geo_polygons"]
                geo_polygons_next_image = self.panels_data[next_image_path]["geo_polygons"]
                
                geo_middle_polygon = geo_polygons_image[e_middle_polygon]
                geo_middle_polygon_next = geo_polygons_next_image[e_middle_polygon_next]
                
                
                
                
                
                
                
                
        
        
    
        pass
    
    def correct_H(self):
        pass
    
    def correct_yaw(self):
        pass
    
 


            