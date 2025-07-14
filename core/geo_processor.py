import os
from tqdm import tqdm
import json
import numpy as np
import string
import utm
from core.metadata_manager import MetadataManager
import math
from core.camera_processor import CameraProcessor

class GeoProcessor:

    
    def save_georef_matriz(self, list_images, metadata_path, geonp_path):
        # Genera las matrices georeferenciadas de las imágenes
        try:
            for image_path in tqdm(list_images, desc="Generando Matrices Georeferenciadas de las imágenes"):
                try:
                    # Carga la metadata de la imagen
                    metadata_file = f'{metadata_path}/{image_path[:-4]}.txt'
                    if not os.path.exists(metadata_file):
                        print(f"Advertencia: No se encontró el archivo de metadata {metadata_file}")
                        continue
                        
                    with open(metadata_file, 'r') as archivo:
                        data = json.load(archivo)
                        
                    m = self.get_georef_matriz(data, data['offset_E_tot'], data['offset_N_tot'], data['offset_yaw'], data['offset_altura'])
                    if m is not None:
                        geo_name = f'{geonp_path}/{image_path[:-4]}.npy'
                        np.save(geo_name, m)
                    else:
                        print(f"Error: No se pudo generar la matriz para {image_path}")
                        
                except FileNotFoundError as e:
                    print(f"Error: No se encontró el archivo de metadata para {image_path}: {e}")
                except json.JSONDecodeError as e:
                    print(f"Error: JSON inválido en metadata de {image_path}: {e}")
                except KeyError as e:
                    print(f"Error: Campo faltante en metadata de {image_path}: {e}")
                except Exception as e:
                    print(f"Error inesperado procesando {image_path}: {e}")
                    
            print(f"Matrices Georeferenciadas generadas para todas las imágenes de la carpeta {geonp_path}")
        except Exception as e:
            print(f"Error general en get_geoMatrix: {e}")
    
    def delete_geoMatrix(self, geonp_path):
        # Elimina las matrices georeferenciadas de las imágenes
        try:
            if not os.path.exists(geonp_path):
                print(f"Advertencia: La carpeta {geonp_path} no existe")
                return
                
            for filename in tqdm(os.listdir(geonp_path), desc="Deleting GeoNumpy:"):
                try:
                    file_path = os.path.join(geonp_path, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except PermissionError as e:
                    print(f'Error de permisos al eliminar %s. Razón: %s' % (file_path, e))
                except Exception as e:
                    print(f'Error al eliminar %s. Razón: %s' % (file_path, e))
        except Exception as e:
            print(f"Error general en delete_geoMatrix: {e}")
    
    def get_georef_matriz(self, data, desp_este=0, desp_norte=0, desp_yaw=0, offset_altura=0, modo_altura="relativo", dist=None, ans=None, sig=None):
        try:
            metadata = data
            
            img_height, img_width, yaw, center, desp_pitch, GSD = CameraProcessor().get_camera_processor(metadata, desp_este, desp_norte, desp_yaw, offset_altura, modo_altura, dist, ans, sig)
            
            mid_width = img_width / 2

            Matriz_y = np.zeros((img_height, img_width))
            Matriz_x = np.zeros((img_height, img_width))

            for pixel_y in range(img_height):
                distancia_y = (pixel_y - img_height / 2 + 0.5) * GSD
                Matriz_y[pixel_y, :] = np.ones(img_width) * -1 * distancia_y

            matriz_gsd_y = (np.append(Matriz_y[:, 0], Matriz_y[-1, 0]) - np.append(Matriz_y[0, 0], Matriz_y[:, 0]))
            matriz_gsd_x = matriz_gsd_y[1:-1]  # asumimos pixeles cuadrados
            matriz_gsd_x = np.append(matriz_gsd_x[0], matriz_gsd_x[:])

            for pixel_y in range(img_height):
                gsd_x = matriz_gsd_x[pixel_y]
                distancia_x = -gsd_x * (mid_width - 0.5)
                for pixel_x in range(img_width):
                    Matriz_x[pixel_y, pixel_x] = distancia_x
                    distancia_x = distancia_x + gsd_x

            # AJUSTAR OFFSET DEL GPS, VALORES REFERENCIALES
            Matriz_Este = Matriz_y * np.sin(yaw) - Matriz_x * np.cos(yaw) + center[0] + float(desp_este) + float(desp_pitch) * np.sin(yaw)
            Matriz_Norte = Matriz_y * np.cos(yaw) + Matriz_x * np.sin(yaw) + center[1] + float(desp_norte) + float(desp_pitch) * np.cos(yaw)

            #print(center[0], center[1])

            Matriz_zonas_1 = np.ones((img_height, img_width)) * center[2]
            Matriz_zonas_2 = np.ones((img_height, img_width)) * string.ascii_uppercase.find(center[3])

            matriz_puntos_utm = np.concatenate(
                [Matriz_Este[..., np.newaxis], Matriz_Norte[..., np.newaxis], Matriz_zonas_1[..., np.newaxis],
                Matriz_zonas_2[..., np.newaxis]], axis=-1)
            return matriz_puntos_utm
        except Exception as e:
            print(f"Error en save_georef_matriz: {e}")
            return None
    
    def save_kml_vuelos(self, path_PP, dir_images_path, metadata_path, list_vuelos, name=""):
        try:
            # Verifica que existan las carpetas necesarias
            if not os.path.exists(path_PP):
                os.makedirs(path_PP)
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"La carpeta de metadata {metadata_path} no existe")
                
            # Abre un solo archivo KML para todos los vuelos
            kml_filename = f"{path_PP}/{path_PP.split('/')[-1]}_{name}.kml"
            with open(kml_filename, 'w') as file:
                # Escribe el encabezado del archivo KML
                a = f'''<?xml version="1.0" encoding="UTF-8"?>
        <kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">
        <Document>
            <name>{path_PP.split('/')[-1] + '_' + name}</name>
            '''
                file.write(a)

                for idx, vuelo in tqdm(enumerate(list_vuelos), desc=f"Generando KML para vuelo", total=len(list_vuelos)):
                    try:
                        # Inicia un nuevo folder para cada vuelo
                        a = f'''<Folder>
                    <name>Line_{idx}</name>
                    '''
                        file.write(a)

                        vuelo_ant = ''
                        for f_name in vuelo:
                            try:
                                nombre = f_name[:-4]
                                vuelo = 'cvat'

                                # Carga la metadata de la imagen
                                str_metada_file = f"{metadata_path}/{nombre}.txt"
                                if not os.path.exists(str_metada_file):
                                    print(f"Advertencia: No se encontró metadata para {nombre}")
                                    continue
                                    
                                with open(str_metada_file) as metadata_file:
                                    data2 = json.load(metadata_file)

                                modo_altura = data2['modo_altura']
                                m = self.get_georef_matriz(data2, data2['offset_E_tot'], data2['offset_N_tot'], data2['offset_yaw'], data2['offset_altura'], modo_altura)
                                
                                if m is None:
                                    print(f"Error: No se pudo generar matriz para {nombre}")
                                    continue
                                    
                                p1_ll = utm.to_latlon(m[0][0][0], m[0][0][1], int(m[0][0][2]), string.ascii_uppercase[int(m[0][0][3])])
                                p2_ll = utm.to_latlon(m[0][-1][0], m[0][-1][1], int(m[0][-1][2]), string.ascii_uppercase[int(m[0][-1][3])])
                                p3_ll = utm.to_latlon(m[-1][-1][0], m[-1][-1][1], int(m[-1][-1][2]), string.ascii_uppercase[int(m[-1][-1][3])])
                                p4_ll = utm.to_latlon(m[-1][0][0], m[-1][0][1], int(m[-1][0][2]), string.ascii_uppercase[int(m[-1][0][3])])

                                # Coordenadas para el kml
                                cordinates = f"{str(p4_ll[1])},{str(p4_ll[0])},0 {str(p3_ll[1])},{str(p3_ll[0])},0 {str(p2_ll[1])},{str(p2_ll[0])},0 {str(p1_ll[1])},{str(p1_ll[0])},0 "

                                txt_desplazamiento = "_DN" + str(data2['offset_N']) + \
                                                    "_DE" + str(data2['offset_E']) + \
                                                    "_DY" + str(data2['offset_yaw']) + \
                                                    "_DV" + str(data2['desface_gps']) + \
                                                    "_DA" + str(data2['offset_altura']) + \
                                                    "_MA" + str(data2['modo_altura'])

                                txt_href = f'{dir_images_path.split("/")[-1]}/{nombre}.JPG'
                                a = f'''<GroundOverlay>
                    <name>{nombre + txt_desplazamiento}</name>
                    <Icon>
                        <href>{txt_href}</href>
                        <viewBoundScale>0.75</viewBoundScale>
                    </Icon>
                    <gx:LatLonQuad>
                        <coordinates>
                            {cordinates} 
                        </coordinates>
                    </gx:LatLonQuad>
                </GroundOverlay>
                '''
                                file.write(a)
                            except FileNotFoundError as e:
                                print(f"Error: No se encontró archivo para {f_name}: {e}")
                            except json.JSONDecodeError as e:
                                print(f"Error: JSON inválido en metadata de {f_name}: {e}")
                            except KeyError as e:
                                print(f"Error: Campo faltante en metadata de {f_name}: {e}")
                            except Exception as e:
                                print(f"Error procesando {f_name}: {e}")

                        # Cierra el folder del vuelo actual
                        a = '''</Folder>'''
                        file.write(a)
                    except Exception as e:
                        print(f"Error procesando vuelo {idx}: {e}")

                # Cierra el documento KML
                a = '''</Document>
        </kml>'''
                file.write(a)

            print(f"KML generado para todos los vuelos en la carpeta {kml_filename}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except PermissionError as e:
            print(f"Error de permisos al crear KML: {e}")
        except Exception as e:
            print(f"Error general en save_kml_vuelos: {e}")
            
    def get_yaw_angle(self, puntos):
        """
        Recibe una lista de puntos [(lon, lat, z), ...] y retorna el ángulo en grados 
        respecto al norte del lado más largo.
        Sistema de coordenadas: 0°=Norte, 90°=Este, 180°=Sur, 270°=Oeste
        """
        lados = []
        n = len(puntos)
        
        for i in range(n - 1):
            x1, y1, _ = puntos[i]
            x2, y2, _ = puntos[i+1]
            # Distancia euclidiana (aproximación para distancias cortas)
            distancia = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            lados.append((distancia, i, i+1))
        
        # Encuentra el lado más largo
        lado_mas_largo = max(lados, key=lambda l: l[0])
        _, idx1, idx2 = lado_mas_largo

        x1, y1, _ = puntos[idx1]
        x2, y2, _ = puntos[idx2]
        
        # Asegurarse de que el punto 1 sea el mas hacia el norte y el punto 2 el mas hacia el sur
        if y1 > y2:
            x1, y1, _ = puntos[idx2]
            x2, y2, _ = puntos[idx1]
        
        
        pos_utm_start = utm.from_latlon(x1, y1)
        pos_utm_end = utm.from_latlon(x2, y2)

        
        # Calcular el angulo con respecto al norte
        delta_x = pos_utm_end[0] - pos_utm_start[0]
        delta_y = pos_utm_end[1] - pos_utm_start[1]
        angulo_rad = math.atan2(delta_x, delta_y)
        angulo_deg = math.degrees(angulo_rad)
        
        # Convertir al sistema de coordenadas deseado: 0°=Norte, 90°=Este, 180°=Sur, 270°=Oeste
        # Si el ángulo está entre 270° y 360°, convertirlo a valores negativos
        if angulo_deg > 270:
            angulo_deg = angulo_deg - 360
        
        print(f"Angulo de tracker: {angulo_deg}")

        return angulo_deg
        
    def get_angle_line(self, puntos):
        """
        Recibe una lista de puntos [(lon, lat, z), ...] y retorna el ángulo en grados 
        respecto al norte del lado más largo.
        Sistema de coordenadas: 0°=Norte, 90°=Este, 180°=Sur, 270°=Oeste
        """      
        # quiero asegurarme de que siempre el x1 sea el mas hacia el norte y el x2 el mas hacia el sur
      

        pos_start = puntos[0]
        pos_end = puntos[1]
        
        # Asegurarse de que el punto 1 sea el mas hacia el norte y el punto 2 el mas hacia el sur
        if pos_start[1] > pos_end[1]:
            pos_start, pos_end = pos_end, pos_start
        
        
        delta_x = pos_end[0] - pos_start[0]
        delta_y = pos_end[1] - pos_start[1]
        angulo_rad = math.atan2(delta_x, delta_y)
        angulo_deg = math.degrees(angulo_rad)
        
        
        
        # Convertir al sistema de coordenadas deseado: 0°=Norte, 90°=Este, 180°=Sur, 270°=Oeste
        # Si el ángulo está entre 270° y 360°, convertirlo a valores negativos
        if angulo_deg > 270:
            angulo_deg = angulo_deg - 360
            
        print(f"Angulo de la linea: {angulo_deg}")

        return angulo_deg        
    
    