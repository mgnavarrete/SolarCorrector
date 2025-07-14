import numpy as np
import utm

class CameraProcessor:
    
    def dms2dd(self, data):
        try:
            if len(data) < 4:
                raise ValueError("Los datos DMS deben tener al menos 4 elementos")
            dd = float(data[0]) + float(data[1]) / 60 + float(data[2]) / (60 * 60)
            if data[3] == 'W' or data[3] == 'S':
                dd *= -1
            return dd
        except (ValueError, IndexError) as e:
            print(f"Error convirtiendo DMS a DD: {e}")
            return None
    
    def get_image_pos_utm(self, data):
        try:
            # Obtiene las posiciones en el formato que sale con exiftools
            if 'GPSLatitude' not in data or 'GPSLongitude' not in data:
                raise ValueError("Datos GPS faltantes en metadata")
                
            lat = data['GPSLatitude'].replace('\'', '').replace('"', '').split(' ')
            lng = data['GPSLongitude'].replace('\'', '').replace('"', '').split(' ')
            
            # Elimina la palabra 'deg' de los datos
            for v in lat:
                if v == 'deg':
                    lat.pop(lat.index(v))
            for v in lng:
                if v == 'deg':
                    lng.pop(lng.index(v))
                    
            # Calcula la posición en coordenadas UTM
            lat_dd = self.dms2dd(lat)
            lng_dd = self.dms2dd(lng)
            
            if lat_dd is None or lng_dd is None:
                raise ValueError("Error en conversión de coordenadas")
                
            pos = utm.from_latlon(lat_dd, lng_dd)
            return pos
        except Exception as e:
            print(f"Error obteniendo posición UTM: {e}")
            return None
        
    def mavic2_processor(self, data,  desp_este=0, desp_norte=0, desp_yaw=0, offset_altura=0, modo_altura="relativo", dist=None, ans=None, sig=None):
        img_height = int(data['ImageHeight'])
        img_width = int(data['ImageWidth'])
        tamano_pix = 0.000012
        dis_focal = 9 / 1000  # mavic 2 enterprice
        if data["GimbalYawDegree"] is not None:
            yaw = np.pi * (float(data["GimbalYawDegree"]) + float(desp_yaw)) / 180
        else:
            yaw = 0
        center = self.get_image_pos_utm(data)
        if center is None:
            raise ValueError("No se pudo obtener la posición UTM")
        if modo_altura == "relativo":
            #altura = float(data['RelativeAltitude']) - float(offset_altura)
            if float(data['RelativeAltitude']) < 3:
                relAltitude = 3
            else:
                relAltitude = float(data['RelativeAltitude'])
            altura = relAltitude - float(offset_altura)
        else:
            altura = offset_altura
        GSD = tamano_pix * (altura) / dis_focal
        # Cálculo del desplazamiento debido al pitch de la cámara
        pitch = np.pi * (float(data["GimbalPitchDegree"])) / 180.0
        desp_pitch = altura * np.tan(-np.pi / 2 + pitch)
        
        return img_height, img_width, yaw, center, desp_pitch, GSD
        
        
    def M3T_processor(self, data, desp_este=0, desp_norte=0, desp_yaw=0, offset_altura=0, modo_altura="relativo", dist=None, ans=None, sig=None):
        
        img_height = int(data['ImageHeight'])
        img_width = int(data['ImageWidth'])
        tamano_pix = 0.000012
        dis_focal = 9 / 1000  # mavic 2 enterprice
        if data["GimbalYawDegree"] is not None:
            yaw = np.pi * (float(data["GimbalYawDegree"]) + float(desp_yaw)) / 180
        else:
            yaw = 0
        center = self.get_image_pos_utm(data)
        if center is None:
            raise ValueError("No se pudo obtener la posición UTM")
        if modo_altura == "relativo":
            if float(data['RelativeAltitude']) < 3:
                relAltitude = 3
            else:
                relAltitude = float(data['RelativeAltitude'])
            altura = relAltitude - float(offset_altura)
        else:
            altura = offset_altura
        GSD = tamano_pix * (altura) / dis_focal
        # Cálculo del desplazamiento debido al pitch de la cámara
        pitch = np.pi * (float(data["GimbalPitchDegree"])) / 180.0
        desp_pitch = altura * np.tan(-np.pi / 2 + pitch)
        
        return img_height, img_width, yaw, center, desp_pitch, GSD
        
        
    def xt2_processor(self, data,  desp_este=0, desp_norte=0, desp_yaw=0, offset_altura=0, modo_altura="relativo", dist=None, ans=None, sig=None):
        img_height = int(data['ImageHeight'])
        img_width = int(data['ImageWidth'])
        tamano_pix = 0.000012
        dis_focal = 9 / 1000  # mavic 2 enterprice
        if data["GimbalYawDegree"] is not None:
            yaw = np.pi * (float(data["GimbalYawDegree"]) + float(desp_yaw)) / 180
        else:
            yaw = 0
        center = self.get_image_pos_utm(data)
        if center is None:
            raise ValueError("No se pudo obtener la posición UTM")
        if modo_altura == "relativo":
            altura = float(data['RelativeAltitude']) - float(offset_altura)
        else:
            altura = float(offset_altura)
        GSD = tamano_pix * (altura) / dis_focal
        # Cálculo del desplazamiento debido al pitch de la cámara
        pitch = np.pi * (float(data["GimbalPitchDegree"])) / 180.0
        desp_pitch = altura * np.tan(-np.pi / 2 + pitch) 
        
        return img_height, img_width, yaw, center, desp_pitch, GSD  
        
    def zh20t_processor(self, data,  desp_este=0, desp_norte=0, desp_yaw=0, offset_altura=0, modo_altura="relativo", dist=None, ans=None, sig=None):
        
        img_height = int(data['ImageHeight'])
        img_width = int(data['ImageWidth'])
        tamano_pix = 0.000012
        dis_focal = float(data['FocalLength'][:-2]) / 1000
        # yaw = np.pi * (float(data["FlightYawDegree"]) + desp_yaw) / 180
        if data["GimbalYawDegree"] is not None:
            yaw = np.pi * (float(data["GimbalYawDegree"]) + float(desp_yaw)) / 180
        else:
            yaw = 0
        pitch = np.pi * (float(data["GimbalPitchDegree"])) / 180.0

        try:
            distancia_laser = float(data["LRFTargetDistance"]) #if dist is not None else dist
            lat_laser = float(data["LRFTargetLat"])
            lon_laser = float(data["LRFTargetLon"])
            altura = distancia_laser * abs(np.sin(pitch))
            GSD = tamano_pix * altura / dis_focal
            if ans is not None and sig is not None:
                if float(sig["LRFTargetLat"]) < lat_laser < float(ans["LRFTargetLat"]):
                    lon_laser += float(sig["LRFTargetLon"]) + float(ans["LRFTargetLon"])
                    lon_laser /= 3
            usar_posicion_laser = False
            if usar_posicion_laser:
                center = utm.from_latlon(lat_laser, lon_laser)
                desp_pitch = 0
            else:
                center = self.get_image_pos_utm(data)
                if center is None:
                    raise ValueError("No se pudo obtener la posición UTM")
                desp_pitch = altura * np.tan(-np.pi / 2 + pitch)
                
            return img_height, img_width, yaw, center, desp_pitch, GSD

        except Exception as e:
            print(f"Error con datos láser, usando GPS: {e}")
            center = self.get_image_pos_utm(data)
            if center is None:
                raise ValueError("No se pudo obtener la posición UTM")
            if modo_altura == "relativo":
                altura = float(data['RelativeAltitude']) - float(offset_altura)
            else:
                altura = float(offset_altura)
            GSD = tamano_pix * (altura) / dis_focal
            # Cálculo del desplazamiento debido al pitch de la cámara
            pitch = np.pi * (float(data["GimbalPitchDegree"])) / 180.0
            desp_pitch = altura * np.tan(-np.pi / 2 + pitch)
            
            return img_height, img_width, yaw, center, desp_pitch, GSD
        
        
        
    def get_camera_processor(self, metadata,  desp_este=0, desp_norte=0, desp_yaw=0, offset_altura=0, modo_altura="relativo", dist=None, ans=None, sig=None):
        if metadata['Model'] == "MAVIC2-ENTERPRISE-ADVANCED":
            return self.mavic2_processor(metadata, desp_este, desp_norte, desp_yaw, offset_altura, modo_altura, dist, ans, sig)
                
        elif metadata['Model'] == "M3T":
            return self.M3T_processor(metadata, desp_este, desp_norte, desp_yaw, offset_altura, modo_altura, dist, ans, sig)
        
        elif metadata['Model'] == "XT2":
            return self.xt2_processor(metadata, desp_este, desp_norte, desp_yaw, offset_altura, modo_altura, dist, ans, sig)
        
        elif metadata['Model'] == "ZH20T":
            return self.zh20t_processor(metadata, desp_este, desp_norte, desp_yaw, offset_altura, modo_altura, dist, ans, sig)
        
        else:
            print("===================================================")
            print("CÁMARA NO DEFINIDA")
            return None, None, None, None, None, None