import numpy as np
from core.geo_processor import GeoProcessor
import cv2
from core.image_processor import ImageHandler

class PolygonProcessor:
    
    def rotate_point(self, pt, centro, ang):
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]])
        return R @ (pt - centro) + centro
        
    def aling_line(self, P1, P2, Q1, Q2):
        

        # === 2. Calcula los vectores y ángulos de ambas líneas ===

        vec_img1 = P2 - P1
        angle_img1 = np.arctan2(vec_img1[1], vec_img1[0])

        vec_img2 = Q2 - Q1
        angle_img2 = np.arctan2(vec_img2[1], vec_img2[0])
        delta_yaw = angle_img1 - angle_img2

        offset = P2 - Q1
        desp_este = offset[0]
        desp_norte = offset[1]
        desp_yaw = np.degrees(delta_yaw)
        
        return desp_este, desp_norte, desp_yaw

    def align_yaw(self, P1, P2, Q1, Q2):
        # Paso 1: Alinear yaw
        vec_img1 = P2 - P1
        vec_img2 = Q2 - Q1
        
        angle_img1 = np.arctan2(vec_img1[1], vec_img1[0])
        angle_img2 = np.arctan2(vec_img2[1], vec_img2[0])
        delta_yaw = angle_img1 - angle_img2
        desp_yaw = np.degrees(delta_yaw)
        return desp_yaw
    
    def align_east(self, P1, P2, Q1, Q2):
   

        u = (P2 - P1) / np.linalg.norm(P2 - P1)  # unitario sobre la línea
        n = np.array([-u[1], u[0]])              # perpendicular (Este/Oeste relativo a la línea)
        d = Q1 - P1
        perp_offset = np.dot(d, n)
        offset = -perp_offset * n

        desp_este = offset[0]
   

        return desp_este
    
    def get_main_direction(self, polygons, W, H):
        angles = []
        for poly in polygons:
            p1 = np.array(poly[0]); p2 = np.array(poly[1])
            p3 = np.array(poly[2]); p4 = np.array(poly[3])
            # Línea 1
            vec1 = p2 - p1
            angle1 = np.arctan2(vec1[1], vec1[0])
            angles.append(angle1)
            # Línea 2
            vec2 = p4 - p3
            angle2 = np.arctan2(vec2[1], vec2[0])
            angles.append(angle2)
        # Moda por histograma (en grados 0-180)
        angles_deg = np.degrees(angles)
        angles_deg = np.mod(angles_deg, 180)
        hist, bin_edges = np.histogram(angles_deg, bins=36, range=(0, 180))
        max_bin = np.argmax(hist)
        in_mode = (angles_deg >= bin_edges[max_bin]) & (angles_deg < bin_edges[max_bin+1])
        mode_angles = np.array(angles_deg)[in_mode]
        mean_mode_angle = mode_angles.mean()
        mean_mode_angle_rad = np.radians(mean_mode_angle)
        # Línea principal que cruza toda la imagen horizontalmente
        cx, cy = W // 2, H // 2
        # Usar el ancho completo de la imagen para que cruce horizontalmente
        L = W
        dx = np.cos(mean_mode_angle_rad) * L / 2
        dy = np.sin(mean_mode_angle_rad) * L / 2
        start_point = (int(cx - dx), int(cy - dy))
        end_point = (int(cx + dx), int(cy + dy))
        return mean_mode_angle_rad, start_point, end_point

    def get_main_direction_horizontal(self, polygons, W, H):
        angles = []
        for poly in polygons:
            p1 = np.array(poly[0]); p2 = np.array(poly[1])
            p3 = np.array(poly[2]); p4 = np.array(poly[3])
            # Línea horizontal superior (p2 a p3)
            vec1 = p3 - p2
            angle1 = np.arctan2(vec1[1], vec1[0])
            angles.append(angle1)
            # Línea horizontal inferior (p4 a p1)
            vec2 = p1 - p4
            angle2 = np.arctan2(vec2[1], vec2[0])
            angles.append(angle2)
        # Moda por histograma (en grados 0-180)
        angles_deg = np.degrees(angles)
        angles_deg = np.mod(angles_deg, 180)
        hist, bin_edges = np.histogram(angles_deg, bins=36, range=(0, 180))
        max_bin = np.argmax(hist)
        in_mode = (angles_deg >= bin_edges[max_bin]) & (angles_deg < bin_edges[max_bin+1])
        mode_angles = np.array(angles_deg)[in_mode]
        mean_mode_angle = mode_angles.mean()
        mean_mode_angle_rad = np.radians(mean_mode_angle)
        # Línea principal que cruza toda la imagen horizontalmente
        cx, cy = W // 2, H // 2
        # Usar el ancho completo de la imagen para que cruce horizontalmente
        L = W
        dx = np.cos(mean_mode_angle_rad) * L / 2
        dy = np.sin(mean_mode_angle_rad) * L / 2
        start_point = (int(cx - dx), int(cy - dy))
        end_point = (int(cx + dx) - 1, int(cy + dy))
        return start_point, end_point
    
    def get_desp_line_yaw(self, points, metadatas):
        
        x1, y1 = points[0]
        x2, y2 = points[1]
        x1_next, y1_next = points[2]
        x2_next, y2_next = points[3]
        
        
        metadata = metadatas[0]
        metadata_next = metadatas[1]
        
        geo_data = GeoProcessor().get_georef_matriz(metadata, metadata['offset_E_tot'], metadata['offset_N_tot'], metadata['offset_yaw'], metadata['offset_altura'])
        geo_data_next = GeoProcessor().get_georef_matriz(metadata_next, metadata_next['offset_E_tot'], metadata_next['offset_N_tot'], metadata_next['offset_yaw'], metadata_next['offset_altura'])
        
                    
        x1_utm, y1_utm = geo_data[y1][x1][0], geo_data[y1][x1][1]
        x2_utm, y2_utm = geo_data[y2][x2][0], geo_data[y2][x2][1]
        
        x1_utm_next, y1_utm_next = geo_data_next[y1_next][x1_next][0], geo_data_next[y1_next][x1_next][1]
        x2_utm_next, y2_utm_next = geo_data_next[y2_next][x2_next][0], geo_data_next[y2_next][x2_next][1]


        P1 = np.array([x1_utm, y1_utm])  # inicio de línea 1 en imagen 1
        P2 = np.array([x2_utm, y2_utm])  # fin de línea 1 en imagen 1
        Q1 = np.array([x1_utm_next, y1_utm_next])  # inicio de línea 1 en imagen 2
        Q2 = np.array([x2_utm_next, y2_utm_next])  # fin de línea 1 en imagen 2
                    
        
        
        desp_yaw = self.align_yaw(P1, P2, Q1, Q2)

        
        return desp_yaw
    
    
    def get_middle_line(self, save_path,dir_path, image_path, panels_data, save_images: bool = False):        

        data_image = cv2.imread(f"{dir_path}/{image_path}")
        H, W, _ = data_image.shape
        
   
        
        polygons_image = panels_data[image_path]["polygons"]
   
        start_point, end_point = PolygonProcessor().get_main_direction_horizontal(polygons_image, W, H)
                    
        if save_images:
            data_image_copy = ImageHandler().draw_center_line(save_path, image_path, start_point, end_point)
            cv2.imwrite(f"{save_path}/{image_path}", data_image_copy)

        
        return start_point, end_point
    
    def get_desp_yaw_image(self, transformer, puntos, start_point, end_point, metadata):
        
        angle_yaw = GeoProcessor().get_yaw_angle(puntos)
        
        x1, y1 = start_point
        x2, y2 = end_point

        geo_data = GeoProcessor().get_georef_matriz(metadata, metadata['offset_E_tot'], metadata['offset_N_tot'], metadata['offset_yaw'], metadata['offset_altura'])
                    
        x1_utm, y1_utm = geo_data[y1][x1][0], geo_data[y1][x1][1]
        x2_utm, y2_utm = geo_data[y2][x2][0], geo_data[y2][x2][1]
        
        lon1, lat1 = transformer.transform(x1_utm, y1_utm)
        lon2, lat2 = transformer.transform(x2_utm, y2_utm)
  
        angle_img = GeoProcessor().get_angle_line([(lon1, lat1), (lon2, lat2)])


        delta_yaw = angle_yaw - angle_img

        return delta_yaw
 
