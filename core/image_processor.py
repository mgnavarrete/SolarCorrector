import os
import cv2
import numpy as np
import math

class ImageHandler:

    def get_image_data(self, image_path: str):
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"No se encontró la imagen en la ruta: {image_path}")
            
            data_image = cv2.imread(image_path)
            if data_image is None:
                raise ValueError(f"No se pudo cargar la imagen desde: {image_path}")
            
            return data_image
        except FileNotFoundError as e:
            print(f"Error de archivo: {e}")
            return None
        except ValueError as e:
            print(f"Error de valor: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado en get_image_data: {e}")
            return None
    
    def process_image(self, image_path: str):
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"No se encontró la imagen en la ruta: {image_path}")
            
            data_image = cv2.imread(image_path)
            if data_image is None:
                raise ValueError(f"No se pudo cargar la imagen desde: {image_path}")
            
            H, W, _ = data_image.shape
            img_resized = cv2.resize(data_image, (640, 640))

            # pasar a escala de grises
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            # aplicar filtro gaussiano
            img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
            
           

            # aplicar mas brillo
            img_gray = cv2.addWeighted(img_gray, 1.5, img_gray, 0, 0)
            
             # Aplicar un sharpening más intenso
            kernel = np.array([[0, -2, 0], [-2, 9, -2], [0, -2, 0]])
            img_gray = cv2.filter2D(img_gray, -1, kernel)
            

            # pasar a canal BGR pero manteniendo los bordes y el color gris
            img_final = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            
            return img_final, H, W
        except FileNotFoundError as e:
            print(f"Error de archivo: {e}")
            return None, None, None
        except ValueError as e:
            print(f"Error de valor: {e}")
            return None, None, None
        except Exception as e:
            print(f"Error inesperado en process_image: {e}")
            return None, None, None
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float):
        try:
            # Validar que las coordenadas estén en rangos válidos
            if not (-90 <= lat1 <= 90) or not (-90 <= lat2 <= 90):
                raise ValueError("Las latitudes deben estar entre -90 y 90 grados")
            
            if not (-180 <= lon1 <= 180) or not (-180 <= lon2 <= 180):
                raise ValueError("Las longitudes deben estar entre -180 y 180 grados")
            
            R = 6371  # Radio de la Tierra en kilómetros
            dist_lat = np.radians(lat2 - lat1)
            dist_lon = np.radians(lon2 - lon1)
            a = np.sin(dist_lat/2) * np.sin(dist_lat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dist_lon/2) * np.sin(dist_lon/2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            distance = R * c
            return distance
        except ValueError as e:
            print(f"Error de valor en haversine_distance: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado en haversine_distance: {e}")
            return None


    def get_area_polygon(self, points: list[tuple[float, float]]):
        
        try:
            if not points or len(points) < 3:
                raise ValueError("Se necesitan al menos 3 puntos para calcular el área de un polígono")
            
            n = len(points)
            area = 0
            for i in range(n):
                j = (i + 1) % n
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            area = abs(area) / 2.0
            return area
        except ValueError as e:
            print(f"Error de valor en get_area_polygon: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado en get_area_polygon: {e}")
            return None

    def get_centroid(self, points: list[tuple[float, float]]):
        try:
            if not points:
                raise ValueError("La lista de puntos no puede estar vacía")
            
            x = sum(point[0] for point in points) / len(points)
            y = sum(point[1] for point in points) / len(points)
            return x, y
        except ValueError as e:
            print(f"Error de valor en get_centroid: {e}")
            return None, None
        except Exception as e:
            print(f"Error inesperado en get_centroid: {e}")
            return None, None

    def get_angle_centroid(self, point: tuple[float, float], centroid: tuple[float, float]):
        try:
            if not point or not centroid:
                raise ValueError("Tanto el punto como el centroide deben ser válidos")
            
            return math.atan2(point[1] - centroid[1], point[0] - centroid[0])
        except ValueError as e:
            print(f"Error de valor en get_angle_centroid: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado en get_angle_centroid: {e}")
            return None

    def order_points_dis(self, points: list[tuple[float, float]]):
        try:
            if not points or len(points) != 4:
                raise ValueError("Se necesitan exactamente 4 puntos para ordenar")
            
            centroid = self.get_centroid(points)
            if centroid[0] is None:
                raise ValueError("No se pudo calcular el centroide")
            
            points = sorted(points, key=lambda point: self.get_angle_centroid(point, centroid))

            return [points[0], points[2], points[1], points[3]]
        except ValueError as e:
            print(f"Error de valor en order_points_dis: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado en order_points_dis: {e}")
            return None

    def order_points(self, points):
        try:
            if not points or len(points) != 4:
                raise ValueError("Se necesitan exactamente 4 puntos para ordenar")
            
            print("ENTRO A ORDER POINTS")
            
            # Ordenar los puntos basándose en su coordenada x
            points = sorted(points, key=lambda point: point[0])

            # Separar los puntos en dos grupos basados en su posición x
            left_points = points[:2]
            right_points = points[2:]

            # Dentro de cada grupo, ordenarlos por su coordenada y
            left_points = sorted(left_points, key=lambda point: point[1])
            right_points = sorted(right_points, key=lambda point: point[1], reverse=True)

            # El orden final es: superior izquierdo, inferior izquierdo, inferior derecho, superior derecho
            return [left_points[0], left_points[1], right_points[0], right_points[1]]
        except ValueError as e:
            print(f"Error de valor en order_points: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado en order_points: {e}")
            return None
    
    def draw_segmented_image(self, data_image: np.ndarray, points: list[tuple[float, float]]):
        try:
            if data_image is None:
                raise ValueError("La imagen de datos no puede ser None")
            
            if not points or len(points) != 4:
                raise ValueError("Se necesitan exactamente 4 puntos para dibujar la segmentación")
            
            data_image_copy = data_image.copy()
            points_np = np.array(points, np.int32)
            points_np = points_np.reshape((-1, 1, 2))
            cv2.polylines(data_image_copy, [points_np], isClosed=True, color=(0, 255, 0), thickness=3)

            cv2.circle(data_image_copy, (points[0][0], points[0][1]), 5, (0, 0, 255), -1)
            cv2.circle(data_image_copy, (points[1][0], points[1][1]), 5, (255, 0, 255), -1)
            cv2.circle(data_image_copy, (points[2][0], points[2][1]), 5, (255, 0, 0), -1)
            cv2.circle(data_image_copy, (points[3][0], points[3][1]), 5, (255, 255, 0), -1)
            
            return data_image_copy
        except ValueError as e:
            print(f"Error de valor en draw_segmented_image: {e}")
            return None
        except Exception as e:
            print(f"Error inesperado en draw_segmented_image: {e}")
            return None
        
