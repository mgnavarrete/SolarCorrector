# TODO

## 1. Calcular el ángulo real de los paneles en el lado largo (corrección de yaw)

- Revisar cómo se calcula actualmente el ángulo de los paneles.
- Analizar si se está usando la orientación correcta (lado largo vs. lado corto).
- Implementar una función que detecte el lado largo de los paneles y calcule el ángulo real respecto a una referencia (por ejemplo, el norte).
- Integrar este cálculo en la función de corrección de yaw.

## 2. Mejorar la función FindFlights para imágenes orientadas al este

- Analizar el código actual de FindFlights para entender por qué falla con esa orientación.
- Explorar el uso de segmentación de paneles para detectar los vuelos automáticamente:
  - Usar la segmentación para identificar la posición y orientación de los paneles.
  - Inferir la dirección de vuelo a partir de la disposición de los paneles segmentados.
- Considerar otras alternativas, como el análisis de metadatos de las imágenes o el uso de patrones de movimiento.
