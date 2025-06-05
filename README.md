# Algoritmos Genéticos - Trabajo Práctico N° 1

Este repositorio contiene la implementación en Python del **Algoritmo Genético Canónico** aplicado a la función:

$$ f(x) = (x / coef)² $$ 

donde 

$$ coef = 2^{30} - 1, x ∈ [0, 2^{30} - 1] $$

El trabajo incluye las tres variantes requeridas por el enunciado:  
- **Opción A**: Selección por **Ruleta**  
- **Opción B**: Selección por **Torneo**  
- **Opción C**: Aplicación de **Elitismo**

## 📂 Estructura del Repositorio

Cada variante se encuentra implementada en un archivo Python independiente.  
Se pueden ejecutar directamente desde la terminal para probar las versiones con 20, 100 y 200 generaciones (excepto en la opción C, que trabaja con 100 iteraciones fijas, como indica el enunciado).

## ▶️ Ejecución

### Windows
```bash
python nombre_del_archivo.py
```
### Linux / Mac
```bash
python3 nombre_del_archivo.py
```
## 🗂️ Archivos disponibles
- ex_1A_ruleta.py
- ex_1B_torneo.py
- ex_1C_ruleta_elitismo.py
- ex_1C_torneo_elitismo.py
> Asegúrese de tener los archivos en el mismo directorio donde ejecuta los comandos.

## 📦 Requisitos
Antes de ejecutar los scripts, instale las dependencias necesarias:
### Windows
```bash
pip install matplotlib pandas openpyxl
```
### Linux / Mac
```bash
pip3 install matplotlib pandas openpyxl
```
## 🧾 Salida del programa
Cada ejecución genera:
- Un directorio automático con timestamp, por ejemplo: resultados_AG_ruleta_20250605_153010
- Un archivo Excel con estadísticas completas de cada corrida
- Gráficas de evolución del fitness por generación
- Datos del mejor individuo encontrado: fitness, cromosoma binario y valor decimal
> Las salidas cumplen con los requerimientos del enunciado: mostrar los valores máximo, mínimo y promedio de cada población, y permitir comparaciones entre diferentes configuraciones.
>
## 📌 Notas
- Los parámetros (crossover, mutación, longitud, etc.) están configurados según los valores indicados en el enunciado.
- Puede modificar fácilmente estos valores directamente desde el código fuente si desea experimentar con diferentes configuraciones.

Este proyecto fue desarrollado como anexo al informe principal del Trabajo Práctico N°1 de Algoritmos Genéticos.
