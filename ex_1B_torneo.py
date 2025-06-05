import random
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

COEF = 2**30 - 1
NUM_CROMOSOMAS = 10
LONGITUD = 30
PROBABILIDAD_CROSSOVER = 0.75
PROBABILIDAD_MUTACION = 0.05

def generar_poblacion(n, longitud):
    """Genera población inicial de n cromosomas binarios"""
    poblacion = []
    for _ in range(n):
        cromosoma = [random.randint(0, 1) for _ in range(longitud)]
        poblacion.append(cromosoma)
    return poblacion

def binario_a_entero(cromosoma):
    """Convierte cromosoma binario a entero"""
    return sum(gen * (2 ** (LONGITUD - 1 - i)) for i, gen in enumerate(cromosoma))

def evaluar_funcion_objetivo(valor):
    """Función objetivo: f(x) = (x/COEF)²"""
    return (valor / COEF) ** 2

def torneo_seleccion(poblacion, f_obj, k=2):
    """Selección por torneo con k=2"""
    candidatos = random.sample(range(len(poblacion)), k)
    mejor_idx = max(candidatos, key=lambda idx: f_obj[idx])
    return mejor_idx

def crossover(padre1, padre2, probabilidad):
    """Crossover de un punto"""
    r = random.random()
    if r < probabilidad:
        punto_corte = random.randint(1, len(padre1) - 1)
        hijo1 = padre1[:punto_corte] + padre2[punto_corte:]
        hijo2 = padre2[:punto_corte] + padre1[punto_corte:]
        return hijo1, hijo2, True, punto_corte
    else:
        return padre1[:], padre2[:], False, None

def mutacion(cromosoma, probabilidad):
    """Mutación que afecta solo un bit aleatorio si ocurre mutación"""
    cromosoma_copia = cromosoma[:]
    hubo_mutacion = False
    
    r = random.random()
    if r < probabilidad:
        gen_mutado = random.randint(0, len(cromosoma_copia) - 1)
        cromosoma_copia[gen_mutado] = 1 - cromosoma_copia[gen_mutado]
        hubo_mutacion = True
    
    return cromosoma_copia, hubo_mutacion

def obtener_estadisticas_poblacion(poblacion):
    """Calcula estadísticas de fitness de la población actual"""
    enteros = [binario_a_entero(x) for x in poblacion]
    f_obj = [evaluar_funcion_objetivo(x) for x in enteros]

    mejor_fitness = max(f_obj)
    peor_fitness = min(f_obj)
    fitness_promedio = sum(f_obj) / len(f_obj)

    mejor_idx = f_obj.index(mejor_fitness)
    mejor_cromosoma = poblacion[mejor_idx]
    mejor_valor = enteros[mejor_idx]

    return {
        'mejor_fitness': mejor_fitness,
        'peor_fitness': peor_fitness,
        'fitness_promedio': fitness_promedio,
        'mejor_cromosoma': mejor_cromosoma,
        'mejor_valor': mejor_valor
    }

def evolucionar_generacion(poblacion):
    """Evoluciona una generación de la población"""
    enteros = [binario_a_entero(x) for x in poblacion]
    f_obj = [evaluar_funcion_objetivo(x) for x in enteros]

    nueva_poblacion = []

    # Generar toda la población
    while len(nueva_poblacion) < NUM_CROMOSOMAS:
        # Selección por torneo
        padre1_idx = torneo_seleccion(poblacion, f_obj)
        padre2_idx = torneo_seleccion(poblacion, f_obj)

        padre1 = poblacion[padre1_idx]
        padre2 = poblacion[padre2_idx]

        # Crossover
        hijo1, hijo2, _, _ = crossover(padre1, padre2, PROBABILIDAD_CROSSOVER)

        # Mutación
        hijo1_final, _ = mutacion(hijo1, PROBABILIDAD_MUTACION)
        hijo2_final, _ = mutacion(hijo2, PROBABILIDAD_MUTACION)

        nueva_poblacion.append(hijo1_final)
        
        # Solo añadir el segundo hijo si no completamos la población
        if len(nueva_poblacion) < NUM_CROMOSOMAS:
            nueva_poblacion.append(hijo2_final)

    return nueva_poblacion

def ejecutar_algoritmo_genetico(num_generaciones):
    """Ejecuta el algoritmo genético por N generaciones"""
    # Inicializa población
    poblacion = generar_poblacion(NUM_CROMOSOMAS, LONGITUD)

    # Listas para almacenar estadísticas
    estadisticas = {
        'generacion': [],
        'mejor_fitness': [],
        'peor_fitness': [],
        'fitness_promedio': []
    }

    mejor_fitness_global = 0
    mejor_cromosoma_global = None
    mejor_valor_global = 0
    mejor_generacion_global = 0

    # Ejecuta generaciones
    for gen in range(num_generaciones):
        stats = obtener_estadisticas_poblacion(poblacion)

        # Guarda estadísticas
        estadisticas['generacion'].append(gen + 1)
        estadisticas['mejor_fitness'].append(stats['mejor_fitness'])
        estadisticas['peor_fitness'].append(stats['peor_fitness'])
        estadisticas['fitness_promedio'].append(stats['fitness_promedio'])

        if stats['mejor_fitness'] > mejor_fitness_global:
            mejor_fitness_global = stats['mejor_fitness']
            mejor_cromosoma_global = stats['mejor_cromosoma'].copy()
            mejor_valor_global = stats['mejor_valor']
            mejor_generacion_global = gen + 1

        if gen < num_generaciones - 1:
            poblacion = evolucionar_generacion(poblacion)
    
    return estadisticas, mejor_fitness_global, mejor_cromosoma_global, mejor_valor_global, mejor_generacion_global

def crear_directorio_resultados():
    """Crea directorio para almacenar resultados si no existe"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directorio = f"resultados_AG_torneo_{timestamp}"
    
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    return directorio

def exportar_estadisticas_excel(todas_estadisticas, directorio):
    """Exporta todas las estadísticas a un archivo Excel con múltiples hojas"""
    filename = f"{directorio}/estadisticas_torneo_completas.xlsx"
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Hoja de resumen
        resumen_data = []
        for generaciones, data in todas_estadisticas.items():
            estadisticas = data['estadisticas']
            mejor_info = data['mejor_info']
            
            resumen_data.append({
                'Generaciones': generaciones,
                'Mejor_Fitness_Final': estadisticas['mejor_fitness'][-1],
                'Peor_Fitness_Final': estadisticas['peor_fitness'][-1],
                'Fitness_Promedio_Final': estadisticas['fitness_promedio'][-1],
                'Mejor_Fitness_Global': mejor_info['mejor_fitness'],
                'Mejor_Generacion_Global': mejor_info['mejor_generacion'],
                'Mejor_Valor_Decimal': mejor_info['mejor_valor'],
                'Mejor_Cromosoma': ''.join(map(str, mejor_info['mejor_cromosoma']))
            })
        
        resumen_df = pd.DataFrame(resumen_data)
        resumen_df.to_excel(writer, sheet_name='Resumen_General', index=False)
        
        # Hoja para cada configuración de generaciones
        for generaciones, data in todas_estadisticas.items():
            estadisticas_df = pd.DataFrame(data['estadisticas'])
            sheet_name = f'Estadisticas_{generaciones}gen'
            estadisticas_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Hoja de parámetros utilizados
        parametros_df = pd.DataFrame([{
            'Parametro': 'Numero_Cromosomas',
            'Valor': NUM_CROMOSOMAS
        }, {
            'Parametro': 'Longitud_Cromosoma',
            'Valor': LONGITUD
        }, {
            'Parametro': 'Probabilidad_Crossover',
            'Valor': PROBABILIDAD_CROSSOVER
        }, {
            'Parametro': 'Probabilidad_Mutacion',
            'Valor': PROBABILIDAD_MUTACION
        }, {
            'Parametro': 'Coeficiente',
            'Valor': COEF
        }, {
            'Parametro': 'Funcion_Objetivo',
            'Valor': 'f(x) = (x/coef)²'
        }, {
            'Parametro': 'Dominio',
            'Valor': f'[0, {COEF}]'
        }, {
            'Parametro': 'Metodo_Seleccion',
            'Valor': 'Torneo (k=2)'
        }, {
            'Parametro': 'Metodo_Crossover',
            'Valor': '1 Punto'
        }, {
            'Parametro': 'Mutacion',
            'Valor': '1 bit aleatorio por cromosoma'
        }])
        parametros_df.to_excel(writer, sheet_name='Parametros', index=False)

    print(f"✓ Archivo Excel completo exportado a: {filename}")    
    return filename

def generar_graficas(estadisticas, num_generaciones, directorio):
    """Genera gráficas de evolución del fitness y las guarda"""
    plt.figure(figsize=(15, 10))

    # Gráfica 1: Evolución del fitness
    plt.subplot(2, 2, 1)
    plt.plot(estadisticas['generacion'], estadisticas['mejor_fitness'], 
             'g-', linewidth=2, label='Mejor Fitness')
    plt.plot(estadisticas['generacion'], estadisticas['fitness_promedio'], 
             'b-', linewidth=2, label='Fitness Promedio')
    plt.plot(estadisticas['generacion'], estadisticas['peor_fitness'], 
             'r-', linewidth=2, label='Peor Fitness')
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title(f'Evolución del Fitness - Selección por Torneo ({num_generaciones} generaciones)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfica 2: Solo mejor fitness
    plt.subplot(2, 2, 2)
    plt.plot(estadisticas['generacion'], estadisticas['mejor_fitness'], 
             'g-', linewidth=2, marker='o', markersize=3)
    plt.xlabel('Generación')
    plt.ylabel('Mejor Fitness')
    plt.title('Evolución del Mejor Fitness')
    plt.grid(True, alpha=0.3)
    
    # Gráfica 3: Diferencia entre mejor y peor
    diferencias = [mejor - peor for mejor, peor in 
                   zip(estadisticas['mejor_fitness'], estadisticas['peor_fitness'])]
    plt.subplot(2, 2, 3)
    plt.plot(estadisticas['generacion'], diferencias, 
             'm-', linewidth=2)
    plt.xlabel('Generación')
    plt.ylabel('Diferencia (Mejor - Peor)')
    plt.title('Diversidad de la Población')
    plt.grid(True, alpha=0.3)

    # Gráfica 4: Distribución final vs inicial
    plt.subplot(2, 2, 4)
    generaciones_muestra = [1, num_generaciones//4, num_generaciones//2, 
                           3*num_generaciones//4, num_generaciones]
    fitness_muestra = [estadisticas['mejor_fitness'][gen-1] for gen in generaciones_muestra if gen <= len(estadisticas['mejor_fitness'])]
    generaciones_muestra = generaciones_muestra[:len(fitness_muestra)]
    
    plt.bar(range(len(generaciones_muestra)), fitness_muestra, 
            color=['red', 'orange', 'yellow', 'lightgreen', 'green'][:len(fitness_muestra)])
    plt.xlabel('Momento de la Evolución')
    plt.ylabel('Mejor Fitness')
    plt.title('Mejor Fitness en Diferentes Momentos')
    plt.xticks(range(len(generaciones_muestra)), 
               [f'Gen {gen}' for gen in generaciones_muestra])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guarda la gráfica
    grafica_filename = f"{directorio}/graficas_torneo_{num_generaciones}_generaciones.png"
    plt.savefig(grafica_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Gráficas guardadas en: {grafica_filename}")
    plt.close()

def main():
    """Función principal que ejecuta el algoritmo para diferentes números de generaciones"""
    generaciones_a_probar = [20, 100, 200]
    todas_estadisticas = {}
    
    # Mostrar resumen inicial
    print("="*80)
    print("ALGORITMO GENÉTICO CON SELECCIÓN POR TORNEO")
    print("="*80)
    print(f"Parámetros:")
    print(f"  • Población: {NUM_CROMOSOMAS} cromosomas")
    print(f"  • Longitud cromosoma: {LONGITUD} bits")
    print(f"  • Prob. crossover: {PROBABILIDAD_CROSSOVER}")
    print(f"  • Prob. mutación: {PROBABILIDAD_MUTACION}")
    print(f"  • Coeficiente: {COEF}")
    print(f"  • Función objetivo: f(x) = (x/{COEF})²")
    print(f"  • Selección: Torneo (k=2)")
    print(f"  • Generaciones a probar: {generaciones_a_probar}")
    print("="*80)
    
    # Crear directorio para resultados
    directorio = crear_directorio_resultados()

    # Variables para almacenar el mejor resultado global
    mejor_fitness_absoluto = 0
    mejor_cromosoma_absoluto = None
    mejor_valor_absoluto = 0
    mejor_generacion_absoluta = 0
    mejor_configuracion = 0

    for num_gen in generaciones_a_probar:
        print(f"\nEjecutando {num_gen} generaciones...")
        
        # Ejecutar algoritmo
        estadisticas, mejor_fitness, mejor_cromosoma, mejor_valor, mejor_generacion = ejecutar_algoritmo_genetico(num_gen)
        
        # Actualizar mejor resultado global
        if mejor_fitness > mejor_fitness_absoluto:
            mejor_fitness_absoluto = mejor_fitness
            mejor_cromosoma_absoluto = mejor_cromosoma
            mejor_valor_absoluto = mejor_valor
            mejor_generacion_absoluta = mejor_generacion
            mejor_configuracion = num_gen
        
        # Almacenar para el archivo Excel
        todas_estadisticas[num_gen] = {
            'estadisticas': estadisticas,
            'mejor_info': {
                'mejor_fitness': mejor_fitness,
                'mejor_cromosoma': mejor_cromosoma,
                'mejor_valor': mejor_valor,
                'mejor_generacion': mejor_generacion
            }
        }
        
        # Generar y guardar gráficas
        generar_graficas(estadisticas, num_gen, directorio)
    
    # Exportar archivo Excel completo con todas las estadísticas
    filename = exportar_estadisticas_excel(todas_estadisticas, directorio)
    
    # Mostrar resumen final
    print("\n" + "="*80)
    print("RESUMEN FINAL DE LA EJECUCIÓN")
    print("="*80)
    print(f"Mejor resultado encontrado:")
    print(f"  • Configuración: {mejor_configuracion} generaciones")
    print(f"  • Generación donde se encontró: {mejor_generacion_absoluta}")
    print(f"  • Mejor fitness: {mejor_fitness_absoluto:.10f}")
    print(f"  • Cromosoma: {''.join(map(str, mejor_cromosoma_absoluto))}")
    print(f"  • Valor decimal: {mejor_valor_absoluto}")
    print(f"  • Archivo Excel generado: {filename}")
    print("="*80)


# Para ejecutar todas las pruebas (20, 100, 200 generaciones):
main()