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
NUM_GENERACIONES = 100
ELITE_SIZE = 2

def generar_poblacion(n, longitud):
    """Genera población inicial de n cromosomas binarios de longitud especificada"""
    poblacion = []
    for _ in range(n):
        cromosoma = [random.randint(0, 1) for _ in range(longitud)]
        poblacion.append(cromosoma)
    return poblacion

def binario_a_entero(cromosoma):
    """Convierte cromosoma binario a valor entero"""
    return sum(bit * (2 ** (LONGITUD - 1 - i)) for i, bit in enumerate(cromosoma))

def evaluar_funcion_objetivo(valor):
    """Evalúa la función objetivo f(x) = (x/coef)²"""
    return (valor / COEF) ** 2

def calcular_fitness(valores_f_obj):
    """Calcula fitness normalizado (proporcional) de cada individuo"""
    total = sum(valores_f_obj)
    fitness = [y / total for y in valores_f_obj]
    return fitness

def calcular_prob_acumuladas(fitness_valores):
    """Calcula probabilidades acumuladas para selección por ruleta"""
    prob_acumuladas = []
    acumulada = 0
    for fitness in fitness_valores:
        acumulada += fitness
        prob_acumuladas.append(acumulada)
    return prob_acumuladas

def ruleta_seleccion(prob_acumuladas):
    """Selección por ruleta: devuelve índice del cromosoma seleccionado"""
    r = random.random()
    for indice, acumulada in enumerate(prob_acumuladas):
        if r <= acumulada:
            return indice
    return len(prob_acumuladas) - 1

def crossover(padre1, padre2, probabilidad):
    """Crossover de 1 punto entre dos padres"""
    r = random.random()
    if r < probabilidad:
        punto_corte = random.randint(1, len(padre1) - 1)
        hijo1 = padre1[:punto_corte] + padre2[punto_corte:]
        hijo2 = padre2[:punto_corte] + padre1[punto_corte:]
        return hijo1, hijo2, True, punto_corte
    else:
        return padre1[:], padre2[:], False, None

def mutacion(hijo1, hijo2, probabilidad):
    """Mutación en ambos hijos"""
    hubo_mutacion = False
    hijo1_copia = hijo1[:]
    hijo2_copia = hijo2[:]

    r1 = random.random()
    if r1 < probabilidad:
        gen_mutado = random.randint(0, len(hijo1_copia) - 1)
        # Si es 1 muta a 0 y si es 0 muta a 1
        hijo1_copia[gen_mutado] = 1 - hijo1_copia[gen_mutado]
        hubo_mutacion = True

    r2 = random.random()
    if r2 < probabilidad:
        gen_mutado = random.randint(0, len(hijo2_copia) - 1)
        hijo2_copia[gen_mutado] = 1 - hijo2_copia[gen_mutado]
        hubo_mutacion = True

    return hijo1_copia, hijo2_copia, hubo_mutacion

def seleccionar_elite(poblacion, elite_size):
    """
    Selecciona los mejores cromosomas (élite) basado en su fitness.
    
    Args:
        poblacion: Lista de cromosomas
        elite_size: Número de individuos a seleccionar para la élite
    
    Returns:
        elite: Lista con los mejores cromosomas
        indices_elite: Índices de los cromosomas élite en la población original
    """
    # Calcular fitness de toda la población
    enteros = [binario_a_entero(x) for x in poblacion]
    f_obj = [evaluar_funcion_objetivo(x) for x in enteros]
    
    # Crear lista de tuplas (fitness, índice, cromosoma)
    fitness_con_indices = [(f_obj[i], i, poblacion[i]) for i in range(len(poblacion))]
    
    # Ordenar por fitness de mayor a menor
    fitness_con_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Seleccionar los mejores
    elite = [cromosoma for _, _, cromosoma in fitness_con_indices[:elite_size]]
    indices_elite = [indice for _, indice, _ in fitness_con_indices[:elite_size]]
    
    return elite, indices_elite


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

def evolucionar_generacion_con_elitismo(poblacion, elite_size):
    """
    Evoluciona una generación aplicando elitismo.
    
    Proceso:
    1. Selecciona élite (mejores cromosomas) y los copia directamente
    2. Para el resto de la población, aplica selección por ruleta, crossover y mutación
    3. Combina élite + nuevos individuos = nueva población
    
    Args:
        poblacion: Población actual
        elite_size: Tamaño de la élite
    
    Returns:
        nueva_poblacion: Nueva población con elitismo aplicado
    """
    # Paso 1: Seleccionar élite
    elite, indices_elite = seleccionar_elite(poblacion, elite_size)
    
    # Paso 2: Calcular fitness para selección por ruleta del resto
    enteros = [binario_a_entero(x) for x in poblacion]
    f_obj = [evaluar_funcion_objetivo(x) for x in enteros]
    fitness_valores = calcular_fitness(f_obj)
    prob_acumuladas = calcular_prob_acumuladas(fitness_valores)
    
    # Paso 3: Generar el resto de la población (NUM_CROMOSOMAS - elite_size)
    nuevos_individuos = []
    individuos_a_generar = NUM_CROMOSOMAS - elite_size
    
    # Asegurar que generamos un número par de individuos (por el crossover)
    pares_a_generar = individuos_a_generar // 2
    
    for _ in range(pares_a_generar):
        # Selección por ruleta
        padre1_idx = ruleta_seleccion(prob_acumuladas)
        padre2_idx = ruleta_seleccion(prob_acumuladas)
        
        padre1 = poblacion[padre1_idx]
        padre2 = poblacion[padre2_idx]
        
        # Crossover
        hijo1, hijo2, _, _ = crossover(padre1, padre2, PROBABILIDAD_CROSSOVER)
        
        # Mutación
        hijo1_final, hijo2_final, _ = mutacion(hijo1, hijo2, PROBABILIDAD_MUTACION)
        
        nuevos_individuos.extend([hijo1_final, hijo2_final])
    
    # Si el número de individuos a generar es impar, generar uno más
    if individuos_a_generar % 2 == 1:
        padre1_idx = ruleta_seleccion(prob_acumuladas)
        padre2_idx = ruleta_seleccion(prob_acumuladas)
        
        padre1 = poblacion[padre1_idx]
        padre2 = poblacion[padre2_idx]
        
        hijo1, _, _, _ = crossover(padre1, padre2, PROBABILIDAD_CROSSOVER)
        hijo1_final, _, _ = mutacion(hijo1, [0]*LONGITUD, PROBABILIDAD_MUTACION)
        
        nuevos_individuos.append(hijo1_final)
    
    # Paso 4: Combinar élite + nuevos individuos
    nueva_poblacion = elite + nuevos_individuos
    
    return nueva_poblacion

def ejecutar_algoritmo_genetico_elitismo(num_generaciones, elite_size):
    """
    Ejecuta el algoritmo genético con elitismo por N generaciones.
    
    Args:
        num_generaciones: Número de generaciones a ejecutar
        elite_size: Tamaño de la élite (individuos que pasan sin modificación)
    
    Returns:
        estadisticas: Diccionario con estadísticas por generación
        mejor_fitness_global: Mejor fitness encontrado
        mejor_cromosoma_global: Mejor cromosoma encontrado
        mejor_valor_global: Valor decimal del mejor cromosoma
        mejor_generacion_global: Generación donde se encontró el mejor
    """
    print(f"="*80)
    print(f"AG CON ELITISMO - {num_generaciones} GENERACIONES")
    print(f"="*80)

    print(f"Parámetros:")
    print(f" - Población: {NUM_CROMOSOMAS} cromosomas")
    print(f" - Élite: {elite_size} cromosomas")
    print(f" - Nuevos por generación: {NUM_CROMOSOMAS - elite_size} cromosomas")
    print(f" - Longitud cromosoma: {LONGITUD} bits")
    print(f" - Prob. crossover: {PROBABILIDAD_CROSSOVER}")
    print(f" - Prob. mutación: {PROBABILIDAD_MUTACION}")
    print(f" - Generaciones: {num_generaciones}")
    print(f" - Coeficiente: {COEF}")

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

        # Actualiza mejor global
        if stats['mejor_fitness'] > mejor_fitness_global:
            mejor_fitness_global = stats['mejor_fitness']
            mejor_cromosoma_global = stats['mejor_cromosoma'].copy()
            mejor_valor_global = stats['mejor_valor']
            mejor_generacion_global = gen + 1

        # Evoluciona población (excepto en la última generación)
        if gen < num_generaciones - 1:
            poblacion = evolucionar_generacion_con_elitismo(poblacion, elite_size)
    
    return estadisticas, mejor_fitness_global, mejor_cromosoma_global, mejor_valor_global, mejor_generacion_global

def crear_directorio_resultados():
    """Crea directorio para almacenar resultados si no existe"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directorio = f"resultados_AG_elitismo_{timestamp}"
    
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    return directorio

def exportar_estadisticas_csv(estadisticas, num_generaciones, directorio, mejor_cromosoma_info, elite_size):
    """Exporta estadísticas a archivo CSV"""
    filename = f"{directorio}/estadisticas_elitismo_{num_generaciones}_gen_elite{elite_size}.csv"
    
    df = pd.DataFrame(estadisticas)
    df.to_csv(filename, index=False, float_format='%.10f')
    
    # Archivo adicional con información del mejor cromosoma
    mejor_info_filename = f"{directorio}/mejor_cromosoma_elitismo_{num_generaciones}_gen_elite{elite_size}.csv"
    mejor_df = pd.DataFrame([{
        'Generaciones_Ejecutadas': num_generaciones,
        'Elite_Size': elite_size,
        'Mejor_Generacion': mejor_cromosoma_info['mejor_generacion'],
        'Mejor_Fitness': mejor_cromosoma_info['mejor_fitness'],
        'Mejor_Valor_Decimal': mejor_cromosoma_info['mejor_valor'],
        'Mejor_Valor_Normalizado': mejor_cromosoma_info['mejor_valor'] / COEF,
        'Mejor_Cromosoma': ''.join(map(str, mejor_cromosoma_info['mejor_cromosoma']))
    }])
    mejor_df.to_csv(mejor_info_filename, index=False, float_format='%.10f')
    
    print(f"✓ Estadísticas exportadas a: {filename}")
    print(f"✓ Mejor cromosoma exportado a: {mejor_info_filename}")

def mostrar_tabla_estadisticas(estadisticas, num_generaciones):
    """Muestra tabla con estadísticas por generación"""
    print(f"\n{'='*80}")
    print("TABLA DE ESTADÍSTICAS POR GENERACIÓN (CON ELITISMO)")
    print("="*80)

    df = pd.DataFrame(estadisticas)

    # Muestra tabla completa para generaciones pequeñas
    # Muestra resumida para grandes generaciones
    if num_generaciones <= 20:
        print(df.to_string(index=False, float_format='%.6f'))
    else:
        # Muestra primeras 10, últimas 10 y algunas intermedias
        print("Primeras 10 generaciones:")
        print(df.head(10).to_string(index=False, float_format='%.6f'))

        if num_generaciones > 40:
            print(f"\n... (generaciones intermedias omitidas) ...\n")

            # Generaciones intermedias
            medio_inicio = num_generaciones // 2 - 2
            medio_fin = num_generaciones // 2 + 3
            print(f"Generaciones ({medio_inicio}-{medio_fin}):")
            print(df.iloc[medio_inicio-1:medio_fin].to_string(index=False, float_format='%.6f'))

        print(f"\nÚltimas 10 generaciones:")
        print(df.tail(10).to_string(index=False, float_format='%.6f'))

def generar_graficas_elitismo(estadisticas, num_generaciones, directorio, elite_size):
    """Genera gráficas de evolución del fitness con elitismo y las guarda"""
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
    plt.title(f'Evolución del Fitness con Elitismo (Elite={elite_size}, {num_generaciones} gen.)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfica 2: Solo mejor fitness
    plt.subplot(2, 2, 2)
    plt.plot(estadisticas['generacion'], estadisticas['mejor_fitness'], 
             'g-', linewidth=2, marker='o', markersize=3)
    plt.xlabel('Generación')
    plt.ylabel('Mejor Fitness')
    plt.title('Evolución del Mejor Fitness (Con Elitismo)')
    plt.grid(True, alpha=0.3)
    
    # Gráfica 3: Diferencia entre mejor y peor
    diferencias = [mejor - peor for mejor, peor in 
                   zip(estadisticas['mejor_fitness'], estadisticas['peor_fitness'])]
    plt.subplot(2, 2, 3)
    plt.plot(estadisticas['generacion'], diferencias, 
             'm-', linewidth=2)
    plt.xlabel('Generación')
    plt.ylabel('Diferencia (Mejor - Peor)')
    plt.title('Diversidad de la Población (Con Elitismo)')
    plt.grid(True, alpha=0.3)

    # Gráfica 4: Convergencia (últimas 20 generaciones)
    plt.subplot(2, 2, 4)
    ultimas_20 = min(20, num_generaciones)
    gen_finales = estadisticas['generacion'][-ultimas_20:]
    fitness_finales = estadisticas['mejor_fitness'][-ultimas_20:]
    
    plt.plot(gen_finales, fitness_finales, 'ro-', linewidth=2, markersize=4)
    plt.xlabel('Generación')
    plt.ylabel('Mejor Fitness')
    plt.title(f'Convergencia (Últimas {ultimas_20} generaciones)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guarda la gráfica
    grafica_filename = f"{directorio}/graficas_elitismo_{num_generaciones}_gen_elite{elite_size}.png"
    plt.savefig(grafica_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Gráficas guardadas en: {grafica_filename}")

def mostrar_mejor_cromosoma(mejor_fitness, mejor_cromosoma, mejor_valor, mejor_generacion):
    """Muestra información del mejor cromosoma encontrado"""
    print(f"\n{'='*80}")
    print("MEJOR CROMOSOMA ENCONTRADO (CON ELITISMO)")
    print("="*80)
    
    print(f"Generación donde se encontró: {mejor_generacion}")
    print(f"Cromosoma: {mejor_cromosoma}")
    print(f"Valor decimal: {mejor_valor}")
    print(f"Valor normalizado: {mejor_valor / COEF:.10f}")
    print(f"Fitness: {mejor_fitness:.10f}")
    print(f"Función objetivo f(x) = (x/{COEF})² = {mejor_fitness:.10f}")

def exportar_estadisticas_excel_elitismo(estadisticas, directorio, num_generaciones, elite_size, mejor_info):
    """Exporta estadísticas de elitismo a archivo Excel"""
    filename = f"{directorio}/estadisticas_elitismo_completas.xlsx"
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Hoja de estadísticas por generación
        estadisticas_df = pd.DataFrame(estadisticas)
        estadisticas_df.to_excel(writer, sheet_name='Estadisticas_Elitismo', index=False)
        
        # Hoja de resumen
        resumen_df = pd.DataFrame([{
            'Generaciones': num_generaciones,
            'Elite_Size': elite_size,
            'Mejor_Fitness_Final': estadisticas['mejor_fitness'][-1],
            'Peor_Fitness_Final': estadisticas['peor_fitness'][-1],
            'Fitness_Promedio_Final': estadisticas['fitness_promedio'][-1],
            'Mejor_Fitness_Global': mejor_info['mejor_fitness'],
            'Mejor_Generacion_Global': mejor_info['mejor_generacion'],
            'Mejor_Valor_Decimal': mejor_info['mejor_valor'],
            'Mejor_Cromosoma': ''.join(map(str, mejor_info['mejor_cromosoma']))
        }])
        resumen_df.to_excel(writer, sheet_name='Resumen_Elitismo', index=False)
        
        # Hoja de parámetros utilizados
        parametros_df = pd.DataFrame([{
            'Parametro': 'Numero_Cromosomas',
            'Valor': NUM_CROMOSOMAS
        }, {
            'Parametro': 'Elite_Size',
            'Valor': elite_size
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
            'Parametro': 'Metodo_Seleccion',
            'Valor': 'Elitismo + Ruleta'
        }, {
            'Parametro': 'Metodo_Crossover',
            'Valor': '1 Punto'
        }, {
            'Parametro': 'Estrategia',
            'Valor': f'Elitismo (top {elite_size}) + Evolución resto'
        }])
        parametros_df.to_excel(writer, sheet_name='Parametros', index=False)
    
    print(f"✓ Archivo Excel completo exportado a: {filename}")

def main():
    """
    Función principal que ejecuta el algoritmo genético con elitismo.
    
    Configuración fija:
    - 100 generaciones
    - Élite de 2 individuos
    - Mismos parámetros que Parte A
    """
    print(f"{'#'*100}")
    print(f"ALGORITMO GENÉTICO CON ELITISMO - PARTE C")
    print(f"{'#'*100}")
    
    # Configuración para Parte C
    num_generaciones = 100
    elite_size = ELITE_SIZE
    
    # Crear directorio para resultados
    directorio = crear_directorio_resultados()
    print(f"📁 Resultados se guardarán en: {directorio}")
    
    print(f"\n🚀 Iniciando ejecución con elitismo...")
    print(f"   - Generaciones: {num_generaciones}")
    print(f"   - Tamaño élite: {elite_size}")
    print(f"   - Población total: {NUM_CROMOSOMAS}")
    
    # Ejecutar algoritmo con elitismo
    estadisticas, mejor_fitness, mejor_cromosoma, mejor_valor, mejor_generacion = ejecutar_algoritmo_genetico_elitismo(
        num_generaciones, elite_size
    )
    
    # Mostrar tabla de estadísticas
    mostrar_tabla_estadisticas(estadisticas, num_generaciones)
    
    # Mostrar mejor cromosoma
    mostrar_mejor_cromosoma(mejor_fitness, mejor_cromosoma, mejor_valor, mejor_generacion)
    
    # Exportar estadísticas a CSV
    mejor_cromosoma_info = {
        'mejor_fitness': mejor_fitness,
        'mejor_cromosoma': mejor_cromosoma,
        'mejor_valor': mejor_valor,
        'mejor_generacion': mejor_generacion
    }
    exportar_estadisticas_csv(estadisticas, num_generaciones, directorio, mejor_cromosoma_info, elite_size)
    
    # Generar y guardar gráficas
    print(f"\nGenerando gráficas para {num_generaciones} generaciones con elitismo...")
    generar_graficas_elitismo(estadisticas, num_generaciones, directorio, elite_size)
    
    # Exportar archivo Excel completo
    print(f"\n{'='*80}")
    print("EXPORTANDO ARCHIVO EXCEL COMPLETO")
    print("="*80)
    exportar_estadisticas_excel_elitismo(estadisticas, directorio, num_generaciones, elite_size, mejor_cromosoma_info)
    
    print(f"\n🎉 PROCESO COMPLETADO - PARTE C (ELITISMO)")
    print(f"📊 Todos los archivos de resultados están en: {directorio}")
    print(f"   - Archivos CSV con estadísticas y mejor cromosoma")
    print(f"   - Archivo Excel completo con todas las estadísticas")
    print(f"   - Gráficas en formato PNG")
    print(f"\n💡 ELITISMO APLICADO:")
    print(f"   - {elite_size} mejores cromosomas pasan directamente cada generación")
    print(f"   - {NUM_CROMOSOMAS - elite_size} cromosomas generados por evolución (ruleta + crossover + mutación)")

if __name__ == "__main__":
    # Ejecutar
    main()
    
    # Para ejecutar solo una prueba específica (descomentar si es necesario):
    # directorio = crear_directorio_resultados()
    # estadisticas, mejor_fitness, mejor_cromosoma, mejor_valor, mejor_generacion = ejecutar_algoritmo_genetico_elitismo(100, 2)
    # mostrar_tabla_estadisticas(estadisticas, 100)
    # mostrar_mejor_cromosoma(mejor_fitness, mejor_cromosoma, mejor_valor, mejor_generacion)
    # mejor_cromosoma_info = {'mejor_fitness': mejor_fitness, 'mejor_cromosoma': mejor_cromosoma, 'mejor_valor': mejor_valor, 'mejor_generacion': mejor_generacion}
    # exportar_estadisticas_csv(estadisticas, 100, directorio, mejor_cromosoma_info, 2)
    # generar_graficas_elitismo(estadisticas, 100, directorio, 2)