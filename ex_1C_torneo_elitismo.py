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
NUM_ELITES = 2
NUM_GENERACIONES = 100

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

def obtener_mejores_individuos(poblacion, f_obj, num_elites):
    """
    Obtiene los mejores 'num_elites' individuos de la población (para elitismo múltiple)
    Retorna lista de tuplas (cromosoma_copia, fitness, índice_original)
    """
    # Crear lista de índices ordenados por fitness (descendente)
    indices_ordenados = sorted(range(len(f_obj)), key=lambda i: f_obj[i], reverse=True)
    
    elites = []
    for i in range(min(num_elites, len(poblacion))):
        idx = indices_ordenados[i]
        elite = {
            'cromosoma': poblacion[idx][:],  # Copia del cromosoma
            'fitness': f_obj[idx],
            'indice_original': idx
        }
        elites.append(elite)
    
    return elites

def obtener_estadisticas_poblacion(poblacion):
    """Calcula estadísticas de la población actual"""
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
    """
    Evoluciona una generación usando elitismo múltiple
    - Conserva los NUM_ELITES mejores individuos sin alteraciones
    - Genera el resto (NUM_CROMOSOMAS - NUM_ELITES) mediante selección, crossover y mutación
    """
    enteros = [binario_a_entero(x) for x in poblacion]
    f_obj = [evaluar_funcion_objetivo(x) for x in enteros]
    
    # ==================== ELITISMO MÚLTIPLE ====================
    # Obtener los mejores NUM_ELITES individuos
    elites = obtener_mejores_individuos(poblacion, f_obj, NUM_ELITES)
    
    # Inicializar nueva población con los elites (sin alteraciones)
    nueva_poblacion = []
    for elite in elites:
        nueva_poblacion.append(elite['cromosoma'])
    
    print(f"  → Elites conservados: {len(nueva_poblacion)} individuos")
    for i, elite in enumerate(elites):
        print(f"    Elite {i+1}: Fitness={elite['fitness']:.8f}, Valor={binario_a_entero(elite['cromosoma'])}")
    # ============================================================

    # Generar el resto de la población (NUM_CROMOSOMAS - NUM_ELITES individuos)
    individuos_restantes = NUM_CROMOSOMAS - NUM_ELITES
    print(f"  → Generando {individuos_restantes} nuevos individuos mediante selección/crossover/mutación")
    
    while len(nueva_poblacion) < NUM_CROMOSOMAS:
        # Selección por torneo
        padre1_idx = torneo_seleccion(poblacion, f_obj)
        padre2_idx = torneo_seleccion(poblacion, f_obj)

        padre1 = poblacion[padre1_idx]
        padre2 = poblacion[padre2_idx]

        # Crossover
        hijo1, hijo2, hubo_crossover, punto_corte = crossover(padre1, padre2, PROBABILIDAD_CROSSOVER)

        # Mutación
        hijo1_final, hubo_mutacion1 = mutacion(hijo1, PROBABILIDAD_MUTACION)
        hijo2_final, hubo_mutacion2 = mutacion(hijo2, PROBABILIDAD_MUTACION)

        nueva_poblacion.append(hijo1_final)
        
        # Solo añadir el segundo hijo si no completamos la población
        if len(nueva_poblacion) < NUM_CROMOSOMAS:
            nueva_poblacion.append(hijo2_final)

    return nueva_poblacion

def ejecutar_algoritmo_genetico():
    """Ejecuta el algoritmo genético con elitismo múltiple - Parte C"""
    print(f"="*80)
    print(f"ALGORITMO GENÉTICO CON ELITISMO MÚLTIPLE - PARTE C")
    print(f"="*80)

    print(f"Parámetros:")
    print(f" - Población: {NUM_CROMOSOMAS} cromosomas")
    print(f" - Longitud cromosoma: {LONGITUD} bits")
    print(f" - Prob. crossover: {PROBABILIDAD_CROSSOVER}")
    print(f" - Prob. mutación: {PROBABILIDAD_MUTACION}")
    print(f" - Generaciones: {NUM_GENERACIONES}")
    print(f" - Coeficiente: {COEF}")
    print(f" - Selección: Torneo (k=2)")
    print(f" - Elitismo: {NUM_ELITES} individuos")
    print(f" - Función objetivo: f(x) = (x/{COEF})²")

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
    print(f"\nIniciando evolución...")
    for gen in range(NUM_GENERACIONES):
        print(f"\nGeneración {gen + 1}/{NUM_GENERACIONES}")
        
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
            print(f"  ★ NUEVO MEJOR GLOBAL: Fitness={mejor_fitness_global:.8f} en generación {mejor_generacion_global}")

        print(f"  Mejor: {stats['mejor_fitness']:.8f}, Promedio: {stats['fitness_promedio']:.8f}, Peor: {stats['peor_fitness']:.8f}")

        # Evolucionar para la próxima generación (excepto en la última)
        if gen < NUM_GENERACIONES - 1:
            poblacion = evolucionar_generacion(poblacion)
    
    print(f"\n🎉 Evolución completada!")
    return estadisticas, mejor_fitness_global, mejor_cromosoma_global, mejor_valor_global, mejor_generacion_global

def crear_directorio_resultados():
    """Crea directorio para almacenar resultados si no existe"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directorio = f"resultados_AG_elitismo_parte_c_{timestamp}"
    
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    return directorio

def exportar_estadisticas_csv(estadisticas, directorio, mejor_cromosoma_info):
    """Exporta estadísticas a archivo CSV"""
    filename = f"{directorio}/estadisticas_elitismo_{NUM_ELITES}_individuos_{NUM_GENERACIONES}_generaciones.csv"
    
    df = pd.DataFrame(estadisticas)
    df.to_csv(filename, index=False, float_format='%.10f')
    
    # Archivo adicional con información del mejor cromosoma
    mejor_info_filename = f"{directorio}/mejor_cromosoma_elitismo_{NUM_ELITES}_individuos_{NUM_GENERACIONES}_generaciones.csv"
    mejor_df = pd.DataFrame([{
        'Generaciones_Ejecutadas': NUM_GENERACIONES,
        'Numero_Elites': NUM_ELITES,
        'Mejor_Generacion': mejor_cromosoma_info['mejor_generacion'],
        'Mejor_Fitness': mejor_cromosoma_info['mejor_fitness'],
        'Mejor_Valor_Decimal': mejor_cromosoma_info['mejor_valor'],
        'Mejor_Valor_Normalizado': mejor_cromosoma_info['mejor_valor'] / COEF,
        'Mejor_Cromosoma': ''.join(map(str, mejor_cromosoma_info['mejor_cromosoma']))
    }])
    mejor_df.to_csv(mejor_info_filename, index=False, float_format='%.10f')
    
    print(f"✓ Estadísticas exportadas a: {filename}")
    print(f"✓ Mejor cromosoma exportado a: {mejor_info_filename}")

def exportar_estadisticas_excel(estadisticas, mejor_cromosoma_info, directorio):
    """Exporta todas las estadísticas a un archivo Excel"""
    filename = f"{directorio}/estadisticas_elitismo_{NUM_ELITES}_individuos_completas.xlsx"
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Hoja de estadísticas por generación
        estadisticas_df = pd.DataFrame(estadisticas)
        estadisticas_df.to_excel(writer, sheet_name='Estadisticas_Generacion', index=False)
        
        # Hoja de resumen del mejor cromosoma
        resumen_df = pd.DataFrame([{
            'Generaciones_Ejecutadas': NUM_GENERACIONES,
            'Numero_Elites': NUM_ELITES,
            'Mejor_Fitness': mejor_cromosoma_info['mejor_fitness'],
            'Mejor_Generacion': mejor_cromosoma_info['mejor_generacion'],
            'Mejor_Valor_Decimal': mejor_cromosoma_info['mejor_valor'],
            'Mejor_Valor_Normalizado': mejor_cromosoma_info['mejor_valor'] / COEF,
            'Mejor_Cromosoma': ''.join(map(str, mejor_cromosoma_info['mejor_cromosoma'])),
            'Fitness_Final_Mejor': estadisticas['mejor_fitness'][-1],
            'Fitness_Final_Promedio': estadisticas['fitness_promedio'][-1],
            'Fitness_Final_Peor': estadisticas['peor_fitness'][-1]
        }])
        resumen_df.to_excel(writer, sheet_name='Resumen_Mejor', index=False)
        
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
            'Parametro': 'Numero_Generaciones',
            'Valor': NUM_GENERACIONES
        }, {
            'Parametro': 'Numero_Elites',
            'Valor': NUM_ELITES
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
            'Parametro': 'Elitismo',
            'Valor': f'Múltiple ({NUM_ELITES} individuos)'
        }, {
            'Parametro': 'Mutacion',
            'Valor': '1 bit aleatorio por cromosoma'
        }])
        parametros_df.to_excel(writer, sheet_name='Parametros', index=False)
    
    print(f"✓ Archivo Excel completo exportado a: {filename}")

def mostrar_tabla_estadisticas(estadisticas):
    """Muestra tabla con estadísticas por generación"""
    print(f"\n{'='*80}")
    print("TABLA DE ESTADÍSTICAS POR GENERACIÓN - PARTE C")
    print("="*80)

    df = pd.DataFrame(estadisticas)

    # Para 100 generaciones, mostrar selectivamente
    print("Primeras 10 generaciones:")
    print(df.head(10).to_string(index=False, float_format='%.8f'))

    print(f"\nGeneraciones intermedias (45-55):")
    print(df.iloc[44:55].to_string(index=False, float_format='%.8f'))

    print(f"\nÚltimas 10 generaciones:")
    print(df.tail(10).to_string(index=False, float_format='%.8f'))

def generar_graficas(estadisticas, directorio):
    """Genera gráficas de evolución del fitness y las guarda"""
    plt.figure(figsize=(16, 12))

    # Gráfica 1: Evolución completa del fitness
    plt.subplot(2, 3, 1)
    plt.plot(estadisticas['generacion'], estadisticas['mejor_fitness'], 
             'g-', linewidth=2, label='Mejor Fitness', alpha=0.8)
    plt.plot(estadisticas['generacion'], estadisticas['fitness_promedio'], 
             'b-', linewidth=2, label='Fitness Promedio', alpha=0.8)
    plt.plot(estadisticas['generacion'], estadisticas['peor_fitness'], 
             'r-', linewidth=2, label='Peor Fitness', alpha=0.8)
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title(f'Evolución del Fitness - Elitismo {NUM_ELITES} individuos\n({NUM_GENERACIONES} generaciones)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfica 2: Solo mejor fitness con marcadores
    plt.subplot(2, 3, 2)
    plt.plot(estadisticas['generacion'], estadisticas['mejor_fitness'], 
             'g-', linewidth=2, marker='o', markersize=2, alpha=0.8)
    plt.xlabel('Generación')
    plt.ylabel('Mejor Fitness')
    plt.title('Evolución del Mejor Fitness')
    plt.grid(True, alpha=0.3)
    
    # Gráfica 3: Diferencia entre mejor y peor (diversidad)
    diferencias = [mejor - peor for mejor, peor in 
                   zip(estadisticas['mejor_fitness'], estadisticas['peor_fitness'])]
    plt.subplot(2, 3, 3)
    plt.plot(estadisticas['generacion'], diferencias, 
             'm-', linewidth=2, alpha=0.8)
    plt.xlabel('Generación')
    plt.ylabel('Diferencia (Mejor - Peor)')
    plt.title('Diversidad de la Población')
    plt.grid(True, alpha=0.3)

    # Gráfica 4: Fitness promedio vs mejor
    plt.subplot(2, 3, 4)
    plt.plot(estadisticas['generacion'], estadisticas['mejor_fitness'], 
             'g-', linewidth=2, label='Mejor', alpha=0.8)
    plt.plot(estadisticas['generacion'], estadisticas['fitness_promedio'], 
             'b-', linewidth=2, label='Promedio', alpha=0.8)
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title('Mejor vs Promedio')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Gráfica 5: Histograma de fitness en generaciones clave
    plt.subplot(2, 3, 5)
    generaciones_clave = [1, 25, 50, 75, 100]
    fitness_clave = [estadisticas['mejor_fitness'][gen-1] for gen in generaciones_clave]
    colores = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    
    bars = plt.bar(range(len(generaciones_clave)), fitness_clave, color=colores)
    plt.xlabel('Momento de la Evolución')
    plt.ylabel('Mejor Fitness')
    plt.title('Mejor Fitness en Momentos Clave')
    plt.xticks(range(len(generaciones_clave)), 
               [f'Gen {gen}' for gen in generaciones_clave])
    plt.grid(True, alpha=0.3)
    
    # Añadir valores encima de las barras
    for bar, fitness in zip(bars, fitness_clave):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{fitness:.4f}', ha='center', va='bottom', fontsize=8)

    # Gráfica 6: Convergencia (últimas 20 generaciones)
    plt.subplot(2, 3, 6)
    ultimas_20_gen = estadisticas['generacion'][-20:]
    ultimas_20_mejor = estadisticas['mejor_fitness'][-20:]
    ultimas_20_prom = estadisticas['fitness_promedio'][-20:]
    
    plt.plot(ultimas_20_gen, ultimas_20_mejor, 
             'g-', linewidth=3, marker='o', markersize=4, label='Mejor')
    plt.plot(ultimas_20_gen, ultimas_20_prom, 
             'b-', linewidth=3, marker='s', markersize=3, label='Promedio')
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title('Convergencia (Últimas 20 generaciones)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guarda la gráfica
    grafica_filename = f"{directorio}/graficas_elitismo_{NUM_ELITES}_individuos_{NUM_GENERACIONES}_generaciones.png"
    plt.savefig(grafica_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Gráficas guardadas en: {grafica_filename}")

def mostrar_mejor_cromosoma(mejor_fitness, mejor_cromosoma, mejor_valor, mejor_generacion):
    """Muestra información del mejor cromosoma encontrado"""
    print(f"\n{'='*80}")
    print("CROMOSOMA ÓPTIMO ENCONTRADO - PARTE C")
    print("="*80)
    
    print(f"Generación de aparición: {mejor_generacion}")
    print(f"Cromosoma binario: {''.join(map(str, mejor_cromosoma))}")
    print(f"Valor decimal: {mejor_valor:,}")
    print(f"Valor normalizado: {mejor_valor / COEF:.10f}")
    print(f"Fitness: {mejor_fitness:.10f}")
    print(f"Función objetivo f(x) = (x/{COEF:,})² = {mejor_fitness:.10f}")
    
    # Información adicional sobre el elitismo
    print(f"\nInformación del elitismo:")
    print(f"- Número de élites conservados por generación: {NUM_ELITES}")
    print(f"- Individuos generados por selección/crossover/mutación: {NUM_CROMOSOMAS - NUM_ELITES}")
    print(f"- Población total: {NUM_CROMOSOMAS}")

def main():
    """Función principal que ejecuta el algoritmo genético con elitismo múltiple - Parte C"""
    print(f"🧬 INICIANDO ALGORITMO GENÉTICO - PARTE C 🧬")
    print(f"Configuración: Elitismo de {NUM_ELITES} individuos, {NUM_GENERACIONES} generaciones")
    
    # Crear directorio para resultados
    directorio = crear_directorio_resultados()
    print(f"📁 Resultados se guardarán en: {directorio}")

    # Ejecutar algoritmo
    estadisticas, mejor_fitness, mejor_cromosoma, mejor_valor, mejor_generacion = ejecutar_algoritmo_genetico()
    
    # Mostrar cromosoma óptimo
    mostrar_mejor_cromosoma(mejor_fitness, mejor_cromosoma, mejor_valor, mejor_generacion)
    
    # Mostrar tabla de estadísticas
    mostrar_tabla_estadisticas(estadisticas)
    
    # Preparar información del mejor cromosoma para exportación
    mejor_cromosoma_info = {
        'mejor_fitness': mejor_fitness,
        'mejor_cromosoma': mejor_cromosoma,
        'mejor_valor': mejor_valor,
        'mejor_generacion': mejor_generacion
    }
    
    # Exportar estadísticas a CSV
    exportar_estadisticas_csv(estadisticas, directorio, mejor_cromosoma_info)
    
    # Exportar archivo Excel completo
    exportar_estadisticas_excel(estadisticas, mejor_cromosoma_info, directorio)
    
    # Generar y guardar gráficas
    print(f"\nGenerando gráficas de evolución...")
    generar_graficas(estadisticas, directorio)
    
    print(f"\n🎉 PROCESO COMPLETADO - PARTE C")
    print(f"📊 Todos los archivos de resultados están en: {directorio}")
    print(f"   - Archivos CSV con estadísticas detalladas")
    print(f"   - Archivo Excel completo con múltiples hojas")
    print(f"   - Gráficas de evolución en formato PNG")
    print(f"\n⚙️  Para modificar parámetros, editar las constantes al inicio del archivo:")
    print(f"   - NUM_ELITES: {NUM_ELITES} (número de élites a conservar)")
    print(f"   - NUM_GENERACIONES: {NUM_GENERACIONES}")
    print(f"   - PROBABILIDAD_CROSSOVER: {PROBABILIDAD_CROSSOVER}")
    print(f"   - PROBABILIDAD_MUTACION: {PROBABILIDAD_MUTACION}")
    print(f"   - NUM_CROMOSOMAS: {NUM_CROMOSOMAS}")

if __name__ == "__main__":
    main()