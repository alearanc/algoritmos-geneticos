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
    poblacion = []
    for _ in range(n):
        cromosoma = [random.randint(0, 1) for _ in range(longitud)]
        poblacion.append(cromosoma)
    return poblacion

def binario_a_entero(cromosoma):
    return sum(gen * (2 ** (LONGITUD - 1 - i)) for i, gen in enumerate(cromosoma))

def evaluar_funcion_objetivo(valor):
    return (valor / COEF) ** 2

def calcular_fitness(valores_f_obj):
    total = sum(valores_f_obj)
    fitness = [y / total for y in valores_f_obj]
    return fitness

def calcular_prob_acumuladas(fitness_valores):
    prob_acumuladas = []
    acumulada = 0
    for fitness in fitness_valores:
        acumulada += fitness
        prob_acumuladas.append(acumulada)
    return prob_acumuladas

def ruleta_seleccion(prob_acumuladas):
    r = random.random() #Nro. aleatorio entre 0 y 1
    for indice, acumulada in enumerate(prob_acumuladas):
        if r <= acumulada:
            return indice
    return len(prob_acumuladas) - 1   

def crossover(padre1, padre2, probabilidad):
    r = random.random()
    if r < probabilidad:
        punto_corte = random.randint(1, len(padre1) - 1)
        hijo1 = padre1[:punto_corte] + padre2[punto_corte:]
        hijo2 = padre2[:punto_corte] + padre1[punto_corte:]
        return hijo1, hijo2, True, punto_corte
    else:
        return padre1[:], padre2[:], False, None

def mutacion(hijo1, hijo2, probabilidad):
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

def obtener_estadisticas_poblacion(poblacion):
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
    enteros = [binario_a_entero(x) for x in poblacion]
    f_obj = [evaluar_funcion_objetivo(x) for x in enteros]
    fitness_valores = calcular_fitness(f_obj)
    prob_acumuladas = calcular_prob_acumuladas(fitness_valores)

    nueva_poblacion = []
    for _ in range(NUM_CROMOSOMAS // 2):
        padre1_idx = ruleta_seleccion(prob_acumuladas)
        padre2_idx = ruleta_seleccion(prob_acumuladas)

        padre1 = poblacion[padre1_idx]
        padre2 = poblacion[padre2_idx]

        # Crossover
        hijo1, hijo2, _, _ = crossover(padre1, padre2, PROBABILIDAD_CROSSOVER)

        # Mutación
        hijo1_final, hijo2_final, _ = mutacion(hijo1, hijo2, PROBABILIDAD_MUTACION)

        nueva_poblacion.extend([hijo1_final, hijo2_final])

    return nueva_poblacion

def ejecutar_algoritmo_genetico(num_generaciones):
    """Ejecuta el algoritmo genético por N generaciones"""
    print(f"="*80)
    print(f"AG CANÓNICO - {num_generaciones} GENERACIONES")
    print(f"="*80)

    print(f"Parámetros:")
    print(f" - Población: {NUM_CROMOSOMAS} cromosomas")
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
    directorio = f"resultados_AG_{timestamp}"
    
    if not os.path.exists(directorio):
        os.makedirs(directorio)
    
    return directorio

def exportar_estadisticas_csv(estadisticas, num_generaciones, directorio, mejor_cromosoma_info):
    """Exporta estadísticas a archivo CSV"""
    filename = f"{directorio}/estadisticas_{num_generaciones}_generaciones.csv"
    
    df = pd.DataFrame(estadisticas)
    df.to_csv(filename, index=False, float_format='%.10f')
    
    # Archivo adicional con información del mejor cromosoma
    mejor_info_filename = f"{directorio}/mejor_cromosoma_{num_generaciones}_generaciones.csv"
    mejor_df = pd.DataFrame([{
        'Generaciones_Ejecutadas': num_generaciones,
        'Mejor_Generacion': mejor_cromosoma_info['mejor_generacion'],
        'Mejor_Fitness': mejor_cromosoma_info['mejor_fitness'],
        'Mejor_Valor_Decimal': mejor_cromosoma_info['mejor_valor'],
        'Mejor_Valor_Normalizado': mejor_cromosoma_info['mejor_valor'] / COEF,
        'Mejor_Cromosoma': ''.join(map(str, mejor_cromosoma_info['mejor_cromosoma']))
    }])
    mejor_df.to_csv(mejor_info_filename, index=False, float_format='%.10f')
    
    print(f"✓ Estadísticas exportadas a: {filename}")
    print(f"✓ Mejor cromosoma exportado a: {mejor_info_filename}")

def exportar_estadisticas_excel(todas_estadisticas, directorio):
    """Exporta todas las estadísticas a un archivo Excel con múltiples hojas"""
    filename = f"{directorio}/estadisticas_completas.xlsx"
    
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
            'Valor': 'Ruleta'
        }, {
            'Parametro': 'Metodo_Crossover',
            'Valor': '1 Punto'
        }])
        parametros_df.to_excel(writer, sheet_name='Parametros', index=False)
    
    print(f"✓ Archivo Excel completo exportado a: {filename}")

def mostrar_tabla_estadisticas(estadisticas, num_generaciones):
    """Muestra tabla con estadísticas por generación"""
    print(f"\n{'='*80}")
    print("TABLA DE ESTADÍSTICAS POR GENERACIÓN")
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
    plt.title(f'Evolución del Fitness ({num_generaciones} generaciones)')
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
    grafica_filename = f"{directorio}/graficas_{num_generaciones}_generaciones.png"
    plt.savefig(grafica_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Gráficas guardadas en: {grafica_filename}")

def mostrar_mejor_cromosoma(mejor_fitness, mejor_cromosoma, mejor_valor, mejor_generacion):
    """Muestra información del mejor cromosoma encontrado"""
    print(f"\n{'='*80}")
    print("MEJOR CROMOSOMA ENCONTRADO")
    print("="*80)
    
    print(f"Generación donde se encontró: {mejor_generacion}")
    print(f"Cromosoma: {mejor_cromosoma}")
    print(f"Valor decimal: {mejor_valor}")
    print(f"Valor normalizado: {mejor_valor / COEF:.10f}")
    print(f"Fitness: {mejor_fitness:.10f}")
    print(f"Función objetivo f(x) = (x/{COEF})² = {mejor_fitness:.10f}")

def main():
    """Función principal que ejecuta el algoritmo para diferentes números de generaciones"""
    generaciones_a_probar = [20, 100, 200]
    todas_estadisticas = {}
    
    # Crear directorio para resultados
    directorio = crear_directorio_resultados()
    print(f"📁 Resultados se guardarán en: {directorio}")

    for num_gen in generaciones_a_probar:
        print(f"\n\n{'#'*100}")
        print(f"EJECUTANDO ALGORITMO GENÉTICO CON {num_gen} GENERACIONES")
        print(f"{'#'*100}")
        
        # Ejecutar algoritmo
        estadisticas, mejor_fitness, mejor_cromosoma, mejor_valor, mejor_generacion = ejecutar_algoritmo_genetico(num_gen)
        
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
        
        # Mostrar tabla de estadísticas
        mostrar_tabla_estadisticas(estadisticas, num_gen)
        
        # Mostrar mejor cromosoma
        mostrar_mejor_cromosoma(mejor_fitness, mejor_cromosoma, mejor_valor, mejor_generacion)
        
        # Exportar estadísticas a CSV
        mejor_cromosoma_info = {
            'mejor_fitness': mejor_fitness,
            'mejor_cromosoma': mejor_cromosoma,
            'mejor_valor': mejor_valor,
            'mejor_generacion': mejor_generacion
        }
        exportar_estadisticas_csv(estadisticas, num_gen, directorio, mejor_cromosoma_info)
        
        # Generar y guardar gráficas
        print(f"\nGenerando gráficas para {num_gen} generaciones...")
        generar_graficas(estadisticas, num_gen, directorio)
        
        # Pausa entre ejecuciones (opcional)
        if num_gen != generaciones_a_probar[-1]:
            input(f"\nPresiona Enter para continuar con la siguiente prueba...")
    
    # Exportar archivo Excel completo con todas las estadísticas
    print(f"\n{'='*80}")
    print("EXPORTANDO ARCHIVO EXCEL COMPLETO")
    print("="*80)
    exportar_estadisticas_excel(todas_estadisticas, directorio)
    
    print(f"\n🎉 PROCESO COMPLETADO")
    print(f"📊 Todos los archivos de resultados están en: {directorio}")
    print(f"   - Archivos CSV individuales por cada configuración")
    print(f"   - Archivo Excel completo con todas las estadísticas")
    print(f"   - Gráficas en formato PNG")

if __name__ == "__main__":
    # Para ejecutar todas las pruebas (20, 100, 200 generaciones):
    main()
    
    # Para una sola ejecución (descomentar):
    # directorio = crear_directorio_resultados()
    # estadisticas, mejor_fitness, mejor_cromosoma, mejor_valor, mejor_generacion = ejecutar_algoritmo_genetico(20)
    # mostrar_tabla_estadisticas(estadisticas, 20)
    # mostrar_mejor_cromosoma(mejor_fitness, mejor_cromosoma, mejor_valor, mejor_generacion)
    # mejor_cromosoma_info = {'mejor_fitness': mejor_fitness, 'mejor_cromosoma': mejor_cromosoma, 'mejor_valor': mejor_valor, 'mejor_generacion': mejor_generacion}
    # exportar_estadisticas_csv(estadisticas, 20, directorio, mejor_cromosoma_info)
    # generar_graficas(estadisticas, 20, directorio)