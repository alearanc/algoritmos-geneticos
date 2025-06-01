import random

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

    mejor_idx = f_obj.index[mejor_fitness]
    mejor_cromosoma = poblacion[mejor_idx]
    mejor_valor = enteros[mejor_idx]

    return {
        'mejor fitness': mejor_fitness,
        'peor fitness': peor_fitness,
        'fitness promedio': fitness_promedio,
        'mejor cromosoma': mejor_cromosoma,
        'mejor valor': mejor_valor
    }


def mostrar_poblacion(poblacion, titulo = "Población"):
    print(f"\n{titulo}:")
    for i, cromosoma in enumerate(poblacion):
        entero = binario_a_entero(cromosoma)
        f_obj = evaluar_funcion_objetivo(entero)
        print(f" Cromosoma {i+1}: {cromosoma} -> {entero} -> f(x) = {f_obj:.6f}")

# Ejecución principal
print(f"="*60)
print("ALGORITMO GENÉTICO - UNA ITERACIÓN COMPLETA")
print("="*60)

print(f"Parámetros:")
print(f" - Población: {NUM_CROMOSOMAS} cromosomas")
print(f" - Longitud cromosoma: {LONGITUD} bits")
print(f" - Prob. crossover: {PROBABILIDAD_CROSSOVER}")
print(f" - Prob. mutación: {PROBABILIDAD_MUTACION}")
print(f" - Coeficiente: {COEF}")

# Generar población inicial
print(f"\n1. GENERACIÓN DE LA POBLACIÓN INICIAL")
poblacion = generar_poblacion(NUM_CROMOSOMAS, LONGITUD)
mostrar_poblacion(poblacion, "Población inicial")

# Evaluar fitness
print(f"\n2. EVALUACIÓN DE FITNESS")
enteros = [binario_a_entero(x) for x in poblacion]
f_obj = [evaluar_funcion_objetivo(x) for x in enteros]
fitness_valores = calcular_fitness(f_obj)

print("Valores de fitness:")
for i, (entero, obj, fitness) in enumerate(zip(enteros, f_obj, fitness_valores)):
    print(f" Cromosoma {i+1}: valor={entero}, f(x)={obj:.6f}, fitness={fitness:.4f}")

# Calcular probabilidades acumuladas
prob_acumuladas = calcular_prob_acumuladas(fitness_valores)
print(f"\nProbabilidades acumuladas: {[round(p, 4) for p in prob_acumuladas]}")

# Selección de padres
print("\n3. SELECCIÓN DE PADRES")
pares_de_padres = []
for i in range(NUM_CROMOSOMAS // 2): # 5 pares para 10 cromosomas
    padre1_idx = ruleta_seleccion(prob_acumuladas)
    padre2_idx = ruleta_seleccion(prob_acumuladas)
    pares_de_padres.append([padre1_idx, padre2_idx])
    print(f" Par {i+1}: Padre1 = Cromosoma {padre1_idx+1}, Padre2 = Cromosoma {padre2_idx}")

# Crossover y Mutación
print(f"\n4. CROSSOVER Y MUTACIÓN")
nueva_poblacion = []
for i, (padre1_idx, padre2_idx) in enumerate(pares_de_padres):
    padre1 = poblacion[padre1_idx]
    padre2 = poblacion[padre2_idx]

    print(f"\n  Par {i+1}:")
    print(f"    Padre1: {padre1}")
    print(f"    Padre2: {padre2}")

    # Crossover
    hijo1, hijo2, hubo_crossover, punto_corte = crossover(padre1, padre2, PROBABILIDAD_CROSSOVER)
    if hubo_crossover:
        print(f"    CROSSOVER en punto {punto_corte}")
    else:
        print(f"    NO hubo crossover")
    print(f"    Hijo1 (post-crossover): {hijo1}")
    print(f"    Hijo2 (post-crossover): {hijo2}")

    # Mutación
    hijo1_final, hijo2_final, hubo_mutacion = mutacion(hijo1, hijo2, PROBABILIDAD_MUTACION)
    if hubo_mutacion:
        print(f"    MUTACIÓN aplicada")
    else:
        print(f"    NO hubo mutación")

    nueva_poblacion.extend([hijo1_final, hijo2_final])

# 6. Mostrar nueva población
print("\n5. NUEVA POBLACIÓN GENERADA")
mostrar_poblacion(nueva_poblacion, "Nueva población")

# 7. Comparar fitness
print("\n6. COMPARACIÓN DE FITNESS")
nuevos_enteros = [binario_a_entero(x) for x in nueva_poblacion]
nuevos_f_obj = [evaluar_funcion_objetivo(x) for x in nuevos_enteros]

print("Población original:")
mejor_original = max(f_obj)
idx_mejor_original = f_obj.index(mejor_original)
print(f"  Mejor fitness: {mejor_original:.6f} (Cromosoma {idx_mejor_original+1})")

print("Nueva población:")
mejor_nuevo = max(nuevos_f_obj)
idx_mejor_nuevo = nuevos_f_obj.index(mejor_nuevo)
print(f"  Mejor fitness: {mejor_nuevo:.6f} (Cromosoma {idx_mejor_nuevo+1})")

if mejor_nuevo > mejor_original:
    print("  ¡MEJORA! La nueva población tiene mejor fitness.")
elif mejor_nuevo == mejor_original:
    print("  IGUAL. El fitness se mantiene.")
else:
    print("  DETERIORO. El fitness empeoró (normal en AG).")

print("\n" + "="*60)
print("ITERACIÓN COMPLETA FINALIZADA")
print("="*60)