import random
import matplotlib.pyplot as plt

# Parámetros del algoritmo
COEF = 2**30 - 1
NUM_CROMOSOMAS = 10
LONGITUD = 30
PROBABILIDAD_CROSSOVER = 0.75
PROBABILIDAD_MUTACION = 0.05
NUM_GENERACIONES = 100
NUM_ELITE = 2

def generar_poblacion(n, longitud):
    return [[random.randint(0, 1) for _ in range(longitud)] for _ in range(n)]

def binario_a_entero(cromosoma):
    return sum(bit * (2 ** (LONGITUD - 1 - i)) for i, bit in enumerate(cromosoma))

def evaluar_funcion_objetivo(x):
    return (x / COEF) ** 2

def calcular_fitness(valores_f_obj):
    total = sum(valores_f_obj)
    return [valor / total for valor in valores_f_obj]

def seleccion_ruleta(fitness):
    prob_acumuladas = []
    suma_acum = 0
    for f in fitness:
        suma_acum += f
        prob_acumuladas.append(suma_acum)
    r = random.random()
    for idx, acumulada in enumerate(prob_acumuladas):
        if r <= acumulada:
            return idx

def crossover_1_punto(padre1, padre2):
    if random.random() < PROBABILIDAD_CROSSOVER:
        punto = random.randint(1, LONGITUD - 1)
        hijo1 = padre1[:punto] + padre2[punto:]
        hijo2 = padre2[:punto] + padre1[punto:]
        return hijo1, hijo2
    else:
        return padre1[:], padre2[:]

def mutar(cromosoma):
    return [(1 - bit if random.random() < PROBABILIDAD_MUTACION else bit) for bit in cromosoma]

# Inicialización
poblacion = generar_poblacion(NUM_CROMOSOMAS, LONGITUD)

# Estadísticas
maximos, minimos, promedios = [], [], []

for generacion in range(NUM_GENERACIONES):
    enteros = [binario_a_entero(ind) for ind in poblacion]
    f_obj = [evaluar_funcion_objetivo(x) for x in enteros]
    fitness = calcular_fitness(f_obj)

    maximos.append(max(f_obj))
    minimos.append(min(f_obj))
    promedios.append(sum(f_obj) / len(f_obj))

    # === Elitismo ===
    elite_indices = sorted(range(len(f_obj)), key=lambda i: f_obj[i], reverse=True)[:NUM_ELITE]
    elite = [poblacion[i][:] for i in elite_indices]

    # Nueva población
    nueva_poblacion = elite[:]
    while len(nueva_poblacion) < NUM_CROMOSOMAS:
        i1 = seleccion_ruleta(fitness)
        i2 = seleccion_ruleta(fitness)
        padre1 = poblacion[i1]
        padre2 = poblacion[i2]
        hijo1, hijo2 = crossover_1_punto(padre1, padre2)
        hijo1 = mutar(hijo1)
        hijo2 = mutar(hijo2)
        nueva_poblacion.append(hijo1)
        if len(nueva_poblacion) < NUM_CROMOSOMAS:
            nueva_poblacion.append(hijo2)

    poblacion = nueva_poblacion

# Resultados finales
print("\nResultados Finales:")
enteros_finales = [binario_a_entero(ind) for ind in poblacion]
f_obj_finales = [evaluar_funcion_objetivo(x) for x in enteros_finales]
mejor = max(f_obj_finales)
peor = min(f_obj_finales)
prom = sum(f_obj_finales) / len(f_obj_finales)
idx_mejor = f_obj_finales.index(mejor)

print("Mejor cromosoma:", ''.join(map(str, poblacion[idx_mejor])), "→", enteros_finales[idx_mejor])
print("Máximo f(x):", mejor)
print("Mínimo f(x):", peor)
print("Promedio f(x):", prom)

# Gráfica
plt.plot(maximos, label="Máximos")
plt.plot(minimos, label="Mínimos")
plt.plot(promedios, label="Promedios")
plt.xlabel("Generación")
plt.ylabel("f(x)")
plt.title("Evolución de f(x) con Ruleta + Elitismo")
plt.legend()
plt.grid()
plt.show()
