import random
import matplotlib.pyplot as plt

# Parámetros
COEF = 2**30 - 1
NUM_CROMOSOMAS = 10
LONGITUD = 30
PROB_CROSSOVER = 0.75
PROB_MUTACION = 0.05
GENERACIONES = 100  
NUM_ELITE = 2

def evaluar_funcion_objetivo(valor):
    return (valor / COEF) ** 2

def binario_a_entero(cromosoma):
    return sum(gen * (2 ** (LONGITUD - 1 - i)) for i, gen in enumerate(cromosoma))

def generar_poblacion(n, longitud):
    return [[random.randint(0, 1) for _ in range(longitud)] for _ in range(n)]

def torneo_seleccion(f_obj, k=2):
    candidatos = random.sample(range(len(f_obj)), k)
    mejor = max(candidatos, key=lambda idx: f_obj[idx])
    return mejor

def crossover(padre1, padre2):
    if random.random() < PROB_CROSSOVER:
        punto = random.randint(1, LONGITUD - 1)
        hijo1 = padre1[:punto] + padre2[punto:]
        hijo2 = padre2[:punto] + padre1[punto:]
        return hijo1, hijo2
    return padre1[:], padre2[:]

def mutar(cromosoma):
    return [1 - bit if random.random() < PROB_MUTACION else bit for bit in cromosoma]

# Inicializar
poblacion = generar_poblacion(NUM_CROMOSOMAS, LONGITUD)

maximos, minimos, promedios = [], [], []

for generacion in range(GENERACIONES):
    enteros = [binario_a_entero(ind) for ind in poblacion]
    f_obj = [evaluar_funcion_objetivo(x) for x in enteros]

    maximos.append(max(f_obj))
    minimos.append(min(f_obj))
    promedios.append(sum(f_obj) / len(f_obj))

    print(f"\nGeneración {generacion+1}")
    print(f"Máximo: {max(f_obj):.6f}, Mínimo: {min(f_obj):.6f}, Promedio: {sum(f_obj)/len(f_obj):.6f}")

    # === Elitismo ===
    elite_indices = sorted(range(len(f_obj)), key=lambda i: f_obj[i], reverse=True)[:NUM_ELITE]
    elite = [poblacion[i][:] for i in elite_indices]

    nueva_poblacion = elite[:]
    while len(nueva_poblacion) < NUM_CROMOSOMAS:
        i1 = torneo_seleccion(f_obj)
        i2 = torneo_seleccion(f_obj)
        padre1, padre2 = poblacion[i1], poblacion[i2]

        hijo1, hijo2 = crossover(padre1, padre2)
        hijo1 = mutar(hijo1)
        hijo2 = mutar(hijo2)

        nueva_poblacion.append(hijo1)
        if len(nueva_poblacion) < NUM_CROMOSOMAS:
            nueva_poblacion.append(hijo2)

    poblacion = nueva_poblacion

# Mostrar mejor solución final
enteros_final = [binario_a_entero(ind) for ind in poblacion]
f_obj_final = [evaluar_funcion_objetivo(x) for x in enteros_final]
mejor_idx = f_obj_final.index(max(f_obj_final))

print("\nMejor solución final:")
print("Cromosoma:", ''.join(map(str, poblacion[mejor_idx])))
print("Entero:", enteros_final[mejor_idx])
print("f(x):", f_obj_final[mejor_idx])

# Gráfica
plt.plot(maximos, label='Máximo')
plt.plot(minimos, label='Mínimo')
plt.plot(promedios, label='Promedio')
plt.xlabel("Generación")
plt.ylabel("f(x)")
plt.title("Evolución de f(x) con Torneo + Elitismo")
plt.legend()
plt.grid(True)
plt.show()
