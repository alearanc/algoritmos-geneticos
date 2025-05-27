import random
import matplotlib.pyplot as plt

# Parámetros del algoritmo
COEF = 2**30 - 1
NUM_CROMOSOMAS = 10
LONGITUD = 30
PROB_CROSSOVER = 0.75
PROB_MUTACION = 0.05
GENERACIONES = 20

# Función objetivo
def evaluar_funcion_objetivo(valor):
    return (valor / COEF) ** 2

# Decodifica un cromosoma binario a entero
def binario_a_entero(cromosoma):
    return sum(gen * (2 ** (LONGITUD - 1 - i)) for i, gen in enumerate(cromosoma))

# Genera una población inicial de n cromosomas
def generar_poblacion(n, longitud):
    return [[random.randint(0, 1) for _ in range(longitud)] for _ in range(n)]

# Selección por torneo (k=2 por defecto)
def torneo_seleccion(f_obj, k=2):
    candidatos = random.sample(range(len(f_obj)), k)
    mejor = max(candidatos, key=lambda idx: f_obj[idx])
    return mejor

# Cruza dos padres con crossover de 1 punto
def crossover(padre1, padre2):
    if random.random() < PROB_CROSSOVER:
        punto = random.randint(1, LONGITUD - 1)
        hijo1 = padre1[:punto] + padre2[punto:]
        hijo2 = padre2[:punto] + padre1[punto:]
        return hijo1, hijo2
    return padre1[:], padre2[:]

# Aplica mutación por inversión de bits
def mutar(cromosoma):
    return [1 - bit if random.random() < PROB_MUTACION else bit for bit in cromosoma]

# Programa principal
poblacion = generar_poblacion(NUM_CROMOSOMAS, LONGITUD)

# Para las gráficas
maximos, minimos, promedios = [], [], []

for generacion in range(GENERACIONES):
    enteros = [binario_a_entero(ind) for ind in poblacion]
    f_obj = [evaluar_funcion_objetivo(x) for x in enteros]

    maximo = max(f_obj)
    minimo = min(f_obj)
    promedio = sum(f_obj) / len(f_obj)

    maximos.append(maximo)
    minimos.append(minimo)
    promedios.append(promedio)

    print(f"\nGeneración {generacion+1}")
    print(f"Máximo: {maximo:.6f}, Mínimo: {minimo:.6f}, Promedio: {promedio:.6f}")

    nueva_poblacion = []
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
mejor_cromosoma = poblacion[mejor_idx]
mejor_entero = enteros_final[mejor_idx]
mejor_fx = f_obj_final[mejor_idx]

print("\nMejor solución final:")
print("Cromosoma:", ''.join(map(str, mejor_cromosoma)))
print("Entero:", mejor_entero)
print("f(x):", mejor_fx)

# Gráfica
plt.plot(maximos, label='Máximo')
plt.plot(minimos, label='Mínimo')
plt.plot(promedios, label='Promedio')
plt.xlabel("Generación")
plt.ylabel("f(x)")
plt.title("Evolución de f(x) con Selección por Torneo")
plt.legend()
plt.grid(True)
plt.show()
