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

# Ejecución principal
poblacion = generar_poblacion(NUM_CROMOSOMAS, LONGITUD)
enteros = [binario_a_entero(x) for x in poblacion]
f_obj = [evaluar_funcion_objetivo(x) for x in enteros]
fitness_valores = calcular_fitness(f_obj)

# Estadísticas
print("Binario\t\t\t\tEntero\t\tf(x)\t\tFitness")
for bin_crom, entero, fx, fit in zip(poblacion, enteros, f_obj, fitness_valores):
    bin_str = ''.join(str(bit) for bit in bin_crom)
    print(f"{bin_str}\t{entero}\t{fx:.6f}\t{fit:.6f}")

print("\nSuma f(x):", sum(f_obj))
print("Promedio f(x):", sum(f_obj) / len(f_obj))
print("Máximo f(x):", max(f_obj))
print("Suma fitness:", sum(fitness_valores))

# Método de la ruleta (selección)
#################################
prob_acumuladas = []
suma_acumulada = 0

for prob in fitness_valores: 
    suma_acumulada += prob
    prob_acumuladas.append(suma_acumulada)

# Función para la selección del individuo
def ruleta_seleccion():
    r = random.random() #Nro. aleatorio entre 0 y 1
    for index, acumulada in enumerate(prob_acumuladas):
        if r <= acumulada:
            return index
        
# Selecciona 5 pares de padres
pares_de_padres = []
for _ in range(5):
    padre1 = ruleta_seleccion()
    padre2 = ruleta_seleccion()
    pares_de_padres.append([padre1, padre2])

# Mostrar los pares seleccionados
print("\nPadres seleccionados:")
for i, (p1, p2) in enumerate(pares_de_padres, start=1):
    print(f"Par {i} - {p1}: {enteros[p1]} y {p2}: {enteros[p2]}")