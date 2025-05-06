import random

COEF = 2**30 - 1
NUM_CROMOSOMAS = 10
LONGITUD = 30

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
fitness = calcular_fitness(f_obj)

# Estadísticas
print("Binario\t\t\t\tEntero\t\tf(x)\t\tFitness")
for bin_crom, entero, fx, fit in zip(poblacion, enteros, f_obj, fitness):
    bin_str = ''.join(str(bit) for bit in bin_crom)
    print(f"{bin_str}\t{entero}\t{fx:.6f}\t{fit:.6f}")

print("\nSuma f(x):", sum(f_obj))
print("Promedio f(x):", sum(f_obj) / len(f_obj))
print("Máximo f(x):", max(f_obj))
print("Suma fitness:", sum(fitness))