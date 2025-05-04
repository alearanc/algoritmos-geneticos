import random

poblacion = []
poblacion_decimal = []
COEF = 2**30 - 1

for _ in range(10):
    cromosoma = []
    for _ in range(30):
        gen = 1 if random.random() > 0.5 else 0
        cromosoma.append(gen)
    poblacion.append(cromosoma)


for cromosoma in poblacion:
    sum = 0
    for indice, gen in enumerate(cromosoma):
        if gen == 1:
            sum += 2**(29-indice)
    poblacion_decimal.append(sum)

f_obj = []

for cromosoma in poblacion_decimal:
    y = (cromosoma / COEF)**2
    f_obj.append(y)

print("X | Y")
for cromosoma_decimal, y in zip(poblacion_decimal, f_obj):
    print(f"{cromosoma_decimal} | {y}")

suma = 0
for y in f_obj:
    suma += y
print("Suma de Y:", suma)
print("Promedio de Y:", suma / len(f_obj))
print("Máximo:", max(f_obj))

fitness = []
total = 0
for y in f_obj:
    fitness.append(y/suma)
print("Fitness:")
for y in fitness:
    print(f"{y}")
print("Suma de Fitness:")
for y in fitness:
    total += y
print(total)