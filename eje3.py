import numpy as np
import matplotlib.pyplot as plt

# Nueva función f(x) = e^(-x) - x
def f(x):
    return np.exp(-x) - x

# Interpolación de Lagrange
def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Método de Bisección
def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:
        raise ValueError("El intervalo no contiene una raíz")
    
    iteraciones = []  # Guardamos las iteraciones
    print("Iteraciones del método de bisección:")
    for i in range(max_iter):
        c = (a + b) / 2
        iteraciones.append(c)  # Guardamos el valor de c en cada iteración
        print(f"Iteración {i+1}: c = {c:.6f}, f(c) = {func(c):.6f}")
        if abs(func(c)) < tol or (b - a) / 2 < tol:
            return c, iteraciones
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2, iteraciones  # Retorna la mejor estimación de la raíz

# Selección de cuatro puntos de interpolación equidistantes
t = np.linspace(0, 1, 4)
y_points = f(t)

# Construcción del polinomio interpolante mediante interpolación de Lagrange
x_vals = np.linspace(0, 1, 100)
y_interp = [lagrange_interpolation(x, t, y_points) for x in x_vals]

# Encontrar raíz del polinomio interpolante usando bisección
root, iteraciones = bisect(lambda x: lagrange_interpolation(x, t, y_points), 0, 1)

# Gráfica
plt.figure(figsize=(8, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = e^(-x) - x", linestyle='dashed', color='blue')
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(root, color='green', linestyle='dotted', label=f"Raíz aproximada: {root:.4f}")
plt.scatter(t, y_points, color='black', label="Puntos de interpolación")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Interpolación y búsqueda de raíces")
plt.legend()
plt.grid(True)
plt.show()

# Gráfico de la convergencia de la bisección
plt.figure(figsize=(8, 6))
plt.plot(range(len(iteraciones)), iteraciones, marker='o', color='purple')
plt.xlabel("Iteración")
plt.ylabel("Valor de la raíz")
plt.title("Convergencia del método de bisección")
plt.grid(True)
plt.show()

# Cálculo de los errores en cada iteración
errores_absolutos = [abs(f(i)) for i in iteraciones]
errores_relativos = [abs(f(i) / i) if i != 0 else 0 for i in iteraciones]
errores_cuadraticos = [err**2 for err in errores_absolutos]

# Gráfico de los errores
plt.figure(figsize=(8, 6))
plt.plot(range(len(errores_absolutos)), errores_absolutos, marker='o', color='orange', label="Error absoluto")
plt.plot(range(len(errores_relativos)), errores_relativos, marker='s', color='blue', label="Error relativo")
plt.plot(range(len(errores_cuadraticos)), errores_cuadraticos, marker='x', color='red', label="Error cuadrático")
plt.xlabel("Iteración")
plt.ylabel("Error")
plt.title("Errores en cada iteración de la bisección")
plt.legend()
plt.grid(True)
plt.savefig("errores_biseccion.png")
plt.show()

# Imprimir la raíz encontrada
print(f"La raíz aproximada usando interpolación es: {root:.6f}")
