import numpy as np
import matplotlib.pyplot as plt

# Nueva función f(x) = x^3 - 6x^2 + 11x - 6
def f(x):
    return x**3 - 6*x**2 + 11*x - 6

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
    
    iteraciones = []
    errores_abs = []
    errores_rel = []
    errores_cuad = []
    print("Iteraciones del método de bisección:")
    for i in range(max_iter):
        c = (a + b) / 2
        iteraciones.append(c)
        error_abs = abs(func(c))
        error_rel = abs(func(c) / c) if c != 0 else 0
        error_cuad = error_abs ** 2
        errores_abs.append(error_abs)
        errores_rel.append(error_rel)
        errores_cuad.append(error_cuad)
        print(f"Iteración {i+1}: c = {c:.6f}, f(c) = {func(c):.6f}, Error absoluto = {error_abs:.6f}, Error relativo = {error_rel:.6f}, Error cuadrático = {error_cuad:.6f}")
        if error_abs < tol or (b - a) / 2 < tol:
            return c, iteraciones, errores_abs, errores_rel, errores_cuad
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2, iteraciones, errores_abs, errores_rel, errores_cuad

# Selección de tres puntos de interpolación
a, b = 1.0, 3.0
x0, x1, x2 = 1.0, 2.0, 3.0
x_points = np.array([x0, x1, x2])
y_points = f(x_points)

# Construcción del polinomio interpolante mediante interpolación de Lagrange
x_vals = np.linspace(a, b, 100)
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]

# Encontrar raíz del polinomio interpolante usando bisección
root, iteraciones, errores_abs, errores_rel, errores_cuad = bisect(lambda x: lagrange_interpolation(x, x_points, y_points), a, b)

# Gráfica de la función y el polinomio interpolante
plt.figure(figsize=(8, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = x^3 - 6x^2 + 11x - 6", linestyle='dashed', color='blue')
plt.plot(x_vals, y_interp, label="Interpolación de Lagrange", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(root, color='green', linestyle='dotted', label=f"Raíz aproximada: {root:.4f}")
plt.scatter(x_points, y_points, color='black', label="Puntos de interpolación")
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

# Gráfico de los errores
plt.figure(figsize=(8, 6))
plt.plot(range(len(errores_abs)), errores_abs, marker='o', color='orange', label='Error absoluto')
plt.plot(range(len(errores_rel)), errores_rel, marker='s', color='red', label='Error relativo')
plt.plot(range(len(errores_cuad)), errores_cuad, marker='^', color='blue', label='Error cuadrático')
plt.xlabel("Iteración")
plt.ylabel("Error")
plt.title("Errores en cada iteración de la bisección")
plt.legend()
plt.grid(True)
plt.show()

# Imprimir la raíz encontrada
print(f"La raíz aproximada usando interpolación es: {root:.6f}")
