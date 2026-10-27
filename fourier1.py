import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. Definición de la función periódica f(t)
# Ejemplo: Onda cuadrada de periodo T
T = 2.0  # Periodo de la función
omega_0 = 2 * np.pi / T  # Frecuencia fundamental

def f(t):
    """Función periódica original (definida en el intervalo [0, T])."""
    t_mod = np.mod(t, T)  # Hace que la función sea periódica fuera de [0, T]
    return 1.0 if t_mod < T / 2 else -1.0

# Vectorizar f(t) para poder evaluarla en arrays de NumPy
f_vec = np.vectorize(f)

# 2. Cálculo de Coeficientes de Fourier mediante integración
def calcular_coeficientes(N):
    # a0 es el promedio de la función
    a0, _ = quad(f, 0, T)
    a0 = a0 / T

    an = np.zeros(N)
    bn = np.zeros(N)

    for n in range(1, N + 1):
        # Integrales para an y bn
        an[n-1], _ = quad(lambda t: f(t) * np.cos(n * omega_0 * t), 0, T)
        bn[n-1], _ = quad(lambda t: f(t) * np.sin(n * omega_0 * t), 0, T)
        
        an[n-1] *= (2 / T)
        bn[n-1] *= (2 / T)

    return a0, an, bn

# 3. Reconstrucción de la Serie de Fourier
def serie_fourier(t, a0, an, bn):
    f_approx = np.full_like(t, a0)
    for n in range(1, len(an) + 1):
        f_approx += an[n-1] * np.cos(n * omega_0 * t) + bn[n-1] * np.sin(n * omega_0 * t)
    return f_approx

# --- Gráfica y Evaluación ---
t_vals = np.linspace(-T, 2*T, 1000)  # Graficar varios periodos
f_real = f_vec(t_vals)

plt.figure(figsize=(10, 5))
plt.plot(t_vals, f_real, 'k--', label="Función Original $f(t)$", linewidth=2)

# Probar con distinto número de armónicos N
for N in [1, 3, 7, 21]:
    a0, an, bn = calcular_coeficientes(N)
    f_approx = serie_fourier(t_vals, a0, an, bn)
    plt.plot(t_vals, f_approx, label=f"Serie de Fourier ($N={N}$)")

plt.title("Aproximación por Series de Fourier")
plt.xlabel("Tiempo (t)")
plt.ylabel("f(t)")
plt.grid(True)
plt.legend()
plt.show()