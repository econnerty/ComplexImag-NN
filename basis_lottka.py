import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the Lotka-Volterra model
def lotka_volterra(t, z, alpha, beta, gamma, delta):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Simulate the Lotka-Volterra model
alpha, beta, gamma, delta = 0.1, 0.02, 0.3, 0.01
initial_conditions = [10, 5]
t_span = [0, 50]
t_eval = np.linspace(t_span[0], t_span[1], 400)
solution = solve_ivp(lotka_volterra, t_span, initial_conditions, args=(alpha, beta, gamma, delta), t_eval=t_eval)

# Prepare the data vector b (complex numbers)
b = solution.y[0] + 1j * solution.y[1]  # Prey as real part, Predator as imaginary part

# Basis matrix A construction
c_real = np.arange(-.1,.1, .1)
#c_real = [0]
d_imag = np.arange(-2.0,2, 0.2)
A = np.ones((len(t_eval), 1 + len(c_real) * len(d_imag)), dtype=np.complex_)  # Start with a column of ones for the constant term
for i, c in enumerate(c_real):
    for j, d in enumerate(d_imag):
        if c == 0 and d == 0:
            continue
        A[:, 1 + i * len(d_imag) + j] = np.exp((c + 1j * d) * t_eval)

# Solve for coefficients
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
print(f"Residuals: {residuals}")

# Visualization of the fit
fitted = A.dot(x)

# Plot in the real-imaginary plane
plt.figure(figsize=(10, 8))
plt.plot(solution.y[0], solution.y[1], 'r', label='Actual Trajectory')
plt.plot(fitted.real, fitted.imag, 'b--', label='Fitted Trajectory')
plt.title('Lotka-Volterra Dynamics in the Complex Plane')
plt.xlabel('Prey Population (Real Part)')
plt.ylabel('Predator Population (Imaginary Part)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
