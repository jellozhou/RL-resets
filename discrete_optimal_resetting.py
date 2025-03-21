import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import log1p


# plot optimal r* for different n
def asech(x):
    """Inverse hyperbolic secant function."""
    return np.log(1/x + np.sqrt(1/x**2 - 1))

def equation(r, n):
    """Equation to solve for r^*."""
    if r >= 1 or r <= 0:
        return np.inf  # Avoid invalid values
    
    exp_term = np.exp(asech(1 - r)) ** abs(n)
    sqrt_term = np.sqrt((2 - r) / (1 - r)) * np.sqrt(r / (1 - r)) * abs(n)
    return 1 - exp_term + (exp_term * sqrt_term) / (2 - r)

n_values = np.arange(10, 1000, 10)  # Values of n from 10 to 250 in steps of 10
r_star_values = []

for n in n_values:
    r_guess = 0.01  # Initial guess for r^*
    r_star = fsolve(equation, r_guess, args=(n))
    if 0 < r_star[0] < 1:  # Ensure valid solution
        r_star_values.append(r_star[0])
    else:
        r_star_values.append(np.nan)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Linear plot
axs[0].plot(n_values, r_star_values, '-bo', label='$r^*$ values')
axs[0].set_xlabel('$n$', fontsize=14)
axs[0].set_ylabel('$r^*$', fontsize=14)
# axs[0].set_title('Linear Scale: Solutions for $r^*$ as a function of $n$', fontsize=16)
# axs[0].legend()
# axs[0].grid()

# Log-log plot
axs[1].plot(n_values, r_star_values, '-bo', label='$r^*$ values')
axs[1].set_xlabel('$n$', fontsize=14)
axs[1].set_ylabel('$r^*$', fontsize=14)
axs[1].set_xscale('log')
axs[1].set_yscale('log')

# Fit a best-fit linear line in log-log space
log_n_values = np.log(n_values)
log_r_star_values = np.log(r_star_values)
valid_indices = ~np.isnan(log_r_star_values)  # Exclude NaN values
coeffs = np.polyfit(log_n_values[valid_indices], log_r_star_values[valid_indices], 1)
best_fit_line = np.polyval(coeffs, log_n_values)

# Plot the best-fit line
axs[1].plot(n_values, np.exp(best_fit_line), 'r--', label=f'Best-fit line (slope={coeffs[0]:.2f})')
axs[1].legend()

plt.tight_layout()
plt.savefig("optimal_r_1D_discrete.png")
plt.show()

# plot MFPT for different n = 50
def function(r, n=50):
    """Computes the given function for a specific r and n=50."""
    if r >= 1 or r <= 0:
        return np.nan  # Avoid invalid values
    
    acosh_term = np.arccosh(1 / (1 - r))
    exp_term = np.exp(-abs(n) * acosh_term)
    return (1 - exp_term) / (r * exp_term)

r_values_50 = np.linspace(1e-5, 0.008, 1000)  # Avoid r=0 and r=1
r_values_100 = np.linspace(1e-5, 0.001, 100)
# f_values = [function(r) for r in r_values]

# plt.figure(figsize=(8, 6))
# plt.plot(r_values, f_values, 'r-', label='$f(r)$ for $n=50$')
# plt.xlabel('$r$', fontsize=14)
# plt.ylabel('MFPT', fontsize=14)
# plt.title('MFPT for $n=50$', fontsize=16)
# plt.legend()
# plt.grid()
# plt.show()

# r_values = np.linspace(0.01, 0.99, 100)  # Avoid r=0 and r=1
f_values_n50 = [function(r, 50) for r in r_values_50]
f_values_n100 = [function(r, 100) for r in r_values_100]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(r_values_50, f_values_n50, 'r-', label='$f(r)$ for $n=50$')
axs[0].set_xlabel('$r$', fontsize=14)
axs[0].set_ylabel('MFPT', fontsize=14)
axs[0].set_title('MFPT for $n=50$', fontsize=16)
# axs[0].legend()
# axs[0].grid()

axs[1].plot(r_values_100, f_values_n100, 'b-', label='$f(r)$ for $n=100$')
axs[1].set_xlabel('$r$', fontsize=14)
axs[1].set_ylabel('MFPT', fontsize=14)
axs[1].set_title('MFPT for $n=100$', fontsize=16)
# axs[1].legend()
# axs[1].grid()

plt.tight_layout()
plt.savefig("mfpt_comparison.png")
plt.show()
