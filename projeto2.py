import numpy as np
from scipy.linalg import solve_triangular

#------------------Definição dos métodos------------------------

def gauss_ruggiero(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(A)

    # Etapa de Eliminação
    for k in range(n - 1):
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k] = 0
            for j in range(k + 1, n):
                A[i, j] = A[i, j] - m * A[k, j]
            b[i] = b[i] - m * b[k]

    # Resolução do sistema (substituição regressiva)
    x = np.zeros(n)
    x[n - 1] = b[n - 1] / A[n - 1, n - 1]

    for k in range(n - 2, -1, -1):
        s = 0
        for j in range(k + 1, n):
            s = s + A[k, j] * x[j]
        x[k] = (b[k] - s) / A[k, k]

    return x


import numpy as np

def cholesky_ruggiero(A):
    n = A.shape[0]
    G = np.zeros_like(A)

    for k in range(n):
        soma = sum(G[k, j] ** 2 for j in range(k))
        r = A[k, k] - soma
        if r <= 0:
            raise ValueError("A matriz não é definida positiva.")
        G[k, k] = np.sqrt(r)

        for i in range(k + 1, n):
            soma = sum(G[i, j] * G[k, j] for j in range(k))
            G[i, k] = (A[i, k] - soma) / G[k, k]

    return G


import numpy as np

def gauss_jacobi(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(A)
    x = np.zeros(n) if x0 is None else x0.copy()
    x_new = np.zeros(n)

    for k in range(max_iter):
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]

        # Critério de parada
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k + 1  # solução e número de iterações

        x = x_new.copy()

    raise Exception("O método de Gauss-Jacobi não convergiu no número máximo de iterações")


def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(A)
    x = np.zeros(n) if x0 is None else x0.copy()

    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))        # usa valores novos (k+1)
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))  # usa valores antigos (k)
            x[i] = (b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, k + 1  # solução e número de iterações

    raise Exception("O método de Gauss-Seidel não convergiu no número máximo de iterações")


# ---------- Sistema linear ----------


A = np.array([
    [6, -1, 2],
    [-1, 7, 1],
    [2, 1, 8]
], dtype=float)

b = np.array([19, -6, 27], dtype=float)

x0 = np.zeros(3)


# ---------- Resolvendo com os métodos ----------

print("=== Resolvendo o sistema Ax = b ===\n")

# Eliminação de Gauss
try:
    x_gauss = gauss_ruggiero(A.copy(), b.copy())
    print("Eliminação de Gauss:", x_gauss)
except Exception as e:
    print("Erro na Eliminação de Gauss:", e)

# Fatoração de Cholesky
try:
    G = cholesky_ruggiero(A.copy())
    y = solve_triangular(G, b, lower=True)
    x_cholesky = solve_triangular(G.T, y, lower=False)
    print("Cholesky:", x_cholesky)
except Exception as e:
    print("Cholesky falhou:", e)

# Gauss-Jacobi
try:
    x_jacobi, iter_jacobi = gauss_jacobi(A, b, x0)
    print(f"Gauss-Jacobi: {x_jacobi} (em {iter_jacobi} iterações)")
except Exception as e:
    print("Erro no Gauss-Jacobi:", e)

# Gauss-Seidel
try:
    x_seidel, iter_seidel = gauss_seidel(A, b, x0)
    print(f"Gauss-Seidel: {x_seidel} (em {iter_seidel} iterações)")
except Exception as e:
    print("Erro no Gauss-Seidel:", e)