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
print("\n" + "="*40)
print("Resolvendo o sistema 3x3")
print("="*40)

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
    
    
    
#Sistema 12 x 12:
def gerar_sistema_12x12():
    np.random.seed(7)  # Reprodutibilidade

    n = 12

    # Solução real arbitrária com frações e sinais variados
    x_real = np.array([1.5, -2.3, 0.7, 4.2, -3.1, 2.8, 1.1, -0.9, 3.3, -1.7, 2.2, 0.5])

    # Matriz A com valores entre -10 e 10
    A = np.random.randint(-10, 11, size=(n, n))
    A = (A + A.T) // 2  # Simétrica

    # Força dominância diagonal
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i])) + np.random.randint(3, 6)

    b = A @ x_real
    x0 = np.zeros(n)

    return A, b, x_real, x0

# ---------- Gerar o sistema ----------
A, b, x_real, x0 = gerar_sistema_12x12()

# ---------- Resolver com os quatro métodos ----------

# Eliminação de Gauss
x_gauss = gauss_ruggiero(A.copy(), b.copy())
erro_gauss = np.linalg.norm(x_gauss - x_real, ord=np.inf)

# Cholesky
G = cholesky_ruggiero(A.copy())
y = solve_triangular(G, b, lower=True)
x_cholesky = solve_triangular(G.T, y, lower=False)
erro_cholesky = np.linalg.norm(x_cholesky - x_real, ord=np.inf)

# Gauss-Jacobi
x_jacobi, iter_jacobi = gauss_jacobi(A, b, x0)
erro_jacobi = np.linalg.norm(x_jacobi - x_real, ord=np.inf)

# Gauss-Seidel
x_seidel, iter_seidel = gauss_seidel(A, b, x0)
erro_seidel = np.linalg.norm(x_seidel - x_real, ord=np.inf)

# ---------- Impressão dos resultados ----------

print("\n" + "="*40)
print("RESULTADOS: SISTEMA 12x12")
print("="*40)
print("\nSolução real (x_real):")
print(np.array2string(x_real, precision=6, floatmode='fixed'))
print(f"\n{'Método':<20} {'Iterações':<12} {'Erro absoluto máximo'}")
print("-"*55)
print(f"{'Eliminação de Gauss':<20} {'—':<12} {erro_gauss:.2e}")
print(f"{'Cholesky':<20} {'—':<12} {erro_cholesky:.2e}")
print(f"{'Gauss-Jacobi':<20} {iter_jacobi:<12} {erro_jacobi:.2e}")
print(f"{'Gauss-Seidel':<20} {iter_seidel:<12} {erro_seidel:.2e}")
