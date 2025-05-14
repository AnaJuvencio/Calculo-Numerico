import math

# ------------------ MÉTODOS ------------------

def bissecao(f, a, b, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c, i+1
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c, max_iter

def ponto_fixo(g, x0, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        try:
            x1 = g(x0)
        except OverflowError:
            print(f"Overflow na iteração {i+1} com x0 = {x0}")
            return None, i+1
        if abs(x1 - x0) < tol:
            return x1, i+1
        x0 = x1
    return x0, max_iter


def newton(f, df, x0, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        if df(x0) == 0:
            raise ValueError("Derivada zero.")
        x1 = x0 - f(x0) / df(x0)
        if abs(x1 - x0) < tol:
            return x1, i+1
        x0 = x1
    return x0, max_iter

def secante(f, x0, x1, tol=1e-6, max_iter=100):
    for i in range(max_iter):
        if f(x1) - f(x0) == 0:
            return x1, i+1
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x2 - x1) < tol:
            return x2, i+1
        x0, x1 = x1, x2
    return x1, max_iter

# ------------------ EXEMPLO 1 ------------------

def exemplo_1():
    print("\n--- Exemplo 1: f(x) = x³ - 9x + 3 ---")
    f = lambda x: x**3 - 9*x + 3
    df = lambda x: 3*x**2 - 9
    g = lambda x: (x**3 + 3)/9  # Transformação para ponto fixo
    a, b = 0, 1
    x0 = 0.5

    print("Bisseção:", bissecao(f, a, b))
    print("Ponto Fixo:", ponto_fixo(g, x0))
    print("Newton-Raphson:", newton(f, df, x0))
    print("Secante:", secante(f, a, b))

# ------------------ EXEMPLO 2 ------------------

def exemplo_2():
    print("\n--- Exemplo 2: f(x) = 2x³ - 20x - 13 ---")
    f = lambda x: 2*x**3 - 20*x - 13
    df = lambda x: 6*x**2 - 20
    g = lambda x: (2*x**3 - 13)/20  # Transformação para ponto fixo
    a, b = 3, 4
    x0 = 3.5

    print("Bisseção:", bissecao(f, a, b))
    print("Ponto Fixo:", ponto_fixo(g, x0))
    print("Newton-Raphson:", newton(f, df, x0))
    print("Secante:", secante(f, a, b))

# ------------------ EXEMPLO 3 (Não Polinomial) ------------------

def exemplo_3():
    print("\n--- Exemplo 3: f(x) = cos(x) - x ---")
    f = lambda x: math.cos(x) - x
    df = lambda x: -math.sin(x) - 1
    g = lambda x: math.cos(x)
    a, b = 0, 1
    x0 = 0.5

    print("Bisseção:", bissecao(f, a, b))
    print("Ponto Fixo:", ponto_fixo(g, x0))
    print("Newton-Raphson:", newton(f, df, x0))
    print("Secante:", secante(f, a, b))

# ------------------ EXECUTAR ------------------

if __name__ == "__main__":
    exemplo_1()
    exemplo_2()
    exemplo_3()
