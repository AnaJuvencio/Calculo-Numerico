import math

# ------------------ MÉTODOS ------------------

def bissecao(f, a, b, tol):
    if f(a) * f(b) >= 0:
        print("O método da bisseção não pode ser aplicado: f(a)f(b) >= 0")
        return None, 0
    k = 1
    M = f(a)
    while True:
        x = (a + b) / 2
        if M * f(x) > 0:
            a = x
            M = f(a)
        else:
            b = x
        if (b - a) < tol:
            return (a + b) / 2, k
        k += 1


def ponto_fixo(g, f, x0, e1, e2, max_iter=10, convergencia=False, dg=None):
    try:
        if dg is None:
            dg = derivada_aproximada(g, x0)

        if abs(dg) >= 1:
            convergencia = True
        else:
            convergencia = False

        if abs(f(x0)) < e1:
            return x0, 0, convergencia

        k = 1
        while k <= max_iter:
            x1 = g(x0)
            if not isinstance(x1, float) or x1 != x1 or abs(x1) > 1e10:
                return None, k, convergencia
            if abs(f(x1)) < e1 or abs(x1 - x0) < e2:
                return x1, k, convergencia
            x0 = x1
            k += 1

        return x1, k, convergencia
    except OverflowError:
        print("Falha no método - ponto fixo:")
        print("Mensagem de erro: Overflow detectado.")
        return None, k, convergencia




def newton(f, df, x0, e1, e2, max_iter=10):
    try:
        if abs(f(x0)) < e1:
            return x0, 0

        k = 1
        while k <= max_iter:
            dfx0 = df(x0)

            # Verifica se a derivada é muito pequena ou nula (evita divisão por zero)
            if abs(dfx0) < 1e-12:
                print("Falha no método - newton:")
                print("Mensagem de erro: Derivada próxima de zero. Método falhou.")
                return None, k

            x1 = x0 - f(x0) / dfx0

            # Verifica overflow
            if not isinstance(x1, float) or x1 != x1 or abs(x1) > 1e10:
                print("Falha no método - newton:")
                print("Mensagem de erro: Overflow numérico detectado ou valor inválido.")
                return None, k

            if abs(f(x1)) < e1 or abs(x1 - x0) < e2:
                k += 1
                return x1, k

            x0 = x1
            k += 1

        return x1, k  # Retorna a última aproximação após max_iter
    except OverflowError:
        print("Overflow detectado.")
        return None, k


def secante(f, x0, x1, e1=1e-3, e2=1e-3, max_iter=10):
    try:
        #print(f"{'Iteração':<10}{'x':<20}{'f(x)':<20}")
        
        if abs(f(x0)) < e1:
            #print(f"{0:<10}{x0:<20.10f}{f(x0):<20.10f}")
            return x0, 0
        if abs(f(x1)) < e1 or abs(x1 - x0) < e2:
            #print(f"{1:<10}{x1:<20.10f}{f(x1):<20.10f}")
            return x1, 1

        k = 1
        while k <= max_iter:
            fx0 = f(x0)
            fx1 = f(x1)
            denom = fx1 - fx0
            if abs(denom) < 1e-12:
                print("Divisão por valor muito pequeno. Método falhou.")
                return None, k

            x2 = x1 - fx1 * (x1 - x0) / denom
            fx2 = f(x2)

            #print(f"{k:<10}{x2:<20.10f}{fx2:<20.10f}")

            if abs(fx2) < e1 or abs(x2 - x1) < e2:
                return x2, k

            x0, x1 = x1, x2
            k += 1

        return x2, k
    except OverflowError:
        print("Overflow detectado.")
        return None, k



# ------------------ EXEMPLO 1 ------------------

def exemplo_1():
    print("\n--- Exemplo 1: f(x) = x³ - 9x + 3 ---")
    f = lambda x: x**3 - 9*x + 3
    df = lambda x: 3*x**2 - 9
    g = lambda x: (x**3 + 3)/9  # Transformação para ponto fixo
    a, b = 0, 1
    x0 = 0.5    
    epsilon = 5e-4
    dg = derivada_aproximada(g, x0, epsilon)

    raiz, iteracoes = bissecao(f, a, b, tol=1e-3)
    raizpf, iteracoespf, convergencia = ponto_fixo(g, f, x0, e1=epsilon, e2=epsilon, max_iter=10, convergencia=False, dg=dg)
    raiznr, iteracoesnr = newton(f, df, x0, e1=1e-3, e2=1e-3, max_iter=10)
    raizs, iteracoess = secante(f, x0=0, x1=1, e1=1e-3, e2=1e-3)
    print(f"Bisseção: raiz ≈ {raiz}, iterações: {iteracoes}, intervalo: [{a}, {b}], tol = 1e-3")
    if raizpf is not None:
        print(f"Ponto Fixo: raiz ≈ {raizpf}, iterações: {iteracoespf}, ε₁=5e-4, ε₂=5e-4")
    else:
        print(f"Ponto Fixo: falhou após {iteracoes}pf iterações.")
        if convergencia is True:
            print("Atenção: |g'({x0})| = {abs(dg):.4f} ≥ 1 ⇒ pode não convergir.")
        else:
            print("Falha no método - ponto fixo:")
            print("Mensagem de erro: Overflow numérico detectado ou valor inválido.")
    
    if raiznr is not None:
        print(f"Newton-Raphson: raiz ≈ {raiznr}, iterações: {iteracoesnr}, ε₁=1e-4, ε₂=1e-4")
    else:
        print(f"Newton-Raphson: falhou após {iteracoesnr} iterações.")
        
    print(f"Secante: raiz ≈ {raizs}, iterações: {iteracoess}, ε₁=5e-4, ε₂=5e-4 ")

# ------------------ EXEMPLO 2 ------------------


def exemplo_2():
    print("\n--- Exemplo 2: f(x) = 2x³ - 20x - 13 ---")
    f = lambda x: 2*x**3 - 20*x - 13
    df = lambda x: 6*x**2 - 20
    g = lambda x: (2*x**3 - 13)/20
    a, b = 3, 4
    x0 = 3.5


    dg = derivada_aproximada(g, x0)
    raizpf, itpf, convergencia = ponto_fixo(g, f, x0, e1=1e-3, e2=1e-3, max_iter=10, convergencia=False, dg=dg)
    raiz, iteracoes = bissecao(f, a, b, tol=1e-3)
    #raizpf, iteracoespf = ponto_fixo(g, f, x0, e1=1e-3, e2=1e-3, max_iter=10)
    raiznr, iteracoesnr = newton(f, df, x0, e1=1e-3, e2=1e-3, max_iter=10)
    raizs, iteracoess = secante(f, x0=a, x1=b, e1=1e-3, e2=1e-3)

    print(f"Bisseção: raiz ≈ {raiz}, iterações: {iteracoes}, intervalo: [{a}, {b}], tol = 1e-3")
    #if raizpf is not None:
    #    print(f"Ponto Fixo: raiz ≈ {raizpf}, iterações: {iteracoespf}, ε₁=1e-3, ε₂=1e-3")
    #else:
    #    print(f"Ponto Fixo: falhou após {iteracoespf} iterações.")
    if raizpf is not None:
        print(f"Ponto Fixo: raiz ≈ {raizpf}, iterações: {itpf}, ε₁=1e-3, ε₂=1e-3")
    else:
        print(f"Ponto Fixo: falhou após {itpf} iterações.")
        if convergencia is True:
            print("Atenção: |g'({x0})| = {abs(dg):.4f} ≥ 1 ⇒ pode não convergir.")
        else:
            print("Falha no método - ponto fixo:")
            print("Mensagem de erro: Overflow numérico detectado ou valor inválido.")
            
        
    if raiznr is not None:
        print(f"Newton-Raphson: raiz ≈ {raiznr}, iterações: {iteracoesnr}, ε₁=1e-3, ε₂=1e-3")
    else:
        print(f"Newton-Raphson: falhou após {iteracoesnr} iterações.")
        
    print(f"Secante: raiz ≈ {raizs}, iterações: {iteracoess}, ε₁=1e-3, ε₂=1e-3")


# ------------------ EXEMPLO 3 (Não Polinomial) ------------------

def exemplo_3():
    print("\n--- Exemplo 3: f(x) = cos(x) - x ---")
    f = lambda x: math.cos(x) - x
    df = lambda x: -math.sin(x) - 1
    g = lambda x: math.cos(x)
    a, b = 0, 1
    x0 = 0.5

    dg = derivada_aproximada(g, x0)
    raiz, iteracoes = bissecao(f, a, b, tol=1e-3)
    raizpf, iteracoespf, convergencia = ponto_fixo(g, f, x0, e1=1e-3, e2=1e-3, max_iter=10, convergencia=False, dg =dg)
    raiznr, iteracoesnr = newton(f, df, x0, e1=1e-3, e2=1e-3, max_iter=10)
    raizs, iteracoess = secante(f, x0=a, x1=b, e1=1e-3, e2=1e-3)

    print(f"Bisseção: raiz ≈ {raiz}, iterações: {iteracoes}, intervalo: [{a}, {b}], tol = 1e-3")
    if raizpf is not None:
        print(f"Ponto Fixo: raiz ≈ {raizpf}, iterações: {iteracoespf}, ε₁=1e-3, ε₂=1e-3")
    else:
        print(f"Ponto Fixo: falhou após {iteracoespf} iterações.")
        if convergencia is True:
            print("Atenção: |g'({x0})| = {abs(dg):.4f} ≥ 1 ⇒ pode não convergir.")
        else:
            print("Falha no método - ponto fixo:")
            print("Mensagem de erro: Overflow numérico detectado ou valor inválido.")
        
    if raiznr is not None:
        print(f"Newton-Raphson: raiz ≈ {raiznr}, iterações: {iteracoesnr}, ε₁=1e-3, ε₂=1e-3")
    else:
        print(f"Newton-Raphson: falhou após {iteracoesnr} iterações.")
        
    print(f"Secante: raiz ≈ {raizs}, iterações: {iteracoess}, ε₁=1e-3, ε₂=1e-3")


def derivada_aproximada(g, x, h=1e-3):
    return (g(x + h) - g(x - h)) / (2 * h)

x0 = 3.5
g = lambda x: (2*x**3 - 13)/20
dg = derivada_aproximada(g, x0)
        



# ------------------ EXECUTAR ------------------


if __name__ == "__main__":
    exemplo_1()
    exemplo_2()
    exemplo_3()
