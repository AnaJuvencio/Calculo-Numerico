import numpy as np
import matplotlib.pyplot as plt

def gauss_ruggiero(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(A)
    for k in range(n - 1):
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k] = 0
            for j in range(k + 1, n):
                A[i, j] -= m * A[k, j]
            b[i] -= m * b[k]
    x = np.zeros(n)
    x[n - 1] = b[n - 1] / A[n - 1, n - 1]
    for k in range(n - 2, -1, -1):
        s = sum(A[k, j] * x[j] for j in range(k + 1, n))
        x[k] = (b[k] - s) / A[k, k]
    return x


# ANÁLISE: ANOS INICIAIS
anos_iniciais = np.array([2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023])
notas_iniciais = np.array([4.9, 4.8, 5.8, 5.9, 6.0, 6.5, 6.7, 6.7, 6.2, 6.3])

x1 = anos_iniciais - 2005
y1 = notas_iniciais

# Sistema normal
n1 = len(x1)
soma_x1 = np.sum(x1)
soma_x1_2 = np.sum(x1**2)
soma_y1 = np.sum(y1)
soma_x1y1 = np.sum(x1 * y1)

A1 = np.array([[soma_x1_2, soma_x1], [soma_x1, n1]])
b1 = np.array([soma_x1y1, soma_y1])
a1, b01 = gauss_ruggiero(A1, b1)

x_meta1 = (6.8 - b01) / a1
ano_meta1 = x_meta1 + 2005

# Gráfico Anos Iniciais
x_plot1 = np.linspace(min(x1), max(x1) + 5, 100)
y_plot1 = a1 * x_plot1 + b01

plt.figure(figsize=(10, 6))
plt.plot(x_plot1 + 2005, y_plot1, label='Reta ajustada', color='blue')
plt.scatter(anos_iniciais, notas_iniciais, color='red', label='Dados reais')
plt.axhline(6.8, color='green', linestyle='--', label='Meta INEP (6.8)')
plt.axvline(ano_meta1, color='orange', linestyle='--', label=f"Ano estimado: {ano_meta1:.2f}")
plt.title('Ajuste por Mínimos Quadrados (IDEB - Anos Iniciais)')
plt.xlabel('Ano')
plt.ylabel('Nota IDEB')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.figure(1)

# ANÁLISE: ANOS FINAIS

anos_finais = np.array([2005, 2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021, 2023])
notas_finais = np.array([4.4, 4.3, 4.6, 4.6, 4.7, 5.0, 5.2, 5.5, 5.5, 5.5])

x2 = anos_finais - 2005
y2 = notas_finais

n2 = len(x2)
soma_x2 = np.sum(x2)
soma_x2_2 = np.sum(x2**2)
soma_y2 = np.sum(y2)
soma_x2y2 = np.sum(x2 * y2)

A2 = np.array([[soma_x2_2, soma_x2], [soma_x2, n2]])
b2 = np.array([soma_x2y2, soma_y2])
a2, b02 = gauss_ruggiero(A2, b2)

x_meta2 = (6.8 - b02) / a2
ano_meta2 = x_meta2 + 2005

# Questão 1d - Estimar quando será atingida a nota 6.3 (anos finais)
nota_meta_63 = 6.3
x_estimado_63 = (nota_meta_63 - b02) / a2
ano_estimado_63 = 2005 + x_estimado_63


# Gráfico Anos Finais
x_plot2 = np.linspace(min(x2), max(x2) + 5, 100)
y_plot2 = a2 * x_plot2 + b02

plt.figure(figsize=(10, 6))
plt.plot(x_plot2 + 2005, y_plot2, label='Reta ajustada', color='blue')
plt.scatter(anos_finais, notas_finais, color='red', label='Dados reais')
plt.axhline(6.8, color='green', linestyle='--', label='Meta INEP (6.8)')
plt.axvline(ano_meta2, color='orange', linestyle='--', label=f"Ano estimado: {ano_meta2:.2f}")
plt.title('Ajuste por Mínimos Quadrados (IDEB - Anos Finais)')
plt.xlabel('Ano')
plt.ylabel('Nota IDEB')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.figure(2)


# === Gráfico para a meta 6.3 (Questão 1d) ===
plt.figure(figsize=(9, 5))
plt.plot(x_plot2 + 2005, y_plot2, label='Reta ajustada', color='blue')
plt.scatter(anos_finais, notas_finais, color='red', label='Dados reais')
plt.axhline(y=nota_meta_63, color='green', linestyle='--', label='Meta INEP (6.3)')
plt.axvline(x=ano_estimado_63, color='orange', linestyle='--', label=f'Ano estimado: {ano_estimado_63:.2f}')
plt.title('Ajuste por Mínimos Quadrados (IDEB - Anos Finais)\nMeta INEP 6.3')
plt.xlabel('Ano')
plt.ylabel('Nota IDEB')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.figure(3)


# Imprime os resultados antes de mostrar os gráficos
print("======= Análise IDEB Anos Iniciais =======")
print(f"Equação ajustada: y = {a1:.4f}x + {b01:.4f}")
print(f"A nota 6.8 será atingida em aproximadamente {ano_meta1:.2f}\n")

print("======= Análise IDEB Anos Finais =======")
print(f"Equação ajustada: y = {a2:.4f}x + {b02:.4f}")
print(f"A nota 6.8 será atingida em aproximadamente {ano_meta2:.2f}")

# Estimativa para a meta de 6.3
print(f"\n======= Meta de 6.3 (Anos Finais) =======")
print(f"A nota 6.3 será atingida em aproximadamente {ano_estimado_63:.2f}")

# Mostra os dois gráficos ao mesmo tempo
plt.show()

