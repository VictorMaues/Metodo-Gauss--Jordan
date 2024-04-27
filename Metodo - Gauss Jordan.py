# Método de Gauss-Jordan para resolver sistemas de equações lineares
# e encontrar a matriz inversa.

import numpy as np

def obter_matriz():
    """
    Solicita ao usuário o número de linhas e colunas da matriz
    e preenche a matriz com os dados fornecidos.
    """
    num_linhas = int(input("Digite o número de linhas da matriz: "))
    num_colunas = int(input("Digite o número de colunas da matriz: "))

    matriz = np.zeros((num_linhas, num_colunas))

    for i in range(num_linhas):
        for j in range(num_colunas):
            matriz[i, j] = float(input(f"Digite o elemento da posição ({i+1}, {j+1}): "))

    return matriz

def gauss_jordan(A, b):
    """
    Aplica o método de Gauss-Jordan para resolver o sistema de equações
    lineares representado pela matriz A e o vetor b.
    """
    n = len(b)
    AB = np.concatenate((A, b), axis=1)

    for i in range(n):
        max_index = np.argmax(abs(AB[i:, i])) + i
        AB[[i, max_index]] = AB[[max_index, i]]  # Troca as linhas para obter um pivô não nulo
        pivot = AB[i, i]
        AB[i] = AB[i] / pivot

        for j in range(n):
            if i != j:
                AB[j] = AB[j] - AB[j, i] * AB[i]

    x = AB[:, -1]
    return x

# Exemplo de uso
print("Este código resolve sistemas de equações lineares usando o método de Gauss-Jordan.")
print("="*82)
print("Digite os elementos da matriz e do vetor b para encontrar a solução.")
A = obter_matriz()
b = np.array([input("Digite os elementos do vetor b separados por espaço: ").split()], dtype=float)
resultado = gauss_jordan(A, b.T)

print("\nA matriz aumentada após a eliminação de Gauss-Jordan:")
print(np.concatenate((A, b.T), axis=1))
print("\nA solução é:", resultado)
