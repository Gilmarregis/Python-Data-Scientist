import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
%matplotlib inline



#Importando as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
%matplotlib inline



#Visualizando os dados em forma de tabela

A = np.array([ [ 749, 1.],
     [724, 0.99],
     [699,0.89],
     [674,0.79],
     [649,0.68],
     [624,0.57],
     [599,0.47],
     [574,0.37],
     [549,0.28],
     [524,0.17],
     [499,0.07]])

df = pd.DataFrame(A, columns = ['nu (Frequência)','V_p (Pontencial)'])
print(df)

nu, Vp = 1e12*A[:,0], A[:,1]

print(nu, Vp)

#Frequência
nu=1e12*np.array([749,724,699,674,649,624,599,574,549,524,499],dtype=float)

#valores do potencial de paragem 𝑉_p (V)
Vp=np.array([1,0.99,0.89,0.79,0.68,0.57,0.47,0.37,0.28,0.17,0.07])

#Resolvendo o sistema linear para obter os coeficientes da equação linear
alpha_hat = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(Vp)

print(alpha_hat)

#Utilizando somente o valor de interesse
h = - alpha_hat[:1]

print(h)

# Valor predito para um próximo ponto de acordo com os dados obtidos.
yhat = A.dot(alpha_hat)

# Plot dos dados e do valor predito (yhat)
plt.scatter(nu, Vp)
plt.plot(nu, yhat, color='red')
plt.text(6.0,0.4,'h='+str(h))
plt.xlabel(r'$\nu$')
plt.ylabel(r'$V_p$(V)')
plt.legend([r'Ajuste Linear $y = ax +b$', r'Dados'])
plt.title('Determinação da constante de Planck')
plt.show()



# ------------------
# IA: Previsão com Regressão Linear (Machine Learning)
# ------------------
# Neste bloco, ao invés de usar apenas o modelo físico para ajuste,
# aplicamos um algoritmo de IA (Machine Learning) - Regressão Linear -
# para aprender a relação entre a frequência (ν) e o potencial de parada (Vp)
# a partir dos dados experimentais, e então prever novos valores.

from sklearn.linear_model import LinearRegression
import numpy as np

# Dados experimentais
# Frequência da luz incidente (ν) em Hz
# Potencial de parada (Vp) em Volts
# Obs: Valores fictícios apenas para exemplificação
nu = np.array([5.0e14, 5.5e14, 6.0e14, 6.5e14, 7.0e14]).reshape(-1, 1)  # Variável independente
Vp = np.array([0.5, 0.9, 1.3, 1.6, 2.0])  # Variável dependente

# Criando o modelo de Machine Learning
# A regressão linear vai ajustar Vp = m * ν + b sem depender explicitamente da equação física.
model = LinearRegression()

# "Treinando" o modelo com os dados experimentais
model.fit(nu, Vp)

# Definindo novas frequências para prever o potencial de parada
novas_frequencias = np.array([[7.5e14], [8.0e14]])

# Usando o modelo treinado para prever Vp para as novas frequências
Vp_previsto_ml = model.predict(novas_frequencias)

# Exibindo resultados das previsões
for freq, vp in zip(novas_frequencias.flatten(), Vp_previsto_ml):
    print(f"Frequência: {freq:.2e} Hz -> Vp previsto (IA/ML): {vp:.3f} V")

# ------------------
# Visualização gráfica
# ------------------
import matplotlib.pyplot as plt

# Pontos originais do experimento
plt.scatter(nu, Vp, color='blue', label='Dados experimentais')

# Linha de ajuste obtida pelo modelo de IA (ML)
plt.plot(nu, model.predict(nu), color='red', label='Ajuste IA (ML)')

# Pontos de previsão para novas frequências
plt.scatter(novas_frequencias, Vp_previsto_ml, color='green', label='Previsões IA')

# Configurações do gráfico
plt.xlabel('Frequência (Hz)')
plt.ylabel('Potencial de parada (V)')
plt.title('Ajuste e Previsão por IA (Regressão Linear - ML)')
plt.legend()
plt.grid(True)
plt.show()

