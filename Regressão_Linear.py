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



