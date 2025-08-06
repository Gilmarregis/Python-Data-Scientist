# 📊 Python para Cientistas — Ajuste de Dados Experimentais por Regressão Linear

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-orange)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-brightgreen)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-red)](https://scikit-learn.org/)

Este repositório contém um **notebook Jupyter** que demonstra como aplicar **Python** para ajuste de dados experimentais via **regressão linear** e determinação da **constante de Planck** a partir de um experimento de **célula fotoelétrica**.

## 🧪 Descrição do Experimento
O objetivo é calcular a constante de Planck **h** e a função trabalho **W** a partir de medições do **potencial de parada** em função da **frequência da luz incidente**.  

A partir da equação de Einstein para o efeito fotoelétrico:

\[
eV_p = h\nu - W
\]

realiza-se o ajuste linear \( y = mx + b \), onde:
- \( m = h/e \)  
- \( b = -W/e \)  

## 📦 Pacotes Utilizados
- **Python 3.8+**
- **NumPy**
- **Matplotlib**
- **Pandas**
- **scikit-learn** (para versão em Machine Learning)

## 📂 Estrutura do Projeto
```
📦 python-para-cientistas
 ┣ 📜 Aula_Doutorado.ipynb
 ┣ 📜 README.md
 ┣ 📜 requirements.txt
 ┗ 📜 LICENSE
```

## 🚀 Executando Localmente
```bash
pip install -r requirements.txt
jupyter notebook Aula_Doutorado.ipynb
```

## 📊 Funcionalidades
- Ajuste por regressão linear usando **método de mínimos quadrados**
- Visualização gráfica do ajuste e dados experimentais
- Implementação opcional via **Machine Learning (scikit-learn)** para previsões
- Cálculo de \( h \) e \( W \) a partir do ajuste

---

🔍 **Palavras-chave SEO**: python, física experimental, regressão linear, constante de Planck, ciência de dados, efeito fotoelétrico, machine learning
