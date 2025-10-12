# Projeto: Previsão de Inadimplência de Crédito

![Status](https://img.shields.io/badge/status-concluído-success)

Este projeto foi desenvolvido para a disciplina de **Introdução ao Aprendizado de Máquina** (UFRJ), com o objetivo de criar um modelo preditivo para avaliar o risco de inadimplência em solicitações de crédito. A avaliação final foi realizada em uma competição na plataforma Kaggle. 

---

### 🏆 Resultado Principal

O modelo desenvolvido, após um ciclo iterativo de pré-processamento, engenharia de atributos e otimização de hiperparâmetros, alcançou o **1º lugar no ranking de desempenho (Trabalho 1)** entre os alunos da turma.

---

### 📖 Metodologia Aplicada

O processo completo está detalhado no `Relatorio_Tecnico.pdf`, mas as principais etapas foram:
1.  **Análise Exploratória e Limpeza de Dados:** Identificação e tratamento de valores nulos e análise da variável alvo.
2.  **Engenharia de Atributos:** Criação de novas features (`proporcao_renda_extra`, `numero_de_cartoes`, etc.) para extrair mais informação dos dados brutos.
3.  **Seleção de Atributos:** Uso de matriz de correlação para remover ruído e redundância.
4.  **Comparação de Modelos:** Avaliação de múltiplos algoritmos (Regressão Logística, Random Forest, SVM) usando Validação Cruzada.
5.  **Otimização de Hiperparâmetros:** Utilização de `GridSearchCV` e `RandomizedSearchCV` para encontrar a melhor configuração para o modelo campeão (Random Forest).

---

### 🛠️ Tecnologias Utilizadas
- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn

---

### 🚀 Como Executar
1. Clone o repositório:
   ```
   git clone https://github.com/MateusElyachar/Projeto-Intro-Aprendizado-Maquina-UFRJ.git
   ```
2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```
3. O código principal com todo o processo está no arquivo `Trabalho 1.py`


[A colocação pode ser vista aqui!](https://www.kaggle.com/competitions/eel-891-2025-01-trabalho-1/leaderboard)
