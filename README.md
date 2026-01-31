# analise_impacto_ortamenrio
Orçamento vs. Qualidade: Análise econométrica do impacto orçamentário nas 10 maiores Universidades Federais (2014-2024). Dashboard interativo desenvolvido em Python e Streamlit.


# 🎓 Monitor de Qualidade das Universidades Federais

O **Monitor de Qualidade das Universidades Federais** é uma plataforma analítica interativa para **avaliar o impacto do orçamento público sobre a qualidade do ensino superior**, utilizando dados oficiais do **INEP/MEC** e métodos econométricos aplicados.

⚠️ **Importante:** devido a limitações no processo de coleta e padronização dos dados orçamentários, a análise considera **apenas as 10 maiores universidades federais brasileiras**, selecionadas com base no porte institucional e disponibilidade consistente de informações.

🔗 **Acesse o painel online:**
[https://uf-igc-giimcd.streamlit.app/](https://uf-igc-giimcd.streamlit.app/)

---

## 🎯 Objetivo do Projeto

O projeto tem como objetivo:

* Investigar a relação entre **financiamento público** e **qualidade do ensino superior federal**;
* Avaliar possíveis efeitos da política de restrição fiscal introduzida pelo **Teto de Gastos (2017)**;
* Aplicar modelos econométricos de **dados em painel** em um ambiente visual e interativo;
* Apoiar análises acadêmicas, trabalhos de conclusão de curso e estudos em políticas públicas.

---

## 🔍 Escopo e Limitações dos Dados

* **Universo analisado:** 10 maiores universidades federais brasileiras;
* **Critério de seleção:** porte institucional e disponibilidade contínua de dados orçamentários e de qualidade;
* **Motivação da limitação:** restrições na coleta, padronização e consolidação das bases públicas;
* **Implicação:** os resultados **não devem ser generalizados** para todo o sistema federal de ensino superior.

---

## 🔍 Principais Funcionalidades

### 📊 Painel Analítico Interativo

Visualização dinâmica dos dados das universidades federais selecionadas ao longo do tempo.

* Evolução do **orçamento anual** (em milhões de reais);
* Evolução do **IGC** por instituição;
* Destaque individual de universidades para comparação visual.

---

### 📈 Evidência Descritiva

Ferramentas gráficas para análise exploratória dos dados.

* Séries temporais de orçamento e IGC;
* Gráfico de dispersão **log-log** entre orçamento (defasado) e IGC;
* Linha de tendência estimada por regressão OLS.

---

### 🧮 Modelos Econométricos

Estimativas econométricas para avaliação empírica.

* **Efeitos Fixos (FE):** controla heterogeneidades institucionais invariantes no tempo;
* **Efeitos Aleatórios (RE):** modelo alternativo sob hipótese de exogeneidade;
* **Diferença-em-Diferenças (DiD):** avalia alterações no efeito do orçamento após o **Teto de Gastos (2017)**.

Cada modelo apresenta:

* Coeficientes estimados;
* P-valores;
* Interpretação econômica automática.

---

## 🚧 PROJETO EM DESENVOLVIMENTO 🚧

Este projeto foi desenvolvido **exclusivamente para fins acadêmicos e didáticos**, no contexto de estudos em **economia, políticas públicas e avaliação educacional**.

⚠️ As conclusões dependem das hipóteses econométricas adotadas e da limitação do conjunto de dados analisado.

---

## 🧩 Estrutura do Projeto

```
├── codigo                  # Codigo em Phyton para a análise econometrica
├── app.py                  # Aplicativo principal (Streamlit)
├── dados/
│   └── dados_finais.xlsx   # Base de dados (10 maiores universidades federais)
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação
```

---

## 🚀 Como Executar Localmente

### 1️⃣ Pré-requisitos

* Python 3.9 ou superior

### 2️⃣ Ambiente virtual (recomendado)

**Windows**

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / Mac**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3️⃣ Executar a aplicação

```bash
streamlit run app.py
```

---

## 🛠️ Tecnologias Utilizadas

* Python
* Streamlit
* Pandas / NumPy
* Plotly
* Statsmodels / Linearmodels

---

## 📚 Fonte dos Dados

* **INEP / MEC** – Índice Geral de Cursos (IGC)
* **Portal da Transparência** Orçamento das universidades federais
* Elaboração própria

  
