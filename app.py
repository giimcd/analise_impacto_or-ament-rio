import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects

# ===============================
# CONFIGURAÇÃO DA PÁGINA
# ===============================
st.set_page_config(
    page_title="Monitor de Universidades Federais",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Monitor de Qualidade das Universidades Federais")
st.markdown("""
Este painel apresenta uma **análise econométrica aplicada** para investigar  
se **variações orçamentárias impactam a qualidade acadêmica**, medida pelo **IGC**.

📘 *Aplicação prática para Econometria – Painel de Dados*
""")

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("⚙️ Configurações do Painel")

uploaded_file = st.sidebar.file_uploader(
    "📂 Envie o arquivo de dados (CSV ou Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("👈 Faça o upload do arquivo para iniciar a análise.")
    st.stop()

# ===============================
# CARREGAMENTO DE DADOS
# ===============================
@st.cache_data
def carregar_dados(file):
    if file.name.endswith(".csv"):
        try:
            df = pd.read_csv(file)
        except:
            df = pd.read_csv(file, sep=";", encoding="latin1")
    else:
        df = pd.read_excel(file)

    df.columns = df.columns.str.strip()

    mapa = {
        "Orçamento(GND 3+4)": "Orcamento",
        "IGC (Contínuo)": "IGC",
        "IGC (Continuo)": "IGC",
        "Ano ": "Ano"
    }
    df = df.rename(columns=mapa)

    df = df.sort_values(["Universidade", "Ano"])

    # Tratamento
    df["IGC"] = df.groupby("Universidade")["IGC"].transform(
        lambda x: x.interpolate().ffill()
    )

    df["Orcamento_Milhoes"] = df["Orcamento"] / 1_000_000
    df["ln_Orcamento"] = np.log(df["Orcamento"])
    df["ln_IGC"] = np.log(df["IGC"])

    # DiD
    df["Pos_Teto"] = (df["Ano"] >= 2017).astype(int)
    d
