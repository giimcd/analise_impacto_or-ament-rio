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
    page_title="Monitor Federais (MVP)",
    layout="wide",
    page_icon="🎓"
)

# ===============================
# CABEÇALHO
# ===============================
st.title("🎓 Monitor de Qualidade: Universidades Federais")
st.markdown("""
**MVP – Análise Econométrica do Impacto Orçamentário**

Este painel investiga a hipótese de que cortes orçamentários afetam a qualidade acadêmica,
medida pelo **Índice Geral de Cursos (IGC)** das universidades federais.

*Projeto acadêmico – Econometria / Dados em Painel*
""")

# ===============================
# SIDEBAR – INPUTS
# ===============================
st.sidebar.header("⚙️ Configurações")

uploaded_file = st.sidebar.file_uploader(
    "📂 1. Carregue o arquivo de dados (CSV ou Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("👈 Faça o upload do arquivo de dados para iniciar a análise.")
    st.stop()

# ===============================
# CARREGAMENTO E TRATAMENTO DOS DADOS
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

    # Padronização de nomes
    df.columns = df.columns.str.strip()
    mapa = {
        "Orçamento(GND 3+4)": "Orcamento",
        "IGC (Contínuo)": "IGC",
        "IGC (Continuo)": "IGC",
        "Ano ": "Ano"
    }
    df = df.rename(columns=mapa)

    # Ordenação
    df = df.sort_values(["Universidade", "Ano"])

    # Interpolação do IGC
    df["IGC"] = (
        df.groupby("Universidade")["IGC"]
        .transform(lambda x: x.interpolate().ffill())
    )

    # Transformações
    df["Orcamento_Milhoes"] = df["Orcamento"] / 1_000_000
    df["ln_Orcamento"] = np.log(df["Orcamento"])
    df["ln_IGC"] = np.log(df["IGC"])

    # Variáveis institucionais
    df["Pos_Teto"] = (df["Ano"] >= 2017).astype(int)
    df["Interacao"] = df["ln_Orcamento"] * df["Pos_Teto"]

    return df

df = carregar_dados(uploaded_file)

# ===============================
# FILTROS
# ===============================
lista_unis = sorted(df["Universidade"].unique())
uni_selecionada = st.sidebar.selectbox(
    "🏫 2. Universidade em destaque",
    lista_unis
)

modelo_tipo = st.sidebar.radio(
    "📊 3. Modelo Econométrico",
    ["Efeitos Fixos (FE)", "Efeitos Aleatórios (RE)", "Diferença-em-Diferenças (DiD)"]
)

# ===============================
# TABS
# ===============================
tab1, tab2 = st.tabs(
    ["📈 Visualização dos Dados", "🧮 Resultados Econométricos"]
)

# ===============================
# TAB 1 – VISUALIZAÇÃO
# ===============================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evolução do Orçamento (R$ milhões)")
        fig_orc = px.line(
            df,
            x="Ano",
            y="Orcamento_Milhoes",
            color="Universidade",
            markers=True
        )
        fig_orc.update_traces(opacity=0.2)
        fig_orc.update_traces(
            selector=dict(name=uni_selecionada),
            opacity=1,
            line=dict(width=4)
        )
        st.plotly_chart(fig_orc, use_container_width=True)

    with col2:
        st.subheader("Evolução do IGC")
        fig_igc = px.line(
            df,
            x="Ano",
            y="IGC",
            color="Universidade",
            markers=True
        )
        fig_igc.update_traces(opacity=0.2)
        fig_igc.update_traces(
            selector=dict(name=uni_selecionada),
            opacity=1,
            line=dict(width=4)
        )
        st.plotly_chart(fig_igc, use_container_width=True)
        st.caption("Obs.: valores interpolados para anos sem divulgação.")

# ===============================
# TAB 2 – MODELOS
# ===============================
with tab2:
    st.header(f"Modelo Selecionado: {modelo_tipo}")

    df_panel = df.set_index(["Universidade", "Ano"])

    if modelo_tipo == "Efeitos Fixos (FE)":
        st.info("Controla características fixas das universidades.")
        exog = df_panel[["ln_Orcamento"]]
        mod = PanelOLS(df_panel["ln_IGC"], exog, entity_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        st.text(res.summary)

        coef = res.params["ln_Orcamento"]
        p_val = res.pvalues["ln_Orcamento"]

    elif modelo_tipo == "Efeitos Aleatórios (RE)":
        st.info("Assume heterogeneidade aleatória entre universidades.")
        exog = sm.add_constant(df_panel[["ln_Orcamento"]])
        mod = RandomEffects(df_panel["ln_IGC"], exog)
        res = mod.fit()
        st.text(res.summary)

        coef = res.params["ln_Orcamento"]
        p_val = res.pvalues["ln_Orcamento"]

    else:
        st.info("Avalia mudança estrutural após o Teto de Gastos (2017).")
        formula = "ln_IGC ~ ln_Orcamento + Pos_Teto + Interacao + C(Universidade)"
        mod = sm.formula.ols(formula, data=df)
        res = mod.fit(cov_type="cluster", cov_kwds={"groups": df["Universidade"]})
        st.text(res.summary())

        coef = res.params["Interacao"]
        p_val = res.pvalues["Interacao"]

    # ===============================
    # INTERPRETAÇÃO AUTOMÁTICA
    # ===============================
    st.divider()
    st.subheader("🤖 Interpretação Automática")

    col_a, col_b = st.columns(2)
    col_a.metric("Coeficiente", f"{coef:.4f}")
    col_b.metric("P-valor", f"{p_val:.4f}")

    if p_val < 0.05:
        st.success("Resultado estatisticamente significativo.")
        st.markdown(
            f"Um aumento de **1% no orçamento** está associado a uma variação de "
            f"**{coef:.4f}% no IGC**."
        )
    else:
        st.warning("Resultado não estatisticamente significativo.")
        st.markdown(
            "O efeito pode ser diluído pela **inércia do IGC**, "
            "que reage lentamente a mudanças orçamentárias."
        )

