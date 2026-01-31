import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects

# -----------------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------------
st.set_page_config(
    page_title="Monitor Federais (MVP)",
    layout="wide",
    page_icon="🎓"
)

st.title("🎓 Monitor de Qualidade: Universidades Federais")
st.markdown("""
**MVP – Análise Econométrica do Impacto Orçamentário**

Este painel investiga se **alterações no orçamento público**
impactam a **qualidade acadêmica (IGC)** das universidades federais.
""")

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.header("⚙️ Configurações")

uploaded_file = st.sidebar.file_uploader(
    "📂 1. Carregue o arquivo de dados",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("👈 Faça o upload do arquivo para iniciar a análise.")
    st.stop()

# -----------------------------------
# CARREGAMENTO DE DADOS
# -----------------------------------
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

    df["IGC"] = df.groupby("Universidade")["IGC"].transform(
        lambda x: x.interpolate().ffill()
    )

    df["Orcamento_Milhoes"] = df["Orcamento"] / 1_000_000
    df["ln_Orcamento"] = np.log(df["Orcamento"])
    df["ln_IGC"] = np.log(df["IGC"])

    df["Pos_Teto"] = (df["Ano"] >= 2017).astype(int)
    df["Interacao"] = df["ln_Orcamento"] * df["Pos_Teto"]

    return df

df = carregar_dados(uploaded_file)

lista_unis = df["Universidade"].unique()
uni_selecionada = st.sidebar.selectbox(
    "🏫 2. Universidade em destaque",
    lista_unis
)

modelo_tipo = st.sidebar.radio(
    "📊 3. Modelo Econométrico",
    ["Efeitos Fixos (FE)", "Efeitos Aleatórios (RE)", "DiD (Mudança Estrutural)"]
)

# -----------------------------------
# TABS
# -----------------------------------
tab1, tab2 = st.tabs(
    ["📈 Visualização dos Dados", "🧮 Resultados Econométricos"]
)

# -----------------------------------
# TAB 1 — GRÁFICOS
# -----------------------------------
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

        fig_orc.update_traces(opacity=0.25)
        fig_orc.update_traces(
            selector=dict(name=uni_selecionada),
            opacity=1,
            line=dict(width=4)
        )

        fig_orc.add_vline(
            x=2017,
            line_dash="dash",
            annotation_text="Teto de Gastos"
        )

        st.plotly_chart(fig_orc, use_container_width=True)
        st.caption(
            "A linha tracejada marca o início do Teto de Gastos (2017)."
        )

    with col2:
        st.subheader("Evolução da Qualidade Acadêmica (IGC)")
        fig_igc = px.line(
            df,
            x="Ano",
            y="IGC",
            color="Universidade",
            markers=True
        )

        fig_igc.update_traces(opacity=0.25)
        fig_igc.update_traces(
            selector=dict(name=uni_selecionada),
            opacity=1,
            line=dict(width=4)
        )

        fig_igc.add_vline(
            x=2017,
            line_dash="dash",
            annotation_text="Teto de Gastos"
        )

        st.plotly_chart(fig_igc, use_container_width=True)
        st.caption(
            "O IGC tende a reagir de forma gradual a mudanças no orçamento."
        )

# -----------------------------------
# TAB 2 — ECONOMETRIA
# -----------------------------------
with tab2:
    st.subheader(f"Modelo selecionado: {modelo_tipo}")

    df_panel = df.set_index(["Universidade", "Ano"])
    exog = sm.add_constant(df_panel[["ln_Orcamento"]])

    if modelo_tipo == "Efeitos Fixos (FE)":
        st.markdown("""
        **Efeitos Fixos (FE)** controlam características
        não observáveis e constantes de cada universidade.
        """)
        mod = PanelOLS(
            df_panel["ln_IGC"],
            exog,
            entity_effects=True
        )
        res = mod.fit()
        coef = res.params["ln_Orcamento"]
        p_val = res.pvalues["ln_Orcamento"]
        st.text(res.summary)

    elif modelo_tipo == "Efeitos Aleatórios (RE)":
        st.markdown("""
        **Efeitos Aleatórios (RE)** assumem que as diferenças
        entre universidades são aleatórias.
        """)
        mod = RandomEffects(
            df_panel["ln_IGC"],
            exog
        )
        res = mod.fit()
        coef = res.params["ln_Orcamento"]
        p_val = res.pvalues["ln_Orcamento"]
        st.text(res.summary)

    else:
        st.markdown("""
        **Diferenças-em-Diferenças (DiD)** avalia se o impacto
        do orçamento mudou após o Teto de Gastos (2017).
        """)
        mod = sm.formula.ols(
            "ln_IGC ~ ln_Orcamento + Pos_Teto + Interacao + C(Universidade)",
            data=df
        )
        res = mod.fit(
            cov_type="cluster",
            cov_kwds={"groups": df["Universidade"]}
        )
        coef = res.params["Interacao"]
        p_val = res.pvalues["Interacao"]
        st.write(res.summary())

    # -----------------------------------
    # INTERPRETAÇÃO
    # -----------------------------------
    st.divider()
    st.subheader("📌 Interpretação Econômica dos Resultados")

    col1, col2 = st.columns(2)
    col1.metric("Coeficiente estimado", f"{coef:.4f}")
    col2.metric("P-valor", f"{p_val:.4f}")

    if p_val < 0.05:
        st.success("Resultado estatisticamente significativo.")
        st.markdown(
            f"Um aumento de **1% no orçamento** está associado a "
            f"uma variação média de **{coef:.2f}% no IGC**."
        )
    else:
        st.warning("Resultado não estatisticamente significativo.")
        st.markdown(
            "Não há evidência robusta de impacto imediato do orçamento "
            "sobre o IGC, possivelmente devido à inércia do indicador."
        )
