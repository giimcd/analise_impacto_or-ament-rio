import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects

# ======================
# CONFIGURAÇÃO DA PÁGINA
# ======================
st.set_page_config(
    page_title="Monitor Federais (MVP)",
    layout="wide",
    page_icon="🎓"
)

st.title("🎓 Monitor de Qualidade: Universidades Federais")
st.markdown("""
**MVP – Análise Econométrica de Impacto Orçamentário**  
Investigação do efeito do orçamento sobre o IGC.
""")

# ======================
# SIDEBAR
# ======================
st.sidebar.header("⚙️ Configurações")

uploaded_file = st.sidebar.file_uploader(
    "📂 Carregue seu arquivo (CSV ou Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("⬅️ Faça upload do arquivo para iniciar.")
    st.stop()

modelo_tipo = st.sidebar.radio(
    "📊 Modelo Econométrico",
    ["Efeitos Fixos (FE)", "Efeitos Aleatórios (RE)", "DiD (Mudança Estrutural)"]
)

# ======================
# FUNÇÃO DE CARGA (CACHE)
# ======================
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

    df = df.rename(columns={
        "Orçamento(GND 3+4)": "Orcamento",
        "IGC (Contínuo)": "IGC",
        "IGC (Continuo)": "IGC",
        "Ano ": "Ano"
    })

    df = df.sort_values(["Universidade", "Ano"])

    # Interpolação do IGC
    df["IGC"] = df.groupby("Universidade")["IGC"]\
        .transform(lambda x: x.interpolate().ffill())

    df["Orcamento_Milhoes"] = df["Orcamento"] / 1_000_000
    df["ln_Orcamento"] = np.log(df["Orcamento"])
    df["ln_IGC"] = np.log(df["IGC"])

    # DiD
    df["Pos_Teto"] = (df["Ano"] >= 2017).astype(int)
    df["Interacao"] = df["ln_Orcamento"] * df["Pos_Teto"]

    return df

df = carregar_dados(uploaded_file)

# ======================
# FILTRO DE UNIVERSIDADE
# ======================
uni_selecionada = st.selectbox(
    "🏫 Destaque uma universidade",
    df["Universidade"].unique()
)

# ======================
# ABAS
# ======================
tab1, tab2 = st.tabs(["📈 Visualização", "🧮 Econometria"])

# ======================
# TAB 1 – GRÁFICOS
# ======================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        fig_orc = px.line(
            df,
            x="Ano",
            y="Orcamento_Milhoes",
            color="Universidade"
        )
        fig_orc.update_traces(opacity=0.2)
        fig_orc.update_traces(
            selector=dict(name=uni_selecionada),
            opacity=1,
            line=dict(width=4)
        )
        st.plotly_chart(fig_orc, use_container_width=True)

    with col2:
        fig_igc = px.line(
            df,
            x="Ano",
            y="IGC",
            color="Universidade"
        )
        fig_igc.update_traces(opacity=0.2)
        fig_igc.update_traces(
            selector=dict(name=uni_selecionada),
            opacity=1,
            line=dict(width=4)
        )
        st.plotly_chart(fig_igc, use_container_width=True)

# ======================
# TAB 2 – MODELOS
# ======================
with tab2:
    st.subheader(f"Resultado: {modelo_tipo}")

    resultado_box = st.empty()

    df_panel = df.set_index(["Universidade", "Ano"])
    exog = sm.add_constant(df_panel[["ln_Orcamento"]])

    if modelo_tipo == "Efeitos Fixos (FE)":
        mod = PanelOLS(df_panel["ln_IGC"], exog, entity_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        coef = res.params["ln_Orcamento"]
        p_val = res.pvalues["ln_Orcamento"]

        resultado_box.text(res.summary.as_text())

    elif modelo_tipo == "Efeitos Aleatórios (RE)":
        mod = RandomEffects(df_panel["ln_IGC"], exog)
        res = mod.fit()

        coef = res.params["ln_Orcamento"]
        p_val = res.pvalues["ln_Orcamento"]

        resultado_box.text(res.summary.as_text())

    else:
        formula = "ln_IGC ~ ln_Orcamento + Pos_Teto + Interacao + C(Universidade)"
        mod = sm.formula.ols(formula, data=df)
        res = mod.fit(
            cov_type="cluster",
            cov_kwds={"groups": df["Universidade"]}
        )

        coef = res.params["Interacao"]
        p_val = res.pvalues["Interacao"]

        resultado_box.text(res.summary().as_text())

    # ======================
    # INTERPRETAÇÃO
    # ======================
    st.divider()
    colA, colB = st.columns(2)

    colA.metric("Coeficiente", f"{coef:.4f}")
    colB.metric("P-valor", f"{p_val:.4f}")

    if p_val < 0.05:
        st.success("✅ Evidência estatística de impacto do orçamento no IGC.")
    else:
        st.warning("⚠️ Não há evidência estatística robusta de impacto.")
