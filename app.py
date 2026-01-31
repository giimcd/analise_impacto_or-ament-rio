import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects

# ======================================================
# CONFIGURAÇÃO DA PÁGINA
# ======================================================
st.set_page_config(
    page_title="Monitor de Qualidade – Universidades Federais",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Monitor de Qualidade das Universidades Federais")

st.markdown("""
**Avaliação do impacto do orçamento público sobre a qualidade do ensino superior**  
Indicador analisado: **IGC – Índice Geral de Cursos (INEP)**  

📌 Este painel aplica métodos econométricos de **dados em painel**
para investigar se variações orçamentárias estão associadas a mudanças
na qualidade acadêmica das universidades federais brasileiras.
""")

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("⚙️ Configurações")

st.sidebar.markdown("""
### ℹ️ Sobre o painel
- Unidades: Universidades Federais  
- Variável dependente: **IGC (log)**  
- Variável explicativa: **Orçamento (log)**  
- Métodos: FE, RE e DiD (Teto de Gastos – 2017)
""")

uploaded_file = st.sidebar.file_uploader(
    "📂 Carregue o banco de dados (CSV ou Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("⬅️ Faça o upload do arquivo para iniciar a análise.")
    st.stop()

modelo_tipo = st.sidebar.radio(
    "📊 Modelo econométrico",
    ["Efeitos Fixos (FE)", "Efeitos Aleatórios (RE)", "Diferença-em-Diferenças (DiD)"]
)

# ======================================================
# FUNÇÃO DE CARGA DE DADOS
# ======================================================
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
    df["IGC"] = df.groupby("Universidade")["IGC"] \
        .transform(lambda x: x.interpolate().ffill())

    # Transformações
    df["Orcamento_Milhoes"] = df["Orcamento"] / 1_000_000
    df["ln_Orcamento"] = np.log(df["Orcamento"])
    df["ln_IGC"] = np.log(df["IGC"])

    # Defasagem do orçamento (efeito mais realista)
    df["ln_Orcamento_lag"] = df.groupby("Universidade")["ln_Orcamento"].shift(1)

    # Variáveis DiD (Teto de Gastos – 2017)
    df["Pos_Teto"] = (df["Ano"] >= 2017).astype(int)
    df["Interacao"] = df["ln_Orcamento_lag"] * df["Pos_Teto"]

    return df.dropna()

df = carregar_dados(uploaded_file)

# ======================================================
# FILTRO DE UNIVERSIDADE
# ======================================================
uni_selecionada = st.selectbox(
    "🏫 Destaque uma universidade",
    df["Universidade"].unique()
)

# ======================================================
# ABAS
# ======================================================
tab1, tab2 = st.tabs(["📈 Evidência Descritiva", "🧮 Modelos Econométricos"])

# ======================================================
# TAB 1 – GRÁFICOS
# ======================================================
with tab1:
    st.subheader("Evolução temporal")

    col1, col2 = st.columns(2)

    with col1:
        fig_orc = px.line(
            df,
            x="Ano",
            y="Orcamento_Milhoes",
            color="Universidade",
            title="Orçamento das Universidades (R$ milhões)"
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
            color="Universidade",
            title="Evolução do IGC"
        )
        fig_igc.update_traces(opacity=0.2)
        fig_igc.update_traces(
            selector=dict(name=uni_selecionada),
            opacity=1,
            line=dict(width=4)
        )
        st.plotly_chart(fig_igc, use_container_width=True)

    st.subheader("Relação entre orçamento e qualidade")

    fig_scatter = px.scatter(
        df,
        x="ln_Orcamento_lag",
        y="ln_IGC",
        opacity=0.5,
        trendline="ols",
        title="Orçamento (t-1) e IGC (log-log)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ======================================================
# TAB 2 – ECONOMETRIA
# ======================================================
with tab2:
    st.subheader(f"Modelo selecionado: {modelo_tipo}")

    df_panel = df.set_index(["Universidade", "Ano"])
    exog = sm.add_constant(df_panel[["ln_Orcamento_lag"]])

    if modelo_tipo == "Efeitos Fixos (FE)":
        st.markdown("""
**Modelo de Efeitos Fixos**  
Controla características invariantes no tempo de cada universidade
(localização, tradição, perfil institucional).
""")
        mod = PanelOLS(df_panel["ln_IGC"], exog, entity_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        coef = res.params["ln_Orcamento_lag"]
        p_val = res.pvalues["ln_Orcamento_lag"]

        st.text(res.summary.as_text())

    elif modelo_tipo == "Efeitos Aleatórios (RE)":
        st.markdown("""
**Modelo de Efeitos Aleatórios**  
Assume que diferenças institucionais não observadas
não são correlacionadas com o orçamento.
""")
        mod = RandomEffects(df_panel["ln_IGC"], exog)
        res = mod.fit()

        coef = res.params["ln_Orcamento_lag"]
        p_val = res.pvalues["ln_Orcamento_lag"]

        st.text(res.summary.as_text())

    else:
        st.markdown("""
**Diferença-em-Diferenças (DiD)**  
Avalia se o efeito do orçamento mudou após a implementação
do Teto de Gastos (2017).
""")
        formula = "ln_IGC ~ ln_Orcamento_lag + Pos_Teto + Interacao + C(Universidade)"
        mod = sm.formula.ols(formula, data=df)
        res = mod.fit(
            cov_type="cluster",
            cov_kwds={"groups": df["Universidade"]}
        )

        coef = res.params["Interacao"]
        p_val = res.pvalues["Interacao"]

        st.text(res.summary().as_text())

    # ======================================================
    # INTERPRETAÇÃO
    # ======================================================
    st.divider()
    colA, colB = st.columns(2)

    colA.metric("Coeficiente estimado", f"{coef:.4f}")
    colB.metric("P-valor", f"{p_val:.4f}")

    st.markdown("### 📌 Interpretação econômica")

    if p_val < 0.05:
        st.success(
            f"Um aumento de **1% no orçamento** está associado a um aumento médio de "
            f"**{coef*100:.2f}% no IGC**, com significância estatística."
        )
    else:
        st.warning(
            "Não foi encontrada evidência estatística robusta de que o orçamento "
            "afete o IGC no período analisado."
        )
