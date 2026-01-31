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
    page_title="Monitor Federais | IGC",
    layout="wide",
    page_icon="🎓"
)

# ===============================
# CABEÇALHO
# ===============================
st.title("🎓 Monitor de Qualidade das Universidades Federais")
st.markdown("""
### Impacto dos Cortes Orçamentários sobre o IGC  
Análise econométrica com dados em painel (10 maiores universidades federais)

📌 *Projeto acadêmico – Econometria / Políticas Públicas*
""")

st.divider()

# ===============================
# CARREGAMENTO DOS DADOS
# ===============================
@st.cache_data
def carregar_dados():
    df = pd.read_excel("dados/dados_finais.xlsx")

    df.columns = df.columns.str.strip()

    mapa = {
        "Orçamento(GND 3+4)": "Orcamento",
        "IGC (Contínuo)": "IGC",
        "IGC (Continuo)": "IGC",
        "Ano ": "Ano"
    }
    df = df.rename(columns=mapa)

    df = df.sort_values(["Universidade", "Ano"])

    df["IGC"] = (
        df.groupby("Universidade")["IGC"]
        .transform(lambda x: x.interpolate().ffill())
    )

    df["Orcamento_Milhoes"] = df["Orcamento"] / 1_000_000
    df["ln_Orcamento"] = np.log(df["Orcamento"])
    df["ln_IGC"] = np.log(df["IGC"])

    df["Pos_Teto"] = (df["Ano"] >= 2017).astype(int)
    df["Interacao"] = df["ln_Orcamento"] * df["Pos_Teto"]

    return df

df = carregar_dados()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("⚙️ Controles")

uni_selecionada = st.sidebar.selectbox(
    "🏫 Universidade em destaque",
    sorted(df["Universidade"].unique())
)

modelo_tipo = st.sidebar.radio(
    "📊 Modelo Econométrico",
    [
        "Efeitos Fixos (FE)",
        "Efeitos Aleatórios (RE)",
        "Diferença-em-Diferenças (DiD)"
    ]
)

# ===============================
# MÉTRICAS GERAIS
# ===============================
col_m1, col_m2, col_m3 = st.columns(3)

col_m1.metric(
    "Universidades analisadas",
    df["Universidade"].nunique()
)
col_m2.metric(
    "Período",
    f"{df['Ano'].min()} – {df['Ano'].max()}"
)
col_m3.metric(
    "Ano do choque institucional",
    "2017 (Teto de Gastos)"
)

st.divider()

# ===============================
# TABS
# ===============================
tab1, tab2 = st.tabs(
    ["📈 Evolução dos Indicadores", "🧮 Resultados Econométricos"]
)

# ===============================
# TAB 1 – GRÁFICOS
# ===============================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Orçamento (R$ milhões)")
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
        st.subheader("Índice Geral de Cursos (IGC)")
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

        st.caption("Valores interpolados para anos sem divulgação oficial.")

# ===============================
# TAB 2 – MODELOS
# ===============================
with tab2:
    st.subheader(f"Modelo Selecionado: {modelo_tipo}")

    df_panel = df.set_index(["Universidade", "Ano"])

    if modelo_tipo == "Efeitos Fixos (FE)":
        st.info("Controla características fixas não observáveis das universidades.")
        exog = df_panel[["ln_Orcamento"]]

        mod = PanelOLS(
            df_panel["ln_IGC"],
            exog,
            entity_effects=True
        )

        res = mod.fit(
            cov_type="clustered",
            cluster_entity=True
        )

        coef = res.params["ln_Orcamento"]
        p_val = res.pvalues["ln_Orcamento"]

        st.text(res.summary)

    elif modelo_tipo == "Efeitos Aleatórios (RE)":
        st.info("Assume heterogeneidade aleatória entre universidades.")
        exog = sm.add_constant(df_panel[["ln_Orcamento"]])

        mod = RandomEffects(df_panel["ln_IGC"], exog)
        res = mod.fit()

        coef = res.params["ln_Orcamento"]
        p_val = res.pvalues["ln_Orcamento"]

        st.text(res.summary)

    else:
        st.info("Avalia mudança estrutural após o Teto de Gastos (2017).")

        formula = "ln_IGC ~ ln_Orcamento + Pos_Teto + Interacao + C(Universidade)"
        mod = sm.formula.ols(formula, data=df)
        res = mod.fit(
            cov_type="cluster",
            cov_kwds={"groups": df["Universidade"]}
        )

        coef = res.params["Interacao"]
        p_val = res.pvalues["Interacao"]

        st.text(res.summary())

    # ===============================
    # INTERPRETAÇÃO
    # ===============================
    st.divider()
    st.subheader("📌 Interpretação Automática")

    c1, c2 = st.columns(2)
    c1.metric("Coeficiente estimado", f"{coef:.4f}")
    c2.metric("P-valor", f"{p_val:.4f}")

    if p_val < 0.05:
        st.success("Resultado estatisticamente significativo.")
        st.markdown(
            f"Uma variação de **1% no orçamento** está associada a uma variação de "
            f"**{coef:.4f}% no IGC**."
        )
    else:
        st.warning("Resultado não estatisticamente significativo.")
        st.markdown(
            "O IGC apresenta **inércia**, reagindo lentamente a choques orçamentários."
        )


