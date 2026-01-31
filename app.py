import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Monitor Federais (MVP)", layout="wide", page_icon="🎓")

# --- CABEÇALHO ---
st.title("🎓 Monitor de Qualidade: Universidades Federais")
st.markdown("""
**MVP - Análise Econométrica de Impacto Orçamentário**
Este painel investiga a hipótese de que cortes orçamentários afetam a qualidade acadêmica (IGC).
*Desenvolvido para a disciplina de Econometria.*
""")

# --- 1. BARRA LATERAL (INPUTS) ---
st.sidebar.header("⚙️ Configurações")

# Upload de Arquivo
uploaded_file = st.sidebar.file_uploader("📂 1. Carregue seu arquivo CSV", type=["csv", "xlsx"])

# Se não tiver arquivo, usa um aviso
if uploaded_file is None:
    st.info("👆 Por favor, faça o upload do arquivo 'dados.csv' na barra lateral para começar.")
    st.stop()

# --- 2. CARREGAMENTO E LIMPEZA DE DADOS ---
@st.cache_data
def carregar_dados(file):
    try:
        if file.name.endswith('.csv'):
            try:
                df = pd.read_csv(file, sep=',')
            except:
                df = pd.read_csv(file, sep=';', encoding='latin1')
        else:
            df = pd.read_excel(file)

        # Padronização
        df.columns = df.columns.str.strip()
        mapa = {
            'Orçamento(GND 3+4)': 'Orcamento',
            'IGC (Contínuo)': 'IGC', 'IGC (Continuo)': 'IGC', 'Ano ': 'Ano'
        }
        df = df.rename(columns=mapa)

        # Tratamento de Buracos e Logs
        df = df.sort_values(by=['Universidade', 'Ano'])
        df['IGC'] = df.groupby('Universidade')['IGC'].transform(lambda x: x.interpolate(method='linear').fillna(method='ffill'))
        df['Orcamento_Milhoes'] = df['Orcamento'] / 1_000_000
        df['ln_Orcamento'] = np.log(df['Orcamento'])
        df['ln_IGC'] = np.log(df['IGC'])
        
        # Variáveis para DiD
        df['Pos_Teto'] = (df['Ano'] >= 2017).astype(int)
        df['Interacao'] = df['ln_Orcamento'] * df['Pos_Teto']

        return df
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
        return None

df = carregar_dados(uploaded_file)
if df is None:
    st.stop()

# --- FILTROS DA SIDEBAR ---
lista_unis = df['Universidade'].unique()
uni_selecionada = st.sidebar.selectbox("🏫 2. Escolha uma Universidade (Destaque)", lista_unis)
modelo_tipo = st.sidebar.radio("📊 3. Escolha o Modelo Econométrico", ["Efeitos Fixos (FE)", "Efeitos Aleatórios (RE)", "DiD (Mudança Estrutural)"])

# --- 3. DASHBOARD VISUAL (TAB 1) ---
tab1, tab2 = st.tabs(["📈 Visualização de Dados", "🧮 Resultados Econométricos"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Evolução do Orçamento (Milhões)")
        fig_orc = px.line(df, x="Ano", y="Orcamento_Milhoes", color="Universidade", markers=True)
        # Destaca a selecionada e apaga as outras
        fig_orc.update_traces(line=dict(width=1), opacity=0.2)
        fig_orc.update_traces(selector=dict(name=uni_selecionada), line=dict(width=4, color='red'), opacity=1)
        st.plotly_chart(fig_orc, use_container_width=True)

    with col2:
        st.subheader("Evolução da Qualidade (IGC)")
        fig_igc = px.line(df, x="Ano", y="IGC", color="Universidade", markers=True)
        fig_igc.update_traces(line=dict(width=1), opacity=0.2)
        fig_igc.update_traces(selector=dict(name=uni_selecionada), line=dict(width=4, color='blue'), opacity=1)
        st.plotly_chart(fig_igc, use_container_width=True)
        st.caption("*Nota: 2020 interpolado e 2024 projetado (inércia).")

# --- 4. MODELAGEM (TAB 2) ---
with tab2:
    st.header(f"Resultado do Modelo: {modelo_tipo}")
    
    # Prepara Painel
    df_panel = df.set_index(['Universidade', 'Ano'])
    exog_vars = ['ln_Orcamento']
    exog = sm.add_constant(df_panel[exog_vars])

    # Lógica de Seleção de Modelo
    if modelo_tipo == "Efeitos Fixos (FE)":
        st.info("ℹ️ O modelo FE controla as características fixas de cada universidade (ex: tamanho, reputação).")
        mod = PanelOLS(df_panel['ln_IGC'], exog, entity_effects=True)
        res = mod.fit()
        st.text(res.summary)
        
        # Pega dados para conclusão
        p_val = res.pvalues['ln_Orcamento']
        coef = res.params['ln_Orcamento']

    elif modelo_tipo == "Efeitos Aleatórios (RE)":
        st.info("ℹ️ O modelo RE assume que as diferenças entre universidades são aleatórias.")
        mod = RandomEffects(df_panel['ln_IGC'], exog)
        res = mod.fit()
        st.text(res.summary)
        p_val = res.pvalues['ln_Orcamento']
        coef = res.params['ln_Orcamento']

    elif modelo_tipo == "DiD (Mudança Estrutural)":
        st.info("ℹ️ Analisa se o impacto do orçamento mudou após o Teto de Gastos (2017).")
        # DiD usa OLS simples com interação
        formula = 'ln_IGC ~ ln_Orcamento + Pos_Teto + Interacao + C(Universidade)'
        mod = sm.formula.ols(formula=formula, data=df)
        res = mod.fit(cov_type='cluster', cov_kwds={'groups': df['Universidade']})
        st.write(res.summary())
        p_val = res.pvalues['Interacao']
        coef = res.params['Interacao']

    # --- 5. CONCLUSÃO AUTOMÁTICA (TEXTO INTERPRETATIVO) ---
    st.divider()
    st.subheader("🤖 Interpretação dos Resultados")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.metric("P-Valor (Significância)", f"{p_val:.4f}")
    with col_res2:
        st.metric("Coeficiente (Impacto)", f"{coef:.4f}")

    if p_val < 0.05:
        st.success("✅ CONCLUSÃO: Rejeita-se a Hipótese Nula (H0).")
        st.markdown(f"Existe evidência estatística de impacto. Uma variação de 1% no orçamento gera uma variação de **{coef:.4f}%** no IGC.")
    else:
        st.warning("⚠️ CONCLUSÃO: Não se rejeita a Hipótese Nula (H0).")
        st.markdown("Não foi encontrada evidência estatística robusta de impacto imediato neste modelo.")
        st.markdown("**Possível Causa:** O indicador IGC possui inércia (demora a reagir aos cortes).")
