!pip install linearmodels
# ==============================================================================
# 🎓 PROJETO: IMPACTO DOS CORTES ORÇAMENTÁRIOS NA QUALIDADE DAS 10 MAIORES UNIVERSIDADES FEDERAIS (IGC)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects, compare
import os

# --- 1. CARREGAMENTO ROBUSTO DOS DADOS ---
print("--- 1. CARREGAMENTO E LIMPEZA ---")
# Procura o arquivo automaticamente
arquivos = os.listdir()
arquivo_alvo = None
for f in arquivos:
    if ("Base" in f or "dados" in f) and f.endswith((".csv", ".xlsx")):
        arquivo_alvo = f
        break

if arquivo_alvo:
    print(f"✅ Arquivo encontrado: {arquivo_alvo}")
    try:
        # Tenta ler (suporta CSV e Excel)
        if arquivo_alvo.endswith(".csv"):
            try:
                df = pd.read_csv(arquivo_alvo, sep=',')
            except:
                df = pd.read_csv(arquivo_alvo, sep=';', encoding='latin1')
        else:
            df = pd.read_excel(arquivo_alvo)
            
        # Padronização de Colunas
        df.columns = df.columns.str.strip()
        mapa = {
            'Orçamento(GND 3+4)': 'Orcamento',
            'IGC (Contínuo)': 'IGC', 'IGC (Continuo)': 'IGC', 'Ano ': 'Ano'
        }
        df = df.rename(columns=mapa)
        
        # Tratamento de Buracos (2020 e 2024)
        df = df.sort_values(by=['Universidade', 'Ano'])
        df['IGC'] = df.groupby('Universidade')['IGC'].transform(lambda x: x.interpolate(method='linear')) # 2020
        df['IGC'] = df.groupby('Universidade')['IGC'].transform(lambda x: x.ffill())# 2024
        
        # TRANSFORMAÇÃO LOG (Para elasticidade - Exigência Técnica)
        df['ln_Orcamento'] = np.log(df['Orcamento'])
        df['ln_IGC'] = np.log(df['IGC'])
        
        # Configuração do Painel
        df_panel = df.set_index(['Universidade', 'Ano'])
        
        print(f"✅ Dados carregados e tratados! Total de observações: {len(df)}")
        
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")
else:
    print("❌ ERRO: Nenhum arquivo de dados encontrado na pasta.")

# --- 2. ANÁLISE DE CORRELAÇÃO ---
if 'df' in locals():
    print("\n--- 2. ANÁLISE DE CORRELAÇÃO (PEARSON & SPEARMAN) ---")
    
    # Pearson (Linear)
    corr_p = df[['Orcamento', 'IGC']].corr(method='pearson').iloc[0,1]
    # Spearman (Não-Linear / Rank - EXIGIDO PELO PROFESSOR)
    corr_s = df[['Orcamento', 'IGC']].corr(method='spearman').iloc[0,1]
    
    print(f"🔹 Correlação de Pearson (Linear): {corr_p:.4f}")
    print(f"🔹 Correlação de Spearman (Rank):  {corr_s:.4f}")
    
    if corr_s > 0.3:
        print("💡 Interpretação: Existe uma correlação positiva moderada/forte.")
    else:
        print("💡 Interpretação: A correlação é fraca, indicando que a relação não é simples/direta.")

    # Visualização
    plt.figure(figsize=(10, 5))
    sns.regplot(x='ln_Orcamento', y='ln_IGC', data=df, scatter_kws={'s':50, 'alpha':0.6}, line_kws={'color':'red'})
    plt.title("Dispersão com Ajuste Linear: Log Orçamento x Log IGC")
    plt.grid(True, alpha=0.3)
    plt.show()

# --- 3. MODELAGEM ECONOMÉTRICA ---
if 'df_panel' in locals():
    print("\n--- 3. MODELOS DE PAINEL (FE vs RE) ---")
    exog = sm.add_constant(df_panel[['ln_Orcamento']])
    
    # Modelo Efeitos Fixos (FE) - Controla heterogeneidade da universidade
    mod_fe = PanelOLS(df_panel['ln_IGC'], exog, entity_effects=True)
    res_fe = mod_fe.fit()
    
    # Modelo Efeitos Aleatórios (RE)
    mod_re = RandomEffects(df_panel['ln_IGC'], exog)
    res_re = mod_re.fit()
    
    # Comparação (Hausman Lógico)
    print(compare({'FE (Fixos)': res_fe, 'RE (Aleatórios)': res_re}))
    
    print("\n📝 DICA PARA O TESTE DE HAUSMAN:")
    print("Compare os coeficientes de 'ln_Orcamento' nas duas colunas acima.")
    print("Se forem muito diferentes, o Teste de Hausman rejeita o RE. Use o FE (Efeitos Fixos).")
    print("Justificativa: O FE controla características não observadas (tamanho, prestígio) que o RE ignora.")

# --- 4. DIFERENÇAS EM DIFERENÇAS / MUDANÇA ESTRUTURAL ---
if 'df' in locals():
    print("\n--- 4. ANÁLISE DE IMPACTO (DiD - TETO DE GASTOS) ---")
    # Definição do evento: Teto de Gastos (2017 em diante)
    df['Pos_Teto'] = (df['Ano'] >= 2017).astype(int)
    
    # Interação: O efeito do orçamento mudou depois de 2017?
    df['Interacao'] = df['ln_Orcamento'] * df['Pos_Teto']
    
    # Modelo OLS Robust
    modelo_did = sm.formula.ols(
        formula='ln_IGC ~ ln_Orcamento + Pos_Teto + Interacao + C(Universidade)', 
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['Universidade']})
    
    print(modelo_did.summary())
    
    # --- 5. CONCLUSÃO AUTOMÁTICA (TEXTO INTERPRETATIVO - MVP) ---
    print("\n=======================================================")
    print("🤖 CONCLUSÃO AUTOMÁTICA")
    print("=======================================================")
    
    p_valor = res_fe.pvalues['ln_Orcamento']
    coef = res_fe.params['ln_Orcamento']
    
    print(f"📊 P-Valor do Orçamento (Modelo FE): {p_valor:.4f}")
    
    if p_valor < 0.05:
        print("✅ RESULTADO: Rejeita-se a Hipótese Nula (H0).")
        print("INTERPRETAÇÃO: Existe evidência estatística significativa de que cortes orçamentários afetam a qualidade.")
        print(f"MAGNITUDE: Uma variação de 1% no orçamento está associada a uma variação de {coef:.4f}% no IGC.")
    else:
        print("⚠️ RESULTADO: Não se rejeita a Hipótese Nula (H0).")
        print("INTERPRETAÇÃO: Não foi encontrada evidência estatística robusta de impacto IMEDIATO.")
        print("DISCUSSÃO (Para sua defesa): O IGC possui inércia (demora a cair). O impacto dos cortes pode levar anos (lag) para aparecer nos indicadores.")

    print("\nLIMITAÇÕES :")
    print("- O IGC é composto pelo ENADE (trienal), gerando rigidez no indicador.")
    print("- O modelo não captura a eficiência de gestão (fazer mais com menos).")
    print("- Fatores externos (pandemia 2020) geram ruído nos dados.")
