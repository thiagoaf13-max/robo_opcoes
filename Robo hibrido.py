import time
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# ==============================
# CONFIGURAÇÕES
# ==============================
SHEET_ID = "114XbIrVhnToZlgHWZd8v9_L9Tvp_UlIiSw9P-f0OzFo"  # ajuste se necessário
SHEET_NAME_DADOS = "Dados"
SHEET_NAME_PREVISOES = "Previsões"
CREDENCIAIS_JSON = "credenciais.json"  # caminho para seu JSON de service account

NUM_LAGS = 12
LIMITE_CONFIANCA = 0.70
ALERTA_CONFIANCA = 0.90
FREQ_MIN = 5
TAMANHO_TAXA_MOVEL = 24

# ==============================
# UTILIDADES / LOG
# ==============================
def escrever_log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ==============================
# CONEXÃO COM GOOGLE SHEETS
# ==============================
def conectar_google_sheets():
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_file(CREDENCIAIS_JSON, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID)
        aba_dados = sheet.worksheet(SHEET_NAME_DADOS)
        aba_prev = sheet.worksheet(SHEET_NAME_PREVISOES)
        return aba_dados, aba_prev
    except gspread.SpreadsheetNotFound:
        escrever_log("❌ Planilha não encontrada (verifique SHEET_ID e compartilhamento da service account).")
        raise
    except Exception as e:
        escrever_log(f"❌ Erro ao conectar Google Sheets: {e}")
        raise

# ==============================
# LEITURA E PREPARAÇÃO DOS DADOS
# ==============================
def carregar_dados_google(aba_dados):
    try:
        registros = aba_dados.get_all_records()
        df = pd.DataFrame(registros)
        if df.empty:
            return df
        # normaliza nomes (remove espaços extras)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        escrever_log(f"❌ Erro ao carregar dados do Google Sheets: {e}")
        return pd.DataFrame()

def _achar_coluna(df, candidatos):
    # retorna nome real da coluna que casa com algum candidato (case-insensitive), ou None
    cols = list(df.columns)
    for cand in candidatos:
        for c in cols:
            if c.strip().lower() == cand.strip().lower():
                return c
    # tentativa de correspondência parcial
    for c in cols:
        low = c.strip().lower()
        for cand in candidatos:
            if cand.strip().lower() in low or low in cand.strip().lower():
                return c
    return None

def preparar_dados(df):
    """
    Recebe df cru (get_all_records) e retorna DataFrame com:
    - data (datetime)
    - resultado (1/0)
    - datetime_horario
    - hora, minuto, dia_semana
    - lag_1..lag_N
    Observações: não descarta totalmente linhas quando faltam lags (retorna todas
    e o loop principal decide o que usar).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    # identificar colunas (tolerante)
    col_data = _achar_coluna(df, ["data", "Data", "DATA"])
    col_sucesso = _achar_coluna(df, ["Horário Sucesso", "Horario Sucesso", "horário sucesso", "sucesso"])
    col_falha = _achar_coluna(df, ["Horário Falha", "Horario Falha", "horário falha", "falha"])

    if not col_data:
        escrever_log("❌ Coluna 'data' não encontrada (aguarda nome parecido com 'data').")
        return pd.DataFrame()

    # normaliza nomes internos para consistência
    df.rename(columns={col_data: "data"}, inplace=True)
    if col_sucesso:
        df.rename(columns={col_sucesso: "Horário Sucesso"}, inplace=True)
    if col_falha:
        df.rename(columns={col_falha: "Horário Falha"}, inplace=True)

    # parse data (aceita dd/mm/YYYY e variantes)
    df.loc[:, "data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
    antes = len(df)
    df = df.dropna(subset=["data"])
    if len(df) < antes:
        escrever_log(f"⚠️ {antes-len(df)} linhas removidas por 'data' inválida.")

    # Coluna resultado (1 = sucesso, 0 = falha, None se nenhum)
    def achar_resultado(row):
        if "Horário Sucesso" in row.index and pd.notna(row.get("Horário Sucesso")) and str(row.get("Horário Sucesso")).strip() != "":
            return 1
        if "Horário Falha" in row.index and pd.notna(row.get("Horário Falha")) and str(row.get("Horário Falha")).strip() != "":
            return 0
        return None

    df.loc[:, "resultado"] = df.apply(achar_resultado, axis=1)

    # Criar datetime_horario usando Horário Sucesso primeiro, senão Horário Falha
    def parse_horario(row):
        hora_val = None
        if pd.notna(row.get("Horário Sucesso")) and str(row.get("Horário Sucesso")).strip() != "":
            hora_val = str(row.get("Horário Sucesso")).strip()
        elif pd.notna(row.get("Horário Falha")) and str(row.get("Horário Falha")).strip() != "":
            hora_val = str(row.get("Horário Falha")).strip()
        if hora_val is None:
            return pd.NaT
        # tenta combinar com data; aceita H:M, H:M:S e com/sem zeros à esquerda
        try:
            return pd.to_datetime(f"{row['data'].strftime('%Y-%m-%d')} {hora_val}", errors="coerce")
        except Exception:
            return pd.NaT

    df.loc[:, "datetime_horario"] = df.apply(parse_horario, axis=1)
    antes = len(df)
    df = df.dropna(subset=["datetime_horario"])
    if len(df) < antes:
        escrever_log(f"⚠️ {antes-len(df)} linhas removidas por 'horário' inválido.")

    # extrair hora/minuto/dia
    df.loc[:, "hora"] = df["datetime_horario"].dt.hour
    df.loc[:, "minuto"] = df["datetime_horario"].dt.minute
    df.loc[:, "dia_semana"] = df["datetime_horario"].dt.dayofweek

    # criar lags (mantemos mesmo que incompletos)
    for lag in range(1, NUM_LAGS + 1):
        df.loc[:, f"lag_{lag}"] = df["resultado"].shift(lag)

    # criar slot string HH:MM
    df.loc[:, "slot"] = df["hora"].astype(int).astype(str).str.zfill(2) + ":" + df["minuto"].astype(int).astype(str).str.zfill(2)

    return df

# ==============================
# ESTATÍSTICAS POR HORÁRIO
# ==============================
def calcular_probabilidade_horario(df):
    if df is None or df.empty:
        return {}
    tmp = df.dropna(subset=["resultado"]).copy()
    if tmp.empty:
        return {}
    prob = tmp.groupby("slot")["resultado"].mean().to_dict()
    return prob

# ==============================
# PRÓXIMO SLOT MULTIPLO DE FREQ_MIN
# ==============================
def proximo_slot_5min(base_dt=None):
    if base_dt is None:
        base_dt = datetime.now()
    base_dt = base_dt.replace(second=0, microsecond=0)
    minuto = (base_dt.minute // FREQ_MIN) * FREQ_MIN + FREQ_MIN
    hora = base_dt.hour
    dia = base_dt.date()
    if minuto >= 60:
        minuto -= 60
        hora += 1
        if hora >= 24:
            hora = 0
            dia += timedelta(days=1)
    return datetime(dia.year, dia.month, dia.day, hora, minuto)

# ==============================
# REGISTRAR PREVISÃO
# ==============================
def registrar_previsao_google(aba_prev, previsao_label, texto_prev, confianca_final, horario_alvo):
    try:
        linha = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 texto_prev,
                 round(confianca_final, 4),
                 horario_alvo]
        aba_prev.append_row(linha)
    except Exception as e:
        escrever_log(f"❌ Erro ao gravar previsão no Sheets: {e}")

# ==============================
# FUNÇÃO PRINCIPAL
# ==============================
def executar_robo():
    escrever_log("🤖 Robô híbrido iniciado...")

    # conectar
    try:
        aba_dados, aba_prev = conectar_google_sheets()
    except Exception:
        return

    # laço principal
    modelo = None
    historico_resultados = []

    while True:
        try:
            df_raw = carregar_dados_google(aba_dados)
            if df_raw is None or df_raw.empty:
                escrever_log("⚠️ Nenhum dado na aba 'Dados'. Aguardando próximo ciclo.")
                time.sleep(FREQ_MIN * 60)
                continue

            df = preparar_dados(df_raw)
            if df is None or df.empty:
                escrever_log("❌ Nenhum dado válido após preparação. Aguardando próximo ciclo.")
                time.sleep(FREQ_MIN * 60)
                continue

            # features fixas (ordem importante)
            feature_cols = ["hora", "minuto", "dia_semana"] + [f"lag_{i}" for i in range(1, NUM_LAGS + 1)]

            # montar X e y para treino (preencher NaN das lags com 0 para fit)
            X = df[feature_cols].fillna(0)
            y = df["resultado"]

            # treinar/re-treinar modelo
            modelo = RandomForestClassifier(n_estimators=200, random_state=42)
            modelo.fit(X, y)
            escrever_log(f"🔄 Modelo treinado com {len(X)} linhas (lags preenchidos com 0 para fit).")

            # calcular probabilidade por horário
            prob_horario = calcular_probabilidade_horario(df)

            # pegar último exemplo (pode ter NaNs nos lags)
            ultimo = df[feature_cols].iloc[-1:].copy()
            # contar lags disponíveis (não-NaN) no último exemplo
            lags_cols = [f"lag_{i}" for i in range(1, NUM_LAGS + 1)]
            lags_disponiveis = int(ultimo[lags_cols].notna().sum(axis=1).iloc[0])

            if lags_disponiveis == 0:
                escrever_log("⚠️ Último registro não tem lags disponíveis; previsão será mais fraca.")

            # preencher NaN de lags com 0 apenas para predict (mantemos contagem para ajustar confiança)
            ultimo.fillna(0, inplace=True)

            # prever
            pred = modelo.predict(ultimo)[0]
            probas = modelo.predict_proba(ultimo)[0] if hasattr(modelo, "predict_proba") else None
            if probas is None:
                confianca_modelo = 0.5
            else:
                confianca_modelo = float(max(probas))

            # ajustar confiança do modelo de acordo com lags disponíveis
            peso_lags = (lags_disponiveis / NUM_LAGS) if NUM_LAGS > 0 else 1.0
            # peso de confiança do modelo entre 0.5 (poucos lags) e 1.0 (todos lags)
            fator_conf_modelo = 0.5 + 0.5 * peso_lags
            confianca_modelo_ajustada = confianca_modelo * fator_conf_modelo

            # confiança histórica do horário de próximo slot
            prox = proximo_slot_5min()
            slot_str = prox.strftime("%H:%M")
            confianca_horario = prob_horario.get(slot_str, 0.5)

            # combinar (média simples entre modelo ajustado e histórico)
            confianca_final = (confianca_modelo_ajustada + confianca_horario) / 2

            rotulo = "ACERTO (verde)" if int(pred) == 1 else "ERRO (vermelho)"
            texto_prev = f"{rotulo} - {round(confianca_final*100,2)}% (modelo: {round(confianca_modelo_ajustada*100,2)}%, horário: {round(confianca_horario*100,2)}%)"
            horario_alvo = prox.strftime("%Y-%m-%d %H:%M:%S")

            # registrar se acima do limite
            if confianca_final >= LIMITE_CONFIANCA:
                alerta = "🚨" if confianca_final >= ALERTA_CONFIANCA else ""
                escrever_log(f"{alerta} ➡️ Previsão registrada: alvo {horario_alvo} | {texto_prev}")
                registrar_previsao_google(aba_prev, rotulo, texto_prev, confianca_final, horario_alvo)

                # atualizar histórico (apenas memória)
                historico_resultados.append(1 if int(pred) == 1 else 0)
                if len(historico_resultados) > TAMANHO_TAXA_MOVEL:
                    historico_resultados.pop(0)
            else:
                escrever_log(f"⚠️ Previsão ignorada (baixa confiança): {texto_prev}")

            # aguardar próximo slot
            agora = datetime.now()
            proximo = proximo_slot_5min(agora)
            sleep_seg = (proximo - agora).total_seconds()
            if sleep_seg <= 0:
                # se o cálculo der zero/negativo (por pouco), espera FREQ_MIN minutos
                sleep_seg = FREQ_MIN * 60
            time.sleep(sleep_seg)

        except Exception as e:
            escrever_log(f"❌ Erro no loop principal: {e}")
            time.sleep(10)

if __name__ == "__main__":
    executar_robo()
