import pandas as pd
import time
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# ==============================
# CONFIGURAÇÕES
# ==============================
SHEET_ID = "114XbIrVhnToZlgHWZd8v9_L9Tvp_UlIiSw9P-f0OzFo"
SHEET_NAME_DADOS = "Dados"
SHEET_NAME_PREVISOES = "Previsões"
CREDENCIAIS_JSON = "credenciais.json"

NUM_LAGS = 12
LIMITE_CONFIANCA = 0.7
ALERTA_CONFIANCA = 0.9
FREQ_MIN = 5  # Intervalos de 5 minutos
TAMANHO_TAXA_MOVEL = 24  # últimos 24 eventos

# ==============================
# FUNÇÕES AUXILIARES
# ==============================
def escrever_log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def conectar_google_sheets():
    escopos = ["https://www.googleapis.com/auth/spreadsheets",
               "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(CREDENCIAIS_JSON, scopes=escopos)
    cliente = gspread.authorize(creds)
    planilha = cliente.open_by_key(SHEET_ID)
    aba_dados = planilha.worksheet(SHEET_NAME_DADOS)
    aba_prev = planilha.worksheet(SHEET_NAME_PREVISOES)
    return aba_dados, aba_prev

def carregar_dados_google(aba_dados):
    try:
        dados = aba_dados.get_all_records()
        df = pd.DataFrame(dados)
        df.columns = [col.strip() for col in df.columns]
        df.dropna(how="all", inplace=True)
        return df
    except Exception as e:
        escrever_log(f"❌ Erro ao carregar dados: {e}")
        return pd.DataFrame()

def preparar_dados(df):
    if df.empty:
        return df

    df['data'] = pd.to_datetime(df['data'], dayfirst=True, errors='coerce')
    df.dropna(subset=['data'], inplace=True)

    df['resultado'] = df.apply(
        lambda row: 1 if pd.notna(row.get('Horário Sucesso')) else (0 if pd.notna(row.get('Horário Falha')) else None),
        axis=1
    )

    df['datetime_horario'] = df.apply(
        lambda row: (
            pd.to_datetime(f"{row['data'].strftime('%Y-%m-%d')} {str(row['Horário Sucesso']).strip()}", errors='coerce')
            if pd.notna(row.get('Horário Sucesso')) else (
                pd.to_datetime(f"{row['data'].strftime('%Y-%m-%d')} {str(row['Horário Falha']).strip()}", errors='coerce')
                if pd.notna(row.get('Horário Falha')) else None
            )
        ),
        axis=1
    )

    df.dropna(subset=['datetime_horario'], inplace=True)
    df['hora'] = df['datetime_horario'].dt.hour
    df['minuto'] = df['datetime_horario'].dt.minute
    df['dia_semana'] = df['datetime_horario'].dt.dayofweek

    for lag in range(1, NUM_LAGS + 1):
        df[f'lag_{lag}'] = df['resultado'].shift(lag)

    cols_check = ['resultado'] + [f'lag_{i}' for i in range(1, NUM_LAGS + 1)]
    return df.dropna(subset=cols_check)

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

def registrar_previsao_google(aba_prev, modelo, X, df_processado, historico_resultados):
    ultimo_exemplo = X.iloc[-1:]
    previsao = modelo.predict(ultimo_exemplo)[0]

    probas = modelo.predict_proba(ultimo_exemplo)[0] if hasattr(modelo, "predict_proba") else [0.5, 0.5]
    confianca = float(max(probas))

    rotulo_texto = "ACERTO (verde)" if previsao == 1 else "ERRO (vermelho)"
    texto_prev = f"{rotulo_texto} - {round(confianca*100,2)}%"

    if confianca >= LIMITE_CONFIANCA:
        alerta = "🚨" if confianca >= ALERTA_CONFIANCA else ""
        horario_alvo_dt = proximo_slot_5min()
        horario_alvo = horario_alvo_dt.strftime("%Y-%m-%d %H:%M:%S")
        horario_emissao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        escrever_log(f"{alerta} ➡️ Previsão registrada: alvo {horario_alvo} | {texto_prev} (emitida: {horario_emissao})")

        # Registrar previsão na aba
        linha = len(aba_prev.get_all_values()) + 1
        aba_prev.append_row([horario_emissao, texto_prev, round(confianca,4), horario_alvo])

        # Atualizar histórico para cálculo da taxa móvel
        historico_resultados.append(1 if previsao == 1 else 0)
        if len(historico_resultados) > TAMANHO_TAXA_MOVEL:
            historico_resultados.pop(0)

        return int(previsao)
    else:
        escrever_log(f"⚠️ Previsão ignorada (baixa confiança): {texto_prev}")
        return None

# ==============================
# EXECUÇÃO PRINCIPAL
# ==============================
def executar_robo():
    escrever_log("🤖 Robô híbrido iniciado...")

    aba_dados, aba_prev = conectar_google_sheets()
    df = carregar_dados_google(aba_dados)
    if df.empty:
        escrever_log("⚠️ Nenhum dado válido.")
        return

    df_processado = preparar_dados(df)
    if df_processado.empty:
        escrever_log("❌ Nenhum dado válido após preparação.")
        return

    X = df_processado[['hora','minuto','dia_semana'] + [f'lag_{i}' for i in range(1, NUM_LAGS+1)]]
    y = df_processado['resultado']

    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X, y)
    escrever_log("🔄 Modelo treinado com todo o histórico!")

    historico_resultados = []

    # Previsão contínua (sem depender de novos dados)
    while True:
        try:
            registrar_previsao_google(aba_prev, modelo, X, df_processado, historico_resultados)

            agora = datetime.now()
            proximo = proximo_slot_5min(agora)
            sleep_seg = (proximo - agora).total_seconds()
            time.sleep(max(0, sleep_seg))
        except Exception as e:
            escrever_log(f"❌ Erro no loop principal: {e}")
            time.sleep(10)

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    executar_robo()
