import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import os

from dotenv import load_dotenv
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from sklearn.ensemble import RandomForestClassifier


# ==============================
# CONFIGURAÇÕES
# ==============================
# Carrega variáveis de ambiente do arquivo .env (se existir)
load_dotenv()

SHEET_ID = os.getenv("SHEET_ID", "114XbIrVhnToZlgHWZd8v9_L9Tvp_UlIiSw9P-f0OzFo")
SHEET_NAME_DADOS = os.getenv("SHEET_NAME_DADOS", "Dados")
SHEET_NAME_PREVISOES = os.getenv("SHEET_NAME_PREVISOES", "Previsões")
CREDENCIAIS_JSON = os.getenv("CREDENCIAIS_JSON", "credenciais.json")

NUM_LAGS = int(os.getenv("NUM_LAGS", "12"))
LIMITE_CONFIANCA = float(os.getenv("LIMITE_CONFIANCA", "0.70"))
ALERTA_CONFIANCA = float(os.getenv("ALERTA_CONFIANCA", "0.90"))
FREQ_MIN = int(os.getenv("FREQ_MIN", "5"))
TAMANHO_TAXA_MOVEL = int(os.getenv("TAMANHO_TAXA_MOVEL", "24"))


# ==============================
# UTILIDADES / LOG
# ==============================
def escrever_log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# ==============================
# CONEXÃO COM GOOGLE SHEETS
# ==============================
def conectar_google_sheets() -> Tuple[gspread.Worksheet, gspread.Worksheet]:
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(CREDENCIAIS_JSON, scopes=scopes)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID)
    aba_dados = sheet.worksheet(SHEET_NAME_DADOS)
    aba_prev = sheet.worksheet(SHEET_NAME_PREVISOES)
    return aba_dados, aba_prev


# ==============================
# LEITURA E PREPARAÇÃO DOS DADOS
# ==============================
def carregar_dados_google(aba_dados: gspread.Worksheet) -> pd.DataFrame:
    try:
        registros = aba_dados.get_all_records()
        df = pd.DataFrame(registros)
        if df.empty:
            return df
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        escrever_log(f"❌ Erro ao carregar dados do Google Sheets: {e}")
        return pd.DataFrame()


def _achar_coluna(df: pd.DataFrame, candidatos) -> Optional[str]:
    cols = list(df.columns)
    for cand in candidatos:
        for c in cols:
            if c.strip().lower() == cand.strip().lower():
                return c
    for c in cols:
        low = c.strip().lower()
        for cand in candidatos:
            if cand.strip().lower() in low or low in cand.strip().lower():
                return c
    return None


def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    col_data = _achar_coluna(df, ["data", "Data", "DATA"])
    col_sucesso = _achar_coluna(df, ["Horário Sucesso", "Horario Sucesso", "horário sucesso", "sucesso"])
    col_falha = _achar_coluna(df, ["Horário Falha", "Horario Falha", "horário falha", "falha"])

    if not col_data:
        escrever_log("❌ Coluna 'data' não encontrada (aguarda nome parecido com 'data').")
        return pd.DataFrame()

    df.rename(columns={col_data: "data"}, inplace=True)
    if col_sucesso:
        df.rename(columns={col_sucesso: "Horário Sucesso"}, inplace=True)
    if col_falha:
        df.rename(columns={col_falha: "Horário Falha"}, inplace=True)

    df.loc[:, "data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
    antes = len(df)
    df = df.dropna(subset=["data"])
    if len(df) < antes:
        escrever_log(f"⚠️ {antes-len(df)} linhas removidas por 'data' inválida.")

    def achar_resultado(row):
        if "Horário Sucesso" in row.index and pd.notna(row.get("Horário Sucesso")) and str(row.get("Horário Sucesso")).strip() != "":
            return 1
        if "Horário Falha" in row.index and pd.notna(row.get("Horário Falha")) and str(row.get("Horário Falha")).strip() != "":
            return 0
        return None

    df.loc[:, "resultado"] = df.apply(achar_resultado, axis=1)

    def parse_horario(row):
        hora_val = None
        if pd.notna(row.get("Horário Sucesso")) and str(row.get("Horário Sucesso")).strip() != "":
            hora_val = str(row.get("Horário Sucesso")).strip()
        elif pd.notna(row.get("Horário Falha")) and str(row.get("Horário Falha")).strip() != "":
            hora_val = str(row.get("Horário Falha")).strip()
        if hora_val is None:
            return pd.NaT
        try:
            return pd.to_datetime(f"{row['data'].strftime('%Y-%m-%d')} {hora_val}", errors="coerce")
        except Exception:
            return pd.NaT

    df.loc[:, "datetime_horario"] = df.apply(parse_horario, axis=1)
    antes = len(df)
    df = df.dropna(subset=["datetime_horario"])
    if len(df) < antes:
        escrever_log(f"⚠️ {antes-len(df)} linhas removidas por 'horário' inválido.")

    df.loc[:, "hora"] = df["datetime_horario"].dt.hour
    df.loc[:, "minuto"] = df["datetime_horario"].dt.minute
    df.loc[:, "dia_semana"] = df["datetime_horario"].dt.dayofweek

    for lag in range(1, NUM_LAGS + 1):
        df.loc[:, f"lag_{lag}"] = df["resultado"].shift(lag)

    df.loc[:, "slot"] = (
        df["hora"].astype(int).astype(str).str.zfill(2)
        + ":"
        + df["minuto"].astype(int).astype(str).str.zfill(2)
    )

    return df


# ==============================
# ESTATÍSTICAS POR HORÁRIO
# ==============================
def calcular_probabilidade_horario(df: pd.DataFrame) -> Dict[str, float]:
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
def proximo_slot_5min(base_dt: Optional[datetime] = None) -> datetime:
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
def registrar_previsao_google(aba_prev: gspread.Worksheet, texto_prev: str, confianca_final: float, horario_alvo: str) -> None:
    try:
        linha = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            texto_prev,
            round(confianca_final, 4),
            horario_alvo,
        ]
        aba_prev.append_row(linha)
    except Exception as e:
        escrever_log(f"❌ Erro ao gravar previsão no Sheets: {e}")


class RoboHibrido:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()
        self._is_running: bool = False
        self._is_paused: bool = False
        self._ultimo_horario_alvo: Optional[str] = None
        self._ultima_msg: Optional[str] = None

    # ---------- Status ----------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self._is_running,
                "paused": self._is_paused,
                "ultimo_horario_alvo": self._ultimo_horario_alvo,
                "ultima_msg": self._ultima_msg,
            }

    # ---------- Controle ----------
    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._pause_event.clear()
            self._is_paused = False
            self._is_running = True
            self._thread = threading.Thread(target=self._loop, name="RoboHibridoThread", daemon=True)
            self._thread.start()

    def pause(self) -> None:
        with self._lock:
            if not self._is_running:
                return
            self._pause_event.set()
            self._is_paused = True
            self._ultima_msg = "Execução pausada."
            escrever_log("⏸️ Execução pausada.")

    def resume(self) -> None:
        with self._lock:
            if not self._is_running:
                return
            self._pause_event.clear()
            self._is_paused = False
            self._ultima_msg = "Execução retomada."
            escrever_log("▶️ Execução retomada.")

    def stop(self) -> None:
        with self._lock:
            if not self._is_running:
                return
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=15)
        with self._lock:
            self._is_running = False
            self._is_paused = False
            self._ultima_msg = "Execução finalizada."
            escrever_log("🛑 Execução finalizada.")

    # ---------- Laço principal ----------
    def _loop(self) -> None:
        escrever_log("🤖 Robô híbrido iniciado...")

        try:
            aba_dados, aba_prev = conectar_google_sheets()
        except gspread.SpreadsheetNotFound:
            escrever_log("❌ Planilha não encontrada (verifique SHEET_ID e compartilhamento da service account).")
            with self._lock:
                self._is_running = False
            return
        except Exception as e:
            escrever_log(f"❌ Erro ao conectar Google Sheets: {e}")
            with self._lock:
                self._is_running = False
            return

        modelo: Optional[RandomForestClassifier] = None
        historico_resultados = []

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.5)
                continue

            try:
                df_raw = carregar_dados_google(aba_dados)
                if df_raw is None or df_raw.empty:
                    escrever_log("⚠️ Nenhum dado na aba 'Dados'. Aguardando próximo ciclo.")
                    self._sleep_ate_proximo_slot()
                    continue

                df = preparar_dados(df_raw)
                if df is None or df.empty:
                    escrever_log("❌ Nenhum dado válido após preparação. Aguardando próximo ciclo.")
                    self._sleep_ate_proximo_slot()
                    continue

                feature_cols = ["hora", "minuto", "dia_semana"] + [f"lag_{i}" for i in range(1, NUM_LAGS + 1)]
                X = df[feature_cols].fillna(0)
                y = df["resultado"]

                modelo = RandomForestClassifier(n_estimators=200, random_state=42)
                modelo.fit(X, y)
                escrever_log(f"🔄 Modelo treinado com {len(X)} linhas (lags preenchidos com 0 para fit).")

                prob_horario = calcular_probabilidade_horario(df)

                ultimo = df[feature_cols].iloc[-1:].copy()
                lags_cols = [f"lag_{i}" for i in range(1, NUM_LAGS + 1)]
                lags_disponiveis = int(ultimo[lags_cols].notna().sum(axis=1).iloc[0])
                if lags_disponiveis == 0:
                    escrever_log("⚠️ Último registro não tem lags disponíveis; previsão será mais fraca.")
                ultimo.fillna(0, inplace=True)

                pred = modelo.predict(ultimo)[0]
                probas = modelo.predict_proba(ultimo)[0] if hasattr(modelo, "predict_proba") else None
                confianca_modelo = float(max(probas)) if probas is not None else 0.5

                peso_lags = (lags_disponiveis / NUM_LAGS) if NUM_LAGS > 0 else 1.0
                fator_conf_modelo = 0.5 + 0.5 * peso_lags
                confianca_modelo_ajustada = confianca_modelo * fator_conf_modelo

                prox = proximo_slot_5min()
                slot_str = prox.strftime("%H:%M")
                confianca_horario = prob_horario.get(slot_str, 0.5)

                confianca_final = (confianca_modelo_ajustada + confianca_horario) / 2

                rotulo = "ACERTO (verde)" if int(pred) == 1 else "ERRO (vermelho)"
                texto_prev = (
                    f"{rotulo} - {round(confianca_final*100,2)}% (modelo: {round(confianca_modelo_ajustada*100,2)}%, "
                    f"horário: {round(confianca_horario*100,2)}%)"
                )
                horario_alvo = prox.strftime("%Y-%m-%d %H:%M:%S")

                with self._lock:
                    self._ultimo_horario_alvo = horario_alvo

                if confianca_final >= LIMITE_CONFIANCA:
                    alerta = "🚨" if confianca_final >= ALERTA_CONFIANCA else ""
                    escrever_log(f"{alerta} ➡️ Previsão registrada: alvo {horario_alvo} | {texto_prev}")
                    registrar_previsao_google(aba_prev, texto_prev, confianca_final, horario_alvo)

                    historico_resultados.append(1 if int(pred) == 1 else 0)
                    if len(historico_resultados) > TAMANHO_TAXA_MOVEL:
                        historico_resultados.pop(0)
                else:
                    escrever_log(f"⚠️ Previsão ignorada (baixa confiança): {texto_prev}")

                self._sleep_ate_proximo_slot()

            except Exception as e:
                escrever_log(f"❌ Erro no loop principal: {e}")
                time.sleep(2)

        with self._lock:
            self._is_running = False

    def _sleep_ate_proximo_slot(self) -> None:
        agora = datetime.now()
        proximo = proximo_slot_5min(agora)
        sleep_seg = (proximo - agora).total_seconds()
        if sleep_seg <= 0:
            sleep_seg = FREQ_MIN * 60
        # Dorme em passos de 1s respeitando pause/stop
        restante = int(sleep_seg)
        while restante > 0 and not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.5)
                continue
            time.sleep(1)
            restante -= 1


__all__ = [
    "RoboHibrido",
    "escrever_log",
    "SHEET_ID",
    "SHEET_NAME_DADOS",
    "SHEET_NAME_PREVISOES",
    "CREDENCIAIS_JSON",
]


