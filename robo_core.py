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
from uuid import uuid4
from sklearn.calibration import CalibratedClassifierCV


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

# Pesos das contribuições (modelo, histórico por slot, histórico por dia-da-semana+slot, padrão de sequência)
WEIGHT_MODEL = float(os.getenv("WEIGHT_MODEL", "0.15"))
WEIGHT_SLOT = float(os.getenv("WEIGHT_SLOT", "0.4"))
WEIGHT_WEEKDAY_SLOT = float(os.getenv("WEIGHT_WEEKDAY_SLOT", "0.4"))
WEIGHT_PATTERN = float(os.getenv("WEIGHT_PATTERN", "0.05"))

# Calibração e estratégia de treino
CALIBRACAO_TIPO = os.getenv("CALIBRACAO_TIPO", "none").strip().lower()  # none|platt|isotonic
TREINO_CADA_CICLOS = int(os.getenv("TREINO_CADA_CICLOS", "1"))  # 1 mantém comportamento atual
TREINO_SOMENTE_SE_DADOS_MUDARAM = os.getenv("TREINO_SOMENTE_SE_DADOS_MUDARAM", "true").strip().lower() in {"1", "true", "yes", "y"}


# ==============================
# UTILIDADES / LOG
# ==============================
def escrever_log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# ==============================
# CONEXÃO COM GOOGLE SHEETS
# ==============================
def conectar_google_sheets(
    sheet_id: Optional[str] = None,
    sheet_name_dados: Optional[str] = None,
    sheet_name_prev: Optional[str] = None,
) -> Tuple[gspread.Worksheet, gspread.Worksheet]:
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(CREDENCIAIS_JSON, scopes=scopes)
    client = gspread.authorize(creds)
    effective_sheet_id = sheet_id or SHEET_ID
    effective_name_dados = sheet_name_dados or SHEET_NAME_DADOS
    effective_name_prev = sheet_name_prev or SHEET_NAME_PREVISOES

    sheet = client.open_by_key(effective_sheet_id)
    aba_dados = sheet.worksheet(effective_name_dados)
    aba_prev = sheet.worksheet(effective_name_prev)
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


def calcular_probabilidade_semana_slot(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {}
    tmp = df.dropna(subset=["resultado", "dia_semana", "slot"]).copy()
    if tmp.empty:
        return {}
    tmp.loc[:, "semana_slot_key"] = tmp["dia_semana"].astype(int).astype(str) + "-" + tmp["slot"].astype(str)
    prob = tmp.groupby("semana_slot_key")["resultado"].mean().to_dict()
    return prob


def calcular_probabilidade_padrao_sequencia(df: pd.DataFrame, ordem: int) -> Dict[tuple, float]:
    if df is None or df.empty or ordem <= 0:
        return {}
    cols = [f"lag_{i}" for i in range(1, ordem + 1)]
    if not set(cols).issubset(df.columns):
        return {}
    tmp = df.dropna(subset=cols + ["resultado"]).copy()
    if tmp.empty:
        return {}
    # cria tupla de lags como chave
    def mk_key(row):
        try:
            return tuple(int(row[c]) for c in cols)
        except Exception:
            return None
    tmp.loc[:, "padrao_key"] = tmp.apply(mk_key, axis=1)
    tmp = tmp.dropna(subset=["padrao_key"])  # remove onde key None
    if tmp.empty:
        return {}
    prob = tmp.groupby("padrao_key")["resultado"].mean().to_dict()
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
def registrar_previsao_google(
    aba_prev: gspread.Worksheet,
    prediction_id: str,
    rotulo: str,
    texto_prev: str,
    confianca_final: float,
    horario_alvo: str,
) -> None:
    try:
        # Colunas: ID | criado_em | texto | confianca | horario_alvo | label | resultado | resolvido_em
        linha = [
            prediction_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            texto_prev,
            round(confianca_final, 4),
            horario_alvo,
            rotulo,
            "PENDENTE",
            "",
        ]
        aba_prev.append_row(linha)
    except Exception as e:
        escrever_log(f"❌ Erro ao gravar previsão no Sheets: {e}")


def atualizar_resultado_previsao(
    aba_prev: gspread.Worksheet,
    prediction_id: str,
    resultado: str,
) -> None:
    """Atualiza a linha da previsão (pela coluna ID) com o resultado e timestamp de resolução."""
    try:
        cel = aba_prev.find(prediction_id)
        if cel is None:
            return
        row = cel.row
        # Resultado na coluna 7, resolvido_em na coluna 8 (conforme ordem acima)
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        aba_prev.update_cell(row, 7, resultado)
        aba_prev.update_cell(row, 8, agora)
    except Exception as e:
        escrever_log(f"⚠️ Erro ao atualizar resultado no Sheets (ID={prediction_id}): {e}")


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

        # Detalhes da última previsão
        self._ultima_prev_texto: Optional[str] = None
        self._ultima_prev_confianca_modelo: Optional[float] = None
        self._ultima_prev_confianca_horario: Optional[float] = None
        self._ultima_prev_confianca_final: Optional[float] = None
        self._ultima_prev_label: Optional[str] = None
        self._ultima_prev_slot: Optional[str] = None

        # Contadores e métricas simples
        self._registradas_count: int = 0
        self._ignoradas_count: int = 0
        self._taxa_movel_atual: Optional[float] = None

        # Histórico de previsões (mantém até 50 para consulta)
        self._historico_previsoes: list[Dict[str, Any]] = []
        # Controle de treino inteligente
        self._ciclos_desde_ultimo_treino: int = 0
        self._ultimo_hash_dados: Optional[int] = None

        # Telemetria simples
        self._telemetria: Dict[str, Any] = {}

        # Placar e pendências de validação
        self._acertos_count: int = 0
        self._erros_count: int = 0
        self._previsoes_pendentes: list[Dict[str, Any]] = []
        # Controle para emitir nova previsão somente com dados novos
        self._ultimo_hash_previsto: Optional[int] = None

    # ---------- Status ----------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self._is_running,
                "paused": self._is_paused,
                "ultimo_horario_alvo": self._ultimo_horario_alvo,
                "ultima_msg": self._ultima_msg,
                # Última previsão detalhada
                "ultima_previsao": {
                    "texto": self._ultima_prev_texto,
                    "confianca_modelo": self._ultima_prev_confianca_modelo,
                    "confianca_horario": self._ultima_prev_confianca_horario,
                    "confianca_final": self._ultima_prev_confianca_final,
                    "label": self._ultima_prev_label,
                    "slot": self._ultima_prev_slot,
                },
                # Contadores
                "contadores": {
                    "registradas": self._registradas_count,
                    "ignoradas": self._ignoradas_count,
                },
                # Métrica móvel (sobre últimas N registradas)
                "taxa_movel": self._taxa_movel_atual,
                # Últimas 5 previsões
                "historico_previsoes": self._historico_previsoes[-5:],
                # Telemetria
                "telemetria": self._telemetria,
                # Placar
                "placar": {
                    "acertos": self._acertos_count,
                    "erros": self._erros_count,
                    "pendentes": len(self._previsoes_pendentes),
                },
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

                # V1: só gera nova previsão quando houver dados novos na planilha
                try:
                    hash_part_a = tuple(df["datetime_horario"].tail(50).astype(str).tolist())
                    hash_part_b = tuple(df["resultado"].tail(50).fillna(-1).astype(int).tolist())
                    dados_hash_atual = hash((hash_part_a, hash_part_b, len(df)))
                except Exception:
                    dados_hash_atual = hash(len(df))
                if self._ultimo_hash_previsto is not None and self._ultimo_hash_previsto == dados_hash_atual:
                    self._ultima_msg = "Standby: aguardando novos dados na aba Dados."
                    escrever_log("⏸️ (V1) Standby: sem novos dados; aguardando.")
                    time.sleep(10)
                    continue

                feature_cols = ["hora", "minuto", "dia_semana"] + [f"lag_{i}" for i in range(1, NUM_LAGS + 1)]
                y = df["resultado"]
                # V1 simples: não treina modelo; usa apenas pilares (slot, dia+slot e padrão)

                prob_horario = calcular_probabilidade_horario(df)
                prob_semana_slot = calcular_probabilidade_semana_slot(df)
                prob_padrao = calcular_probabilidade_padrao_sequencia(df, ordem=min(5, NUM_LAGS))

                # Usa lags da última linha, porém ajusta hora/minuto/dia_semana para o PRÓXIMO slot
                ultimo = df[feature_cols].iloc[-1:].copy()
                prox = proximo_slot_5min()
                ultimo.loc[:, "hora"] = prox.hour
                ultimo.loc[:, "minuto"] = prox.minute
                # pandas .dt.dayofweek usa Monday=0, Python datetime.weekday() também
                ultimo.loc[:, "dia_semana"] = prox.weekday()

                lags_cols = [f"lag_{i}" for i in range(1, NUM_LAGS + 1)]
                lags_disponiveis = int(ultimo[lags_cols].notna().sum(axis=1).iloc[0])
                if lags_disponiveis == 0:
                    escrever_log("⚠️ Último registro não tem lags disponíveis; previsão será mais fraca.")
                ultimo.fillna(0, inplace=True)

                slot_str = prox.strftime("%H:%M")
                confianca_horario = prob_horario.get(slot_str, 0.5)
                semana_slot_key = f"{prox.weekday()}-{slot_str}"
                confianca_semana_slot = prob_semana_slot.get(semana_slot_key, 0.5)
                # Padrão de sequência: usa lags atuais como chave
                lags_tuple = None
                try:
                    lags_vals = [int(df[f"lag_{i}"].iloc[-1]) for i in range(1, min(5, NUM_LAGS) + 1)]
                    if all(v in (0, 1) for v in lags_vals):
                        lags_tuple = tuple(lags_vals)
                except Exception:
                    lags_tuple = None
                confianca_padrao = prob_padrao.get(lags_tuple, 0.5) if lags_tuple is not None else 0.5

                # V1 simples: decisão sem modelo (média simples entre dia+slot, slot e padrão)
                confianca_modelo_ajustada = None
                confianca_final = (confianca_semana_slot + confianca_horario + confianca_padrao) / 3.0
                pred = 1 if confianca_final >= 0.5 else 0

                rotulo = "ACERTO (verde)" if int(pred) == 1 else "ERRO (vermelho)"
                texto_prev = (
                    f"{rotulo} - {round(confianca_final*100,2)}% (slot: {round(confianca_horario*100,2)}%, "
                    f"semana+slot: {round(confianca_semana_slot*100,2)}%, padrão: {round(confianca_padrao*100,2)}%)"
                )
                horario_alvo = prox.strftime("%Y-%m-%d %H:%M:%S")

                with self._lock:
                    self._ultimo_horario_alvo = horario_alvo
                    self._ultimo_hash_previsto = dados_hash_atual
                    # Atualiza última previsão (independente de registrar ou não)
                    self._ultima_prev_texto = texto_prev
                    self._ultima_prev_confianca_modelo = None
                    self._ultima_prev_confianca_horario = round(confianca_horario, 4)
                    # Novas contribuições
                    try:
                        self._ultima_prev_confianca_semana_slot = round(confianca_semana_slot, 4)
                        self._ultima_prev_confianca_padrao = round(confianca_padrao, 4)
                    except Exception:
                        pass
                    self._ultima_prev_confianca_final = round(confianca_final, 4)
                    self._ultima_prev_label = rotulo
                    self._ultima_prev_slot = slot_str

                    # Registra no histórico (limita a 50 itens)
                    self._historico_previsoes.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "slot": slot_str,
                        "label": rotulo,
                        "confianca_final": round(confianca_final, 4),
                        "confianca_modelo": None,
                        "confianca_horario": round(confianca_horario, 4),
                        "texto": texto_prev,
                    })
                    if len(self._historico_previsoes) > 50:
                        self._historico_previsoes.pop(0)

                if confianca_final >= LIMITE_CONFIANCA:
                    alerta = "🚨" if confianca_final >= ALERTA_CONFIANCA else ""
                    escrever_log(f"{alerta} ➡️ Previsão registrada: alvo {horario_alvo} | {texto_prev}")
                    prediction_id = f"V1-{int(time.time()*1000)}-{uuid4().hex[:6]}"
                    registrar_previsao_google(aba_prev, prediction_id, rotulo, texto_prev, confianca_final, horario_alvo)

                    historico_resultados.append(1 if int(pred) == 1 else 0)
                    if len(historico_resultados) > TAMANHO_TAXA_MOVEL:
                        historico_resultados.pop(0)
                    with self._lock:
                        self._registradas_count += 1
                        # taxa móvel é média do histórico das registradas
                        self._taxa_movel_atual = round(sum(historico_resultados) / len(historico_resultados), 4) if historico_resultados else None
                        # adiciona previsão pendente de validação
                        try:
                            alvo_dt = pd.to_datetime(horario_alvo)
                            self._previsoes_pendentes.append({
                                "alvo_dt": alvo_dt,
                                "alvo_key": alvo_dt.strftime("%Y-%m-%d %H:%M"),
                                "pred_label": int(pred),
                                "prediction_id": prediction_id,
                            })
                        except Exception:
                            pass
                else:
                    escrever_log(f"⚠️ Previsão ignorada (baixa confiança): {texto_prev}")
                    with self._lock:
                        self._ignoradas_count += 1

                # Atualiza telemetria básica
                try:
                    tmp = y.dropna()
                    y_total = int(len(tmp))
                    y_pos = int(tmp.sum()) if y_total > 0 else 0
                    y_neg = int(y_total - y_pos)
                    p_pos = round(y_pos / y_total, 4) if y_total > 0 else None
                    with self._lock:
                        self._telemetria = {
                            "y_total": y_total,
                            "y_pos": y_pos,
                            "y_neg": y_neg,
                            "p_pos": p_pos,
                            "window_rows": int(len(X)),
                            "proximo_slot": slot_str,
                        }
                except Exception:
                    pass

                # Atualiza placar comparando previsões pendentes com resultados da planilha
                try:
                    if not df.empty and self._previsoes_pendentes:
                        df_ok = df.dropna(subset=["datetime_horario", "resultado"]).copy()
                        if not df_ok.empty:
                            # Tolerância de tempo: até FREQ_MIN minutos do alvo
                            tol_sec = max(60, FREQ_MIN * 60)
                            ainda_pendentes: list[Dict[str, Any]] = []
                            for p in self._previsoes_pendentes:
                                alvo_dt = p.get("alvo_dt")
                                if alvo_dt is None:
                                    ainda_pendentes.append(p)
                                    continue
                                # Calcula diferença absoluta para todas as linhas
                                deltas = (df_ok["datetime_horario"] - alvo_dt).abs()
                                idx_min = deltas.idxmin()
                                delta_min = deltas.loc[idx_min]
                                if pd.notna(delta_min) and float(delta_min.total_seconds()) <= tol_sec:
                                    res = int(df_ok.loc[idx_min, "resultado"]) if pd.notna(df_ok.loc[idx_min, "resultado"]) else None
                                    if res is None:
                                        ainda_pendentes.append(p)
                                        continue
                                    if int(p.get("pred_label", 0)) == res:
                                        self._acertos_count += 1
                                        atualizar_resultado_previsao(aba_prev, p.get("prediction_id", ""), "ACERTO")
                                        escrever_log(f"✅ Placar: alvo {alvo_dt} casado com {df_ok.loc[idx_min, 'datetime_horario']} (Δ={int(delta_min.total_seconds())}s) => ACERTO")
                                    else:
                                        self._erros_count += 1
                                        atualizar_resultado_previsao(aba_prev, p.get("prediction_id", ""), "ERRO")
                                        escrever_log(f"❌ Placar: alvo {alvo_dt} casado com {df_ok.loc[idx_min, 'datetime_horario']} (Δ={int(delta_min.total_seconds())}s) => ERRO")
                                else:
                                    # Ainda não há linha correspondente no intervalo de tolerância
                                    ainda_pendentes.append(p)
                            self._previsoes_pendentes = ainda_pendentes
                except Exception:
                    pass

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


