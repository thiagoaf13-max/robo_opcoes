import threading
import time
import os
from datetime import datetime
from typing import Optional, Dict, Any

from dotenv import load_dotenv
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from uuid import uuid4

# Reuso de utilitários do V1
from robo_core import (
    conectar_google_sheets,
    carregar_dados_google,
    preparar_dados,
    calcular_probabilidade_horario,
    proximo_slot_5min,
    registrar_previsao_google,
    escrever_log,
    NUM_LAGS,
    LIMITE_CONFIANCA,
    ALERTA_CONFIANCA,
    FREQ_MIN,
    TAMANHO_TAXA_MOVEL,
)


load_dotenv()

# Parâmetros V2
SHEET_ID_V2 = os.getenv("SHEET_ID_V2", os.getenv("SHEET_ID", "1iP4Im3GPL21i_xOIVawaca2DjzJU-O9yOjBoptn-m5Y"))
SHEET_NAME_DADOS_V2 = os.getenv("SHEET_NAME_DADOS_V2", os.getenv("SHEET_NAME_DADOS", "Dados"))
SHEET_NAME_PREVISOES_V2 = os.getenv("SHEET_NAME_PREVISOES_V2", os.getenv("SHEET_NAME_PREVISOES", "Previsões"))
V2_WINDOW_ROWS = int(os.getenv("V2_WINDOW_ROWS", "0"))  # 0 = usar todo histórico
V2_VAL_FRACTION = float(os.getenv("V2_VAL_FRACTION", "0.2"))
V2_CALIBRACAO_TIPO = os.getenv("V2_CALIBRACAO_TIPO", os.getenv("CALIBRACAO_TIPO", "platt")).strip().lower()
V2_THRESHOLD_METRIC = os.getenv("V2_THRESHOLD_METRIC", "f1").strip().lower()  # f1|accuracy


class RoboHibridoV2:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()
        self._is_running: bool = False
        self._is_paused: bool = False

        self._ultimo_horario_alvo: Optional[str] = None
        self._ultima_msg: Optional[str] = None

        # Exposição de status
        self._ultima_prev_texto: Optional[str] = None
        self._ultima_prev_confianca_modelo: Optional[float] = None
        self._ultima_prev_confianca_horario: Optional[float] = None
        self._ultima_prev_confianca_final: Optional[float] = None
        self._ultima_prev_label: Optional[str] = None
        self._ultima_prev_slot: Optional[str] = None
        self._registradas_count: int = 0
        self._ignoradas_count: int = 0
        self._taxa_movel_atual: Optional[float] = None
        self._historico_previsoes: list[Dict[str, Any]] = []

        # Modelo/threshold
        self._modelo = None
        self._threshold_decisao: float = 0.5

        # Telemetria simples
        self._telemetria: Dict[str, Any] = {}

        # Placar e pendências de validação
        self._acertos_count: int = 0
        self._erros_count: int = 0
        self._previsoes_pendentes: list[Dict[str, Any]] = []

    # ---------- Status ----------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self._is_running,
                "paused": self._is_paused,
                "ultimo_horario_alvo": self._ultimo_horario_alvo,
                "ultima_msg": self._ultima_msg,
                "ultima_previsao": {
                    "texto": self._ultima_prev_texto,
                    "confianca_modelo": self._ultima_prev_confianca_modelo,
                    "confianca_horario": self._ultima_prev_confianca_horario,
                    "confianca_final": self._ultima_prev_confianca_final,
                    "label": self._ultima_prev_label,
                    "slot": self._ultima_prev_slot,
                },
                "contadores": {
                    "registradas": self._registradas_count,
                    "ignoradas": self._ignoradas_count,
                },
                "taxa_movel": self._taxa_movel_atual,
                "historico_previsoes": self._historico_previsoes[-5:],
                "threshold": self._threshold_decisao,
                "calibracao": V2_CALIBRACAO_TIPO,
                "telemetria": self._telemetria,
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
            self._thread = threading.Thread(target=self._loop, name="RoboHibridoV2Thread", daemon=True)
            self._thread.start()

    def pause(self) -> None:
        with self._lock:
            if not self._is_running:
                return
            self._pause_event.set()
            self._is_paused = True
            self._ultima_msg = "Execução pausada."
            escrever_log("⏸️ (V2) Execução pausada.")

    def resume(self) -> None:
        with self._lock:
            if not self._is_running:
                return
            self._pause_event.clear()
            self._is_paused = False
            self._ultima_msg = "Execução retomada."
            escrever_log("▶️ (V2) Execução retomada.")

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
            escrever_log("🛑 (V2) Execução finalizada.")

    # ---------- Laço principal ----------
    def _loop(self) -> None:
        escrever_log("🤖 (V2) Robô híbrido V2 iniciado...")

        try:
            aba_dados, aba_prev = conectar_google_sheets(
                sheet_id=SHEET_ID_V2,
                sheet_name_dados=SHEET_NAME_DADOS_V2,
                sheet_name_prev=SHEET_NAME_PREVISOES_V2,
            )
        except Exception as e:
            escrever_log(f"❌ (V2) Erro ao conectar Google Sheets: {e}")
            with self._lock:
                self._is_running = False
            return

        historico_resultados = []

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.5)
                continue

            try:
                df_raw = carregar_dados_google(aba_dados)
                if df_raw is None or df_raw.empty:
                    escrever_log("⚠️ (V2) Nenhum dado na aba 'Dados'. Aguardando próximo ciclo.")
                    self._sleep_ate_proximo_slot()
                    continue

                df = preparar_dados(df_raw)
                if df is None or df.empty:
                    escrever_log("❌ (V2) Nenhum dado válido após preparação. Aguardando próximo ciclo.")
                    self._sleep_ate_proximo_slot()
                    continue

                feature_cols = ["hora", "minuto", "dia_semana"] + [f"lag_{i}" for i in range(1, NUM_LAGS + 1)]
                X_all = df[feature_cols].fillna(0)
                y_all = df["resultado"].astype(int)

                # Janela deslizante
                if V2_WINDOW_ROWS and V2_WINDOW_ROWS > 0 and len(X_all) > V2_WINDOW_ROWS:
                    X_all = X_all.tail(V2_WINDOW_ROWS)
                    y_all = y_all.tail(V2_WINDOW_ROWS)

                # Split temporal simples
                val_size = max(50, int(len(X_all) * V2_VAL_FRACTION))
                if len(X_all) <= val_size:
                    val_size = max(1, len(X_all) // 5)
                X_train = X_all.iloc[:-val_size]
                y_train = y_all.iloc[:-val_size]
                X_val = X_all.iloc[-val_size:]
                y_val = y_all.iloc[-val_size:]

                base_model = LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                )

                # Construção do modelo com calibração opcional e fallback seguro
                try:
                    if V2_CALIBRACAO_TIPO in {"platt", "isotonic"}:
                        metodo = "sigmoid" if V2_CALIBRACAO_TIPO == "platt" else "isotonic"
                        modelo = CalibratedClassifierCV(base_model, method=metodo, cv=3)
                    else:
                        modelo = base_model
                    modelo.fit(X_train, y_train)
                except Exception as e:
                    escrever_log(f"⚠️ (V2) Falha na calibração ({V2_CALIBRACAO_TIPO}). Usando modelo sem calibrar. Erro: {e}")
                    modelo = base_model
                    modelo.fit(X_train, y_train)

                self._modelo = modelo

                # Otimizar limiar sobre a CONFIANÇA COMBINADA (modelo ajustado + histórico)
                # Probabilidade do modelo no conjunto de validação
                val_proba_model = (
                    modelo.predict_proba(X_val)[:, 1]
                    if hasattr(modelo, "predict_proba") else np.full(len(X_val), 0.5)
                )

                # Fator de lags por linha de validação
                lags_cols = [f"lag_{i}" for i in range(1, NUM_LAGS + 1)]
                try:
                    lags_counts_val = (
                        preparar_dados(df_raw)  # garantir colunas caso df tenha mudado
                        if not set(lags_cols).issubset(df.columns) else df
                    )
                except Exception:
                    lags_counts_val = df
                lags_counts_val = lags_counts_val.loc[X_all.index, lags_cols].notna().sum(axis=1).reindex(X_val.index).fillna(0).astype(int)
                peso_lags_val = np.where(NUM_LAGS > 0, lags_counts_val.values / float(NUM_LAGS), 1.0)
                fator_conf_modelo_val = 0.5 + 0.5 * peso_lags_val
                val_proba_model_ajustada = val_proba_model * fator_conf_modelo_val

                # Confiança histórica por slot das amostras de validação
                prob_horario = calcular_probabilidade_horario(df)
                slots_val = df.loc[X_all.index, "slot"].reindex(X_val.index)
                conf_horario_val = np.array([prob_horario.get(str(s), 0.5) for s in slots_val])

                # Confiança combinada
                val_conf_combinada = (val_proba_model_ajustada + conf_horario_val) / 2.0

                # Busca de limiar
                best_thr = 0.5
                best_score = -1.0
                for thr in np.linspace(0.1, 0.9, 81):
                    preds = (val_conf_combinada >= thr).astype(int)
                    if V2_THRESHOLD_METRIC == "accuracy":
                        score = accuracy_score(y_val, preds)
                    else:
                        score = f1_score(y_val, preds, zero_division=0)
                    if score > best_score:
                        best_score = score
                        best_thr = float(thr)
                self._threshold_decisao = best_thr
                escrever_log(
                    f"🎯 (V2) Limiar otimizado (conf. combinada): {best_thr:.2f} (métrica={V2_THRESHOLD_METRIC}, score={best_score:.4f})"
                )

                # Probabilidade histórica por slot
                prob_horario = calcular_probabilidade_horario(df)

                # Última linha para previsão
                # Usa lags da última linha, porém ajusta hora/minuto/dia_semana para o PRÓXIMO slot
                ultimo = X_all.iloc[[-1]].copy()
                prox = proximo_slot_5min()
                ultimo.loc[:, "hora"] = prox.hour
                ultimo.loc[:, "minuto"] = prox.minute
                ultimo.loc[:, "dia_semana"] = prox.weekday()

                lags_cols = [f"lag_{i}" for i in range(1, NUM_LAGS + 1)]
                lags_disponiveis = int(ultimo[lags_cols].notna().sum(axis=1).iloc[0]) if set(lags_cols).issubset(ultimo.columns) else 0
                ultimo.fillna(0, inplace=True)

                probas = modelo.predict_proba(ultimo)[0] if hasattr(modelo, "predict_proba") else [0.5, 0.5]
                proba_1 = float(probas[1])

                # Ajuste por lags como no V1
                peso_lags = (lags_disponiveis / NUM_LAGS) if NUM_LAGS > 0 else 1.0
                fator_conf_modelo = 0.5 + 0.5 * peso_lags
                confianca_modelo_ajustada = proba_1 * fator_conf_modelo

                slot_str = prox.strftime("%H:%M")
                confianca_horario = prob_horario.get(slot_str, 0.5)
                confianca_final = (confianca_modelo_ajustada + confianca_horario) / 2

                # Decisão usando limiar otimizado NA CONFIANÇA COMBINADA
                pred_label = 1 if confianca_final >= self._threshold_decisao else 0
                rotulo = "ACERTO (verde)" if pred_label == 1 else "ERRO (vermelho)"
                texto_prev = (
                    f"{rotulo} - {round(confianca_final*100,2)}% (modelo: {round(confianca_modelo_ajustada*100,2)}%, "
                    f"horário: {round(confianca_horario*100,2)}%)"
                )
                horario_alvo = prox.strftime("%Y-%m-%d %H:%M:%S")

                with self._lock:
                    self._ultimo_horario_alvo = horario_alvo
                    self._ultima_prev_texto = texto_prev
                    self._ultima_prev_confianca_modelo = round(confianca_modelo_ajustada, 4)
                    self._ultima_prev_confianca_horario = round(confianca_horario, 4)
                    self._ultima_prev_confianca_final = round(confianca_final, 4)
                    self._ultima_prev_label = rotulo
                    self._ultima_prev_slot = slot_str
                    self._historico_previsoes.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "slot": slot_str,
                        "label": rotulo,
                        "confianca_final": round(confianca_final, 4),
                        "confianca_modelo": round(confianca_modelo_ajustada, 4),
                        "confianca_horario": round(confianca_horario, 4),
                        "texto": texto_prev,
                    })
                    if len(self._historico_previsoes) > 50:
                        self._historico_previsoes.pop(0)

                if confianca_final >= LIMITE_CONFIANCA:
                    alerta = "🚨" if confianca_final >= ALERTA_CONFIANCA else ""
                    escrever_log(f"{alerta} ➡️ (V2) Previsão registrada: alvo {horario_alvo} | {texto_prev}")
                    prediction_id = f"V2-{int(time.time()*1000)}-{uuid4().hex[:6]}"
                    registrar_previsao_google(aba_prev, prediction_id, rotulo, texto_prev, confianca_final, horario_alvo)
                    historico_resultados.append(1 if pred_label == 1 else 0)
                    if len(historico_resultados) > TAMANHO_TAXA_MOVEL:
                        historico_resultados.pop(0)
                    with self._lock:
                        self._registradas_count += 1
                        self._taxa_movel_atual = round(sum(historico_resultados) / len(historico_resultados), 4) if historico_resultados else None
                        # adiciona previsão pendente de validação
                        try:
                            alvo_dt = datetime.strptime(horario_alvo, "%Y-%m-%d %H:%M:%S")
                            self._previsoes_pendentes.append({
                                "alvo_dt": alvo_dt,
                                "alvo_key": alvo_dt.strftime("%Y-%m-%d %H:%M"),
                                "pred_label": int(pred_label),
                                "prediction_id": prediction_id,
                            })
                        except Exception:
                            pass
                else:
                    escrever_log(f"⚠️ (V2) Previsão ignorada (baixa confiança): {texto_prev}")
                    with self._lock:
                        self._ignoradas_count += 1

                self._sleep_ate_proximo_slot()

                # Atualiza telemetria básica
                try:
                    tmp = y_all.dropna()
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
                            "window_rows": int(len(X_all)),
                            "proximo_slot": slot_str,
                            "limiar": round(self._threshold_decisao, 4),
                        }
                except Exception:
                    pass

                # Atualiza placar comparando previsões pendentes com resultados da planilha
                try:
                    if not df.empty and self._previsoes_pendentes:
                        keys = df.dropna(subset=["datetime_horario", "resultado"]).copy()
                        keys.loc[:, "_key"] = keys["datetime_horario"].dt.strftime("%Y-%m-%d %H:%M")
                        key_to_result = {k: int(v) for k, v in zip(keys["_key"].tolist(), keys["resultado"].tolist())}
                        ainda_pendentes: list[Dict[str, Any]] = []
                        for p in self._previsoes_pendentes:
                            res = key_to_result.get(p.get("alvo_key"))
                            if res is None:
                                ainda_pendentes.append(p)
                            else:
                                if int(p.get("pred_label", 0)) == int(res):
                                    self._acertos_count += 1
                                    atualizar_resultado_previsao(aba_prev, p.get("prediction_id", ""), "ACERTO")
                                else:
                                    self._erros_count += 1
                                    atualizar_resultado_previsao(aba_prev, p.get("prediction_id", ""), "ERRO")
                        self._previsoes_pendentes = ainda_pendentes
                except Exception:
                    pass

            except Exception as e:
                escrever_log(f"❌ (V2) Erro no loop principal: {e}")
                time.sleep(2)

        with self._lock:
            self._is_running = False

    def _sleep_ate_proximo_slot(self) -> None:
        # Reuso de lógica do V1: aguarda até o próximo múltiplo de FREQ_MIN
        agora = datetime.now()
        prox = proximo_slot_5min(agora)
        sleep_seg = (prox - agora).total_seconds()
        if sleep_seg <= 0:
            sleep_seg = FREQ_MIN * 60
        restante = int(sleep_seg)
        while restante > 0 and not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.5)
                continue
            time.sleep(1)
            restante -= 1


__all__ = ["RoboHibridoV2"]




