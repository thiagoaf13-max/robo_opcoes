import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

from robo_core import (
    conectar_google_sheets,
    carregar_dados_google,
    preparar_dados,
    calcular_probabilidade_padrao_sequencia,
    proximo_slot_5min,
    registrar_previsao_google,
    atualizar_resultado_previsao,
    escrever_log,
    NUM_LAGS,
    LIMITE_CONFIANCA,
    ALERTA_CONFIANCA,
    FREQ_MIN,
    TAMANHO_TAXA_MOVEL,
)


class RoboSequencias:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()
        self._is_running: bool = False
        self._is_paused: bool = False

        self._ultimo_horario_alvo: Optional[str] = None
        self._ultima_msg: Optional[str] = None

        # Última previsão
        self._ultima_prev_texto: Optional[str] = None
        self._ultima_prev_confianca_padrao: Optional[float] = None
        self._ultima_prev_confianca_final: Optional[float] = None
        self._ultima_prev_label: Optional[str] = None
        self._ultima_prev_slot: Optional[str] = None

        self._registradas_count: int = 0
        self._ignoradas_count: int = 0
        self._historico_previsoes: list[Dict[str, Any]] = []
        self._taxa_movel_atual: Optional[float] = None

        self._previsoes_pendentes: list[Dict[str, Any]] = []
        self._acertos_count: int = 0
        self._erros_count: int = 0

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self._is_running,
                "paused": self._is_paused,
                "ultimo_horario_alvo": self._ultimo_horario_alvo,
                "ultima_msg": self._ultima_msg,
                "ultima_previsao": {
                    "texto": self._ultima_prev_texto,
                    "confianca_padrao": self._ultima_prev_confianca_padrao,
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
                "placar": {
                    "acertos": self._acertos_count,
                    "erros": self._erros_count,
                    "pendentes": len(self._previsoes_pendentes),
                },
            }

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._pause_event.clear()
            self._is_paused = False
            self._is_running = True
            self._thread = threading.Thread(target=self._loop, name="RoboSequenciasThread", daemon=True)
            self._thread.start()

    def pause(self) -> None:
        with self._lock:
            if not self._is_running:
                return
            self._pause_event.set()
            self._is_paused = True
            self._ultima_msg = "Execução pausada."
            escrever_log("⏸️ (V3) Execução pausada.")

    def resume(self) -> None:
        with self._lock:
            if not self._is_running:
                return
            self._pause_event.clear()
            self._is_paused = False
            self._ultima_msg = "Execução retomada."
            escrever_log("▶️ (V3) Execução retomada.")

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
            escrever_log("🛑 (V3) Execução finalizada.")

    def _loop(self) -> None:
        escrever_log("🤖 (V3) Robô de Sequências iniciado...")
        try:
            aba_dados, aba_prev = conectar_google_sheets()
        except Exception as e:
            escrever_log(f"❌ (V3) Erro ao conectar Google Sheets: {e}")
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
                    escrever_log("⚠️ (V3) Nenhum dado na aba 'Dados'. Aguardando próximo ciclo.")
                    time.sleep(max(5, FREQ_MIN * 60))
                    continue

                df = preparar_dados(df_raw)
                if df is None or df.empty:
                    escrever_log("❌ (V3) Nenhum dado válido após preparação. Aguardando próximo ciclo.")
                    time.sleep(max(5, FREQ_MIN * 60))
                    continue

                prob_padrao = calcular_probabilidade_padrao_sequencia(df, ordem=min(5, NUM_LAGS))
                # Próximo slot apenas para relógio/slot no texto
                prox = proximo_slot_5min()
                slot_str = prox.strftime("%H:%M")

                # lags atuais
                lags_tuple = None
                try:
                    lags_vals = [int(df[f"lag_{i}"].iloc[-1]) for i in range(1, min(5, NUM_LAGS) + 1)]
                    if all(v in (0, 1) for v in lags_vals):
                        lags_tuple = tuple(lags_vals)
                except Exception:
                    lags_tuple = None
                confianca_padrao = prob_padrao.get(lags_tuple, 0.5) if lags_tuple is not None else 0.5

                confianca_final = float(confianca_padrao)
                pred = 1 if confianca_final >= 0.5 else 0
                rotulo = "ACERTO (verde)" if int(pred) == 1 else "ERRO (vermelho)"
                texto_prev = f"{rotulo} - {round(confianca_final*100,2)}% (padrão: {round(confianca_padrao*100,2)}%)"
                horario_alvo = prox.strftime("%Y-%m-%d %H:%M:%S")

                with self._lock:
                    self._ultimo_horario_alvo = horario_alvo
                    self._ultima_prev_texto = texto_prev
                    self._ultima_prev_confianca_padrao = round(confianca_padrao, 4)
                    self._ultima_prev_confianca_final = round(confianca_final, 4)
                    self._ultima_prev_label = rotulo
                    self._ultima_prev_slot = slot_str

                    self._historico_previsoes.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "slot": slot_str,
                        "label": rotulo,
                        "confianca_final": round(confianca_final, 4),
                        "confianca_padrao": round(confianca_padrao, 4),
                        "texto": texto_prev,
                    })
                    if len(self._historico_previsoes) > 50:
                        self._historico_previsoes.pop(0)

                if confianca_final >= LIMITE_CONFIANCA:
                    alerta = "🚨" if confianca_final >= ALERTA_CONFIANCA else ""
                    escrever_log(f"{alerta} ➡️ (V3) Previsão registrada: alvo {horario_alvo} | {texto_prev}")
                    prediction_id = f"V3-{int(time.time()*1000)}"
                    registrar_previsao_google(aba_prev, prediction_id, rotulo, texto_prev, confianca_final, horario_alvo)
                    historico_resultados.append(1 if int(pred) == 1 else 0)
                    if len(historico_resultados) > TAMANHO_TAXA_MOVEL:
                        historico_resultados.pop(0)
                    with self._lock:
                        self._registradas_count += 1
                        self._taxa_movel_atual = round(sum(historico_resultados) / len(historico_resultados), 4) if historico_resultados else None
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
                    escrever_log(f"⚠️ (V3) Previsão ignorada (baixa confiança): {texto_prev}")
                    with self._lock:
                        self._ignoradas_count += 1

                # casar pendentes
                try:
                    if not df.empty and self._previsoes_pendentes:
                        df_ok = df.dropna(subset=["datetime_horario", "resultado"]).copy()
                        if not df_ok.empty:
                            tol_sec = max(60, FREQ_MIN * 60)
                            ainda: list[Dict[str, Any]] = []
                            for p in self._previsoes_pendentes:
                                alvo_dt = p.get("alvo_dt")
                                if alvo_dt is None:
                                    ainda.append(p)
                                    continue
                                deltas = (df_ok["datetime_horario"] - alvo_dt).abs()
                                idx_min = deltas.idxmin()
                                delta_min = deltas.loc[idx_min]
                                if pd.notna(delta_min) and float(delta_min.total_seconds()) <= tol_sec:
                                    res = int(df_ok.loc[idx_min, "resultado"]) if pd.notna(df_ok.loc[idx_min, "resultado"]) else None
                                    if res is None:
                                        ainda.append(p)
                                        continue
                                    if int(p.get("pred_label", 0)) == res:
                                        self._acertos_count += 1
                                        atualizar_resultado_previsao(aba_prev, p.get("prediction_id", ""), "ACERTO")
                                    else:
                                        self._erros_count += 1
                                        atualizar_resultado_previsao(aba_prev, p.get("prediction_id", ""), "ERRO")
                                else:
                                    ainda.append(p)
                            self._previsoes_pendentes = ainda
                except Exception:
                    pass

                time.sleep(max(2, FREQ_MIN * 60))

            except Exception as e:
                escrever_log(f"❌ (V3) Erro no loop principal: {e}")
                time.sleep(2)


__all__ = ["RoboSequencias"]


