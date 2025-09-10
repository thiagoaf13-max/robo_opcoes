from pathlib import Path
from datetime import datetime
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from robo_core import (
    RoboHibrido,
    conectar_google_sheets,
    registrar_previsao_google,
    SHEET_ID,
    SHEET_NAME_DADOS,
    SHEET_NAME_PREVISOES,
    NUM_LAGS,
    LIMITE_CONFIANCA,
    ALERTA_CONFIANCA,
    FREQ_MIN,
    TAMANHO_TAXA_MOVEL,
)


app = FastAPI(title="Robo Híbrido Controller")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

worker = RoboHibrido()


@app.get("/status")
def status():
    return JSONResponse(worker.status())


@app.post("/start")
def start():
    worker.start()
    return JSONResponse({"ok": True, **worker.status()})


@app.post("/pause")
def pause():
    worker.pause()
    return JSONResponse({"ok": True, **worker.status()})


@app.post("/resume")
def resume():
    worker.resume()
    return JSONResponse({"ok": True, **worker.status()})


@app.post("/stop")
def stop():
    worker.stop()
    return JSONResponse({"ok": True, **worker.status()})


@app.get("/")
def root():
    base = Path(__file__).parent
    index = base / "web" / "index.html"
    return FileResponse(index)


# Endpoint de teste de escrita na planilha
@app.post("/test-write")
def test_write():
    try:
        _, aba_prev = conectar_google_sheets()
        texto_prev = "TESTE API - escrita de verificação"
        horario_alvo = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        registrar_previsao_google(aba_prev, texto_prev, 0.99, horario_alvo)
        return JSONResponse({"ok": True, "mensagem": "Linha de teste gravada em Previsões.", "horario": horario_alvo})
    except Exception as e:
        return JSONResponse({"ok": False, "erro": str(e)}, status_code=500)


# Configurações atuais do robô
@app.get("/config")
def get_config():
    return JSONResponse(
        {
            "SHEET_ID": SHEET_ID,
            "SHEET_NAME_DADOS": SHEET_NAME_DADOS,
            "SHEET_NAME_PREVISOES": SHEET_NAME_PREVISOES,
            "NUM_LAGS": NUM_LAGS,
            "LIMITE_CONFIANCA": LIMITE_CONFIANCA,
            "ALERTA_CONFIANCA": ALERTA_CONFIANCA,
            "FREQ_MIN": FREQ_MIN,
            "TAMANHO_TAXA_MOVEL": TAMANHO_TAXA_MOVEL,
        }
    )


# Logs (tail)
@app.get("/logs")
def get_logs(lines: int = 200):
    try:
        log_path = os.getenv("LOG_PATH", "/var/log/robohibrido/app.log")
        if not os.path.exists(log_path):
            return PlainTextResponse(
                f"Log não encontrado em {log_path}. Se rodando localmente, verifique o terminal.",
                status_code=404,
            )
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.readlines()
        tail = "".join(content[-max(1, min(lines, 2000)):])
        return PlainTextResponse(tail)
    except Exception as e:
        return PlainTextResponse(f"Erro ao ler logs: {e}", status_code=500)

