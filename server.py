from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from robo_core import RoboHibrido


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


