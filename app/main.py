# app/main.py (o donde está tu endpoint de conversión)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pathlib import Path
import os, shutil, sys, traceback

from app.kmz2cad import main as kmz_main  # si tu main() está en app/kmz2cad.py

app = FastAPI()

@app.post("/process")
async def convert(file: UploadFile = File(...), output: str = Form("both")):
    work = Path("/tmp/conv")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    # 1) Guardar input en /tmp con el nombre que tu script espera
    in_kmz = work / "Exportado.kmz"
    with in_kmz.open("wb") as f:
        f.write(await file.read())

    # 2) Copiar la plantilla desde /app/app → /tmp
    tpl_src = Path(__file__).with_name("PLANTILLA.dxf")       # /app/app/PLANTILLA.dxf
    tpl_dst = work / "PLANTILLA.dxf"
    if not tpl_src.exists():
        raise HTTPException(500, f"No existe plantilla en {tpl_src}")
    shutil.copy2(tpl_src, tpl_dst)

    # 3) Ejecutar tu main() trabajando en /tmp
    cwd = os.getcwd()
    os.chdir(work)
    try:
        kmz_main()   # tu main() usa nombres relativos, funcionará en /tmp
    except Exception:
        tb = traceback.format_exc()
        os.chdir(cwd)
        raise HTTPException(500, f"Fallo en main():\n{tb}")
    os.chdir(cwd)

    # 4) Devolver salidas (al menos PDF o DXF)
    pdf = work / "I-01.pdf"
    dxf = work / "PLANTILLA_FINAL.dxf"
    dwg = work / "exportado_wgs84.dwg"

    outs = [p for p in (pdf, dxf, dwg) if p.exists()]
    if not outs:
        raise HTTPException(500, "No se generaron salidas.")

    # si piden ambos, arma ZIP; si no, devuelve lo que pidió
    # (deja tu lógica actual aquí)
    ...
