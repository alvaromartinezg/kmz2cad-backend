# app/main.py
import io, os, tempfile, shutil, zipfile, contextlib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List
import importlib
import pathlib
import sys

# Para que matplotlib no intente usar display
os.environ.setdefault("MPLBACKEND", "Agg")

# Import dinámico del módulo de tu script
# (ruta relativa: app/kmz2cad.py => paquete 'app.kmz2cad')
kmz2cad = importlib.import_module("app.kmz2cad")

app = FastAPI(title="KMZ→CAD Backend", version="1.0.0")

# CORS abierto para pruebas; ajusta a tu dominio en producción
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health")
def health():
    return {"status": "ok"}

def _collect_outputs(workdir: str) -> List[str]:
    """Devuelve rutas absolutas de archivos a empaquetar si existen."""
    wanted = ["exportado_wgs84.dxf", "exportado_wgs84.prj",
              "plano_wgs84.pdf", "exportado_wgs84.dwg"]
    found = []
    for w in wanted:
        p = os.path.join(workdir, w)
        if os.path.exists(p):
            found.append(p)
    if not found:
        raise FileNotFoundError("No se generaron salidas.")
    return found

@app.post("/process")
def process_kmz(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".kmz"):
        raise HTTPException(status_code=400, detail="Sube un archivo .kmz")

    # Límite de tamaño (opcional): 100 MB
    # Puedes cambiarlo; Cloud Run acepta hasta 32MB por defecto en body si no usas streaming detrás de proxy,
    # pero FastAPI en contenedor no limita por sí sola; tú decides aquí.
    content = file.file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="KMZ vacío")

    tmpdir = tempfile.mkdtemp(prefix="kmz2cad_")
    try:
        # Guardamos el KMZ con el nombre esperado por tu script
        kmz_path = os.path.join(tmpdir, "Exportado.kmz")
        with open(kmz_path, "wb") as f:
            f.write(content)

        # Ejecutamos el main() de tu script dentro del tmpdir
        cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            # tu script ya valida y escribe PNG/PGW/DXF/PRJ/DWG/PDF
            kmz2cad.main()
        finally:
            os.chdir(cwd)

        # Empaquetar salidas en un ZIP en memoria
        outputs = _collect_outputs(tmpdir)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for abs_path in outputs:
                zf.write(abs_path, arcname=os.path.basename(abs_path))
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="kmz2cad_outputs.zip"'
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Limpieza
        with contextlib.suppress(Exception):
            shutil.rmtree(tmpdir)
