# app/main.py
import io, os, tempfile, shutil, zipfile, contextlib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List

os.environ.setdefault("MPLBACKEND", "Agg")

app = FastAPI(title="KMZ→CAD Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/")
def root():
    return {"ok": True, "msg": "KMZ→CAD up", "endpoints": ["/health","/process"]}

@app.get("/health")
def health():
    return {"status": "ok"}

def _collect_outputs(workdir: str) -> List[str]:
    wanted = ["exportado_wgs84.dxf", "exportado_wgs84.prj",
              "plano_wgs84.pdf", "exportado_wgs84.dwg"]
    return [os.path.join(workdir, w) for w in wanted if os.path.exists(os.path.join(workdir, w))]

@app.post("/process")
def process_kmz(file: UploadFile = File(...)):
    # ⬇️ Import aquí para no romper el arranque si falla una dependencia
    try:
        import importlib
        kmz2cad = importlib.import_module("app.kmz2cad")
    except Exception as e:
        # Devolvemos el error real para diagnosticar
        raise HTTPException(status_code=500, detail=f"Import error: {e}")

    if not file.filename.lower().endswith(".kmz"):
        raise HTTPException(status_code=400, detail="Sube un archivo .kmz")

    content = file.file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="KMZ vacío")

    tmpdir = tempfile.mkdtemp(prefix="kmz2cad_")
    try:
        kmz_path = os.path.join(tmpdir, "Exportado.kmz")
        with open(kmz_path, "wb") as f:
            f.write(content)

        cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            kmz2cad.main()
        finally:
            os.chdir(cwd)

        outputs = _collect_outputs(tmpdir)
        if not outputs:
            raise HTTPException(status_code=500, detail="No se generaron salidas.")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for abs_path in outputs:
                zf.write(abs_path, arcname=os.path.basename(abs_path))
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="kmz2cad_outputs.zip"'}
        )
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        with contextlib.suppress(Exception):
            shutil.rmtree(tmpdir)
