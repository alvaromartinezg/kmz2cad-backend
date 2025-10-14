# app/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil, os, zipfile, traceback

# IMPORTA TU SCRIPT DE CONVERSIÓN
# Debe existir app/kmz2cad.py con una función main() como la que pegaste
from app import kmz2cad

# ─────────────────────────────
# Config
# ─────────────────────────────
ALLOWED_ORIGINS = [
    "https://alvaromartinezg.github.io",
    "https://alvaromartinezg.github.io/transmission.department",
    "http://localhost:5500",
    "http://localhost:5173",
    "http://127.0.0.1:5500",
]

# Nombre que tu script espera
IN_KMZ_NAME = "Exportado.kmz"
TPL_NAME    = os.getenv("TEMPLATE_DXF_IN", "PLANTILLA.dxf")

# ─────────────────────────────
# App
# ─────────────────────────────
app = FastAPI(title="KMZ → CAD/PDF API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],  # <- para leer filename en el front
)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# ─────────────────────────────
# Helpers
# ─────────────────────────────
def _ensure_workspace() -> Path:
    work = Path("/tmp/conv")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    return work

def _copy_template_to(work: Path) -> Path:
    """
    Copia PLANTILLA.dxf desde el paquete (/app/app/PLANTILLA.dxf) al workspace.
    Si usas otra ubicación, ajusta este resolver.
    """
    # PLANTILLA junto al main.py dentro del paquete "app"
    tpl_src = Path(__file__).with_name("PLANTILLA.dxf")
    if not tpl_src.exists():
        # alterno: tal vez la pusiste en /app directamente
        alt = Path("/app/PLANTILLA.dxf")
        if alt.exists():
            tpl_src = alt
    if not tpl_src.exists():
        raise HTTPException(500, f"No existe plantilla DXF en {tpl_src}")

    tpl_dst = work / TPL_NAME
    shutil.copy2(tpl_src, tpl_dst)
    return tpl_dst

def _pick_media_type(name: str) -> str:
    n = name.lower()
    if n.endswith(".pdf"):
        return "application/pdf"
    if n.endswith(".zip"):
        return "application/zip"
    if n.endswith(".dwg"):
        return "application/octet-stream"
    if n.endswith(".dxf"):
        return "application/dxf"
    return "application/octet-stream"

# ─────────────────────────────
# Endpoint principal
# ─────────────────────────────
@app.post("/process")
async def process(
    file: UploadFile | None = File(None),
    test_kmz: UploadFile | None = File(None),
    output: str = Form("both"),  # "pdf" | "dwg" | "both"
):
    """
    Convierte KMZ/KML → (PDF/DXF/DWG). Usa tu kmz2cad.main() que genera:
      - PLANTILLA_FINAL.dxf
      - I-01.pdf
      - exportado_wgs84.dwg (si ODA disponible)
    """
    # 1) Workspace en /tmp
    work = _ensure_workspace()

    # 2) Archivo de entrada (acepta "file" o "test_kmz")
    up = file or test_kmz
    if not up:
        raise HTTPException(400, "Falta archivo (field: 'file' o 'test_kmz').")

    in_path = work / IN_KMZ_NAME
    contents = await up.read()
    if not contents:
        raise HTTPException(400, "Archivo vacío.")
    in_path.write_bytes(contents)

    # 3) Copiar plantilla al workspace
    _copy_template_to(work)

    # 4) Ejecutar tu main() dentro de /tmp/conv
    cwd = os.getcwd()
    os.chdir(work)
    try:
        print("[INFO] Leyendo KMZ con colores…")
        kmz2cad.main()  # tu función main() ya busca nombres relativos
    except Exception:
        tb = traceback.format_exc()
        print("[ERROR] Falló kmz2cad.main():\n", tb)
        os.chdir(cwd)
        raise HTTPException(500, f"Error interno al procesar.\n{tb}")
    finally:
        os.chdir(cwd)

    # 5) Recolectar salidas
    pdf = work / "I-01.pdf"
    dxf = work / "PLANTILLA_FINAL.dxf"
    dwg = work / "exportado_wgs84.dwg"  # si ODA/Teigha disponible

    outs: list[Path] = []
    if output in ("pdf", "both") and pdf.exists():
        outs.append(pdf)

    # Si piden DWG y no hubo ODA, devolvemos DXF para no dejar sin salida.
    if output in ("dwg", "both"):
        if dwg.exists():
            outs.append(dwg)
        elif dxf.exists():
            outs.append(dxf)

    # Si solo pidieron "pdf" pero no salió PDF, intenta al menos DXF
    if output == "pdf" and not pdf.exists() and dxf.exists():
        outs.append(dxf)

    if not outs:
        raise HTTPException(500, "No se generaron salidas.")

    # 6) Empaquetar/retornar
    if len(outs) == 1:
        p = outs[0]
        out_name = p.name
        media_type = _pick_media_type(out_name)
        bytes_out = p.read_bytes()
        print(f"[OK] Devolviendo: {out_name}")
        return Response(
            content=bytes_out,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
        )

    # ZIP si hay múltiples (ej. both)
    zip_path = work / "resultado.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in outs:
            zf.write(p, arcname=p.name)
    bytes_out = zip_path.read_bytes()
    out_name = zip_path.name
    print(f"[OK] Devolviendo ZIP con {len(outs)} archivos")
    return Response(
        content=bytes_out,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )
