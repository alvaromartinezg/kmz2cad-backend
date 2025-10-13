#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KMZ → DXF (WGS84, con fondo satelital 50%) → DWG (si hay ODA) + PDF
Estrategia: NO reproyectar el raster. Todo en EPSG:4326 (grados) para calce perfecto en CAD.
Compatibilidad con ezdxf 1.4.x (sin kwargs en add_image).
"""

import os, zipfile, math, shutil, subprocess, xml.etree.ElementTree as ET
import requests
from PIL import Image
import matplotlib.pyplot as plt

import ezdxf

# -------- Entradas / Salidas --------
KMZ_NAME = "Exportado.kmz"

DXF_OUT  = "exportado_wgs84.dxf"
DWG_OUT  = "exportado_wgs84.dwg"       # solo si ODA/Teigha está instalado
PRJ_OUT  = "exportado_wgs84.prj"
PDF_OUT  = "plano_wgs84.pdf"

PNG_W84  = "satellite_wgs84.png"       # raster descargado (WGS84)
PGW_W84  = "satellite_wgs84.pgw"       # worldfile en grados

ARCGIS_EXPORT = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"

# -------- KML NS --------
NS = {"kml":"http://www.opengis.net/kml/2.2"}
for p,u in NS.items():
    ET.register_namespace("" if p=="kml" else p, u)

# ==================== Utilidades KML ====================
def parse_coords(text):
    out=[]
    if not text: return out
    for tok in text.strip().split():
        parts = tok.split(",")
        if len(parts)>=2:
            lon = float(parts[0]); lat = float(parts[1])
            out.append((lon, lat))
    return out

def read_kml_roots_from_kmz(kmz_path):
    roots=[]
    with zipfile.ZipFile(kmz_path, "r") as zf:
        names=[n for n in zf.namelist() if n.lower().endswith(".kml")]
        if not names:
            raise FileNotFoundError("El KMZ no contiene ningún .kml")
        for n in names:
            data = zf.read(n)
            roots.append(ET.fromstring(data))
    return roots

def read_geometries_wgs84(kmz_path):
    lines=[]; polys=[]
    roots = read_kml_roots_from_kmz(kmz_path)
    for root in roots:
        for ls in root.findall(".//kml:LineString", NS):
            ce = ls.find("kml:coordinates", NS)
            if ce is None: continue
            pts = parse_coords(ce.text)
            if len(pts)>=2: lines.append(pts)
        for lr in root.findall(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing", NS):
            ce = lr.find("kml:coordinates", NS)
            if ce is None: continue
            ring = parse_coords(ce.text)
            if len(ring)>=3:
                if ring[0]!=ring[-1]: ring = ring + [ring[0]]
                polys.append(ring)
    return lines, polys

# ==================== BBoxes (en WGS84) ====================
def bbox_wgs84(lons, lats, margin_ratio=0.02):
    minlon, minlat = min(lons), min(lats)
    maxlon, maxlat = max(lons), max(lats)
    dx = (maxlon - minlon) * margin_ratio
    dy = (maxlat - minlat) * margin_ratio
    return (minlon - dx, minlat - dy, maxlon + dx, maxlat + dy)

# ==================== Imagen satelital (WGS84) ====================
def fetch_esri_world_imagery_png(bbox_wgs84, out_png, size=2048):
    minx, miny, maxx, maxy = bbox_wgs84
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{size},{size}",
        "format": "png",
        "f": "image",
        "dpi": "96",
    }
    r = requests.get(ARCGIS_EXPORT, params=params, timeout=60)
    r.raise_for_status()
    # Forzar RGB (evita sorpresas con escala de grises/paleta)
    im = Image.open(BytesIO(r.content)) if hasattr(r, "content") else None
    if im:
        im = im.convert("RGB")
        im.save(out_png)
    else:
        # Fallback si el servidor entrega directamente bytes PNG legibles
        with open(out_png, "wb") as f:
            f.write(r.content)

def write_pgw_wgs84(png_path, bbox_wgs84, pgw_path):
    # Worldfile en grados, sin rotación, con coord. del centro del píxel superior-izq.
    with Image.open(png_path) as im:
        w, h = im.size
    minlon, minlat, maxlon, maxlat = bbox_wgs84
    A = (maxlon - minlon) / w         # tamaño pixel X en grados
    E = (minlat - maxlat) / h         # negativo (porque el origen de imagen es arriba)
    C = minlon + A * 0.5              # lon del centro del píxel [0,0]
    F = maxlat + E * 0.5              # lat del centro del píxel [0,0]
    with open(pgw_path, "w", newline="\n") as f:
        f.write(f"{A:.10f}\n")    # A
        f.write("0.0\n")          # D
        f.write("0.0\n")          # B
        f.write(f"{E:.10f}\n")    # E (negativo)
        f.write(f"{C:.8f}\n")     # C
        f.write(f"{F:.8f}\n")     # F
    return (minlon, minlat, maxlon, maxlat)  # bbox WGS84

# ==================== Transparencia DXF (best-effort) ====================
def encode_dxf_transparency(alpha_0_255):
    # DXF: value = 0x02000000 | (alpha << 24); 0=opaco, 255=transparente
    alpha = max(0, min(255, int(alpha_0_255)))
    return (alpha << 24) | 0x02000000

def apply_layer_transparency_if_supported(doc, layer_name, alpha_frac=0.5):
    try:
        layer = doc.layers.get(layer_name)
        setattr(layer.dxf, "transparency", encode_dxf_transparency(int(255 * alpha_frac)))
    except Exception:
        pass

# ==================== DXF (todo en WGS84: grados) ====================
def make_dxf_wgs84(lines_ll, polys_ll, dxf_path, png_path, bbox_wgs84):
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()

    if "SATELLITE" not in doc.layers: doc.layers.add("SATELLITE")
    if "LINES"     not in doc.layers: doc.layers.add("LINES", color=1)      # rojo
    if "POLYGONS"  not in doc.layers: doc.layers.add("POLYGONS", color=3)   # verde

    # 50% transparencia si el visor lo soporta
    apply_layer_transparency_if_supported(doc, "SATELLITE", 0.5)

    # Insertar imagen (coordenadas en grados)
    minlon, minlat, maxlon, maxlat = bbox_wgs84
    width_deg  = max(0.0, maxlon - minlon)
    height_deg = max(0.0, maxlat - minlat)

    if os.path.exists(png_path) and width_deg>0 and height_deg>0:
        try:
            with Image.open(png_path) as _im:
                px_w, px_h = _im.size
        except Exception:
            px_w, px_h = (1, 1)
        imgdef = doc.add_image_def(
            filename=os.path.abspath(png_path),
            size_in_pixel=(px_w, px_h)
        )
        # ezdxf 1.4.x → argumentos POSICIONALES: (image_def, insert, size)
        img = msp.add_image(imgdef, (minlon, minlat), (width_deg, height_deg))
        img.dxf.layer = "SATELLITE"

    # Dibujar vectores
    for pts in lines_ll:
        if len(pts)==2:
            msp.add_line(pts[0], pts[1], dxfattribs={"layer":"LINES"})
        else:
            msp.add_lwpolyline(pts, dxfattribs={"layer":"LINES"})
    for ring in polys_ll:
        if len(ring)>=3:
            msp.add_lwpolyline(ring, dxfattribs={"layer":"POLYGONS", "closed": True})

    # Nota EPSG
    msp.add_mtext("CRS: WGS84 (EPSG:4326)", dxfattribs={"layer":"POLYGONS"}).set_location((minlon, minlat))

    doc.saveas(dxf_path)

# ==================== PRJ (EPSG:4326) ====================
def write_prj_wgs84(prj_path):
    # Intentar con pyproj para WKT; si no está, usar fallback mínimo
    try:
        from pyproj import CRS
        wkt = CRS.from_epsg(4326).to_wkt()
    except Exception:
        wkt = (
            'GEOGCS["WGS 84",'
            'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        )
    with open(prj_path, "w", encoding="utf-8") as f:
        f.write(wkt)

# ==================== DWG (ODA/Teigha) ====================
def try_convert_dxf_to_dwg(dxf_path, dwg_path):
    candidates = [
        "ODAFileConverter", "TeighaFileConverter",
        r"C:\Program Files\ODA\ODAFileConverter\ODAFileConverter.exe",
        r"C:\Program Files\ODA\Teigha File Converter\TeighaFileConverter.exe",
        r"/usr/bin/ODAFileConverter", r"/usr/local/bin/ODAFileConverter",
        r"/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter",
    ]
    exe = None
    for c in candidates:
        if shutil.which(c) or os.path.exists(c):
            exe = shutil.which(c) or c
            break
    if not exe:
        print("[INFO] No se encontró ODA/Teigha File Converter. Se entrega DXF.")
        return False

    out_dir = os.path.dirname(os.path.abspath(dwg_path)) or "."
    in_dir  = os.path.dirname(os.path.abspath(dxf_path)) or "."
    print(f"[INFO] Convirtiendo DXF→DWG con: {exe}")
    cmd = [exe, in_dir, out_dir, "ACAD2018", "1", "1", ""]
    try:
        subprocess.run(cmd, check=True)
        maybe = os.path.join(out_dir, os.path.splitext(os.path.basename(dxf_path))[0] + ".dwg")
        if os.path.exists(maybe):
            if os.path.abspath(maybe) != os.path.abspath(dwg_path):
                shutil.copy2(maybe, dwg_path)
            print(f"[OK] Generado {dwg_path}")
            return True
    except Exception as e:
        print(f"[WARN] ODA converter falló: {e}")
    return False

# ==================== PDF ====================
def render_pdf_wgs84(lines_ll, polys_ll, bbox_wgs84, png_path, pdf_path, title_text):
    minlon, minlat, maxlon, maxlat = bbox_wgs84
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    ax = fig.add_subplot(111)

    if os.path.exists(png_path):
        img = Image.open(png_path)
        ax.imshow(img, extent=[minlon, maxlon, minlat, maxlat], origin="upper", alpha=0.5)

    for pts in lines_ll:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, linewidth=1.2)
    for ring in polys_ll:
        xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
        ax.plot(xs, ys, linewidth=1.0)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([minlon, maxlon]); ax.set_ylim([minlat, maxlat])
    ax.set_xlabel("Longitud (°)"); ax.set_ylabel("Latitud (°)")
    ax.set_title(title_text)
    ax.grid(True, linestyle="--", linewidth=0.3)

    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)

# ==================== MAIN ====================
from io import BytesIO

def main():
    if not os.path.exists(KMZ_NAME):
        print(f"[ERROR] No encuentro {KMZ_NAME} en esta carpeta.")
        return

    print("[INFO] Leyendo geometrías del KMZ (WGS84)…")
    lines_ll, polys_ll = read_geometries_wgs84(KMZ_NAME)
    if not lines_ll and not polys_ll:
        print("[ERROR] El KMZ no contiene LineString ni Polygon utilizables.")
        return

    # BBox en WGS84 (margen pequeño para menos distorsión)
    lons = [lon for g in (lines_ll + polys_ll) for lon,lat in g]
    lats = [lat for g in (lines_ll + polys_ll) for lon,lat in g]
    bbox84 = bbox_wgs84(lons, lats, margin_ratio=0.02)

    print("[INFO] Descargando imagen satelital (WGS84)…")
    fetch_esri_world_imagery_png(bbox84, PNG_W84, size=2048)
    print(f"[OK] Imagen: {PNG_W84}")

    print("[INFO] Escribiendo worldfile (PGW, WGS84)…")
    img_bbox_ll = write_pgw_wgs84(PNG_W84, bbox84, PGW_W84)
    print(f"[OK] Worldfile: {PGW_W84}")

    print("[INFO] Creando DXF (todo en WGS84)…")
    make_dxf_wgs84(lines_ll, polys_ll, DXF_OUT, PNG_W84, img_bbox_ll)
    print(f"[OK] DXF: {DXF_OUT}")

    print("[INFO] Escribiendo PRJ (EPSG:4326)…")
    write_prj_wgs84(PRJ_OUT)
    print(f"[OK] PRJ: {PRJ_OUT}")

    print("[INFO] Intentando convertir a DWG (si ODA/Teigha está instalado)…")
    converted = try_convert_dxf_to_dwg(DXF_OUT, DWG_OUT)
    if converted:
        print(f"[OK] DWG generado: {DWG_OUT}")
    else:
        print("[AVISO] No se generó DWG automáticamente. Se entrega DXF.")

    print("[INFO] Renderizando PDF (WGS84)…")
    render_pdf_wgs84(lines_ll, polys_ll, img_bbox_ll, PNG_W84, PDF_OUT, "Plano (WGS84 EPSG:4326)")
    print(f"[OK] PDF: {PDF_OUT}")

    print("[OK] Proceso finalizado.")

if __name__ == "__main__":
    main()
