#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KMZ → DXF (WGS84) → (opcional) DWG + PDF (UTM)
- PDF en coordenadas UTM (zona detectada automáticamente).
- DXF sigue en WGS84 (EPSG:4326) para calce en CAD.
- Sin fondo satelital. Leyenda (pequeña) y Norte dentro del plano.
- Márgenes internos del 10% alrededor del contenido.
"""

import os, zipfile, math, shutil, subprocess, xml.etree.ElementTree as ET
from io import BytesIO

# Backend sin GUI para matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.ticker import FormatStrFormatter

from PIL import Image
import ezdxf

# ───────── Entradas / Salidas ─────────
KMZ_NAME = "Exportado.kmz"

DXF_OUT  = "exportado_wgs84.dxf"
DWG_OUT  = "exportado_wgs84.dwg"
PRJ_OUT  = "exportado_wgs84.prj"
PDF_OUT  = "plano_wgs84.pdf"

SATELLITE_BACKGROUND = False   # ← ignorado para PDF (no se dibuja)
PNG_W84  = "satellite_wgs84.png"
PGW_W84  = "satellite_wgs84.pgw"

# Símbolo de norte (imagen exacta si existe)
NORTH_PNG = "north_symbol.png"

# ───────── KML NS ─────────
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
            pts = parse_coords(ce.text if ce is not None else "")
            if len(pts)>=2: lines.append(pts)
        for lr in root.findall(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing", NS):
            ce = lr.find("kml:coordinates", NS)
            ring = parse_coords(ce.text if ce is not None else "")
            if len(ring)>=3:
                if ring[0]!=ring[-1]: ring = ring + [ring[0]]
                polys.append(ring)
    return lines, polys

# ==================== BBoxes (WGS84) ====================
def bbox_wgs84(lons, lats, margin_ratio=0.06):
    minlon, minlat = min(lons), min(lats)
    maxlon, maxlat = max(lons), max(lats)
    dx = (maxlon - minlon) * margin_ratio
    dy = (maxlat - minlat) * margin_ratio
    return (minlon - dx, minlat - dy, maxlon + dx, maxlat + dy)

# ==================== Proyección UTM (para PDF) ====================
def get_utm_zone_epsg(lon, lat):
    """Devuelve (epsg, zona_str) para la lat/lon dada."""
    zone = int(math.floor((lon + 180) / 6) + 1)
    south = (lat < 0)
    epsg = 32700 + zone if south else 32600 + zone
    zone_str = f"{zone}{'S' if south else 'N'}"
    return epsg, zone_str

def project_to_utm(lines_ll, polys_ll, bbox_ll):
    """Convierte listas de (lon,lat) a (E,N) en metros usando pyproj."""
    try:
        from pyproj import Transformer
    except Exception as e:
        raise RuntimeError("Se requiere pyproj para dibujar en UTM. Instala con: pip install pyproj") from e

    minlon, minlat, maxlon, maxlat = bbox_ll
    clon = (minlon + maxlon) / 2.0
    clat = (minlat + maxlat) / 2.0
    epsg, zone_str = get_utm_zone_epsg(clon, clat)

    transformer = Transformer.from_crs(4326, epsg, always_xy=True)

    def conv(seq):
        return [transformer.transform(lon, lat) for lon, lat in seq]

    lines_utm = [conv(pts) for pts in lines_ll]
    polys_utm = [conv(ring) for ring in polys_ll]

    # bbox UTM a partir de los datos proyectados
    xs = [x for g in (lines_utm + polys_utm) for x,y in g]
    ys = [y for g in (lines_utm + polys_utm) for x,y in g]
    minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)

    # margen interno del 10 %
    mx = 0.10 * (maxx - minx)
    my = 0.10 * (maxy - miny)
    bbox_utm = (minx - mx, miny - my, maxx + mx, maxy + my)

    return lines_utm, polys_utm, bbox_utm, zone_str

# ==================== DXF (WGS84) ====================
def encode_dxf_transparency(alpha_0_255):
    alpha = max(0, min(255, int(alpha_0_255)))
    return (alpha << 24) | 0x02000000

def apply_layer_transparency_if_supported(doc, layer_name, alpha_frac=0.5):
    try:
        layer = doc.layers.get(layer_name)
        setattr(layer.dxf, "transparency", encode_dxf_transparency(int(255 * alpha_frac)))
    except Exception:
        pass

def make_dxf_wgs84(lines_ll, polys_ll, dxf_path, png_path, bbox_wgs84):
    doc = ezdxf.new("R2018")
    msp = doc.modelspace()

    if "SATELLITE" not in doc.layers: doc.layers.add("SATELLITE")
    if "LINES"     not in doc.layers: doc.layers.add("LINES", color=1)
    if "POLYGONS"  not in doc.layers: doc.layers.add("POLYGONS", color=6)
    if "ANNOTATIONS" not in doc.layers: doc.layers.add("ANNOTATIONS", color=7)

    apply_layer_transparency_if_supported(doc, "SATELLITE", 0.5)

    # No insertar fondo porque SATELLITE_BACKGROUND=False

    for pts in lines_ll:
        if len(pts)==2: msp.add_line(pts[0], pts[1], dxfattribs={"layer":"LINES"})
        else:           msp.add_lwpolyline(pts, dxfattribs={"layer":"LINES"})
    for ring in polys_ll:
        if len(ring)>=3: msp.add_lwpolyline(ring, dxfattribs={"layer":"POLYGONS", "closed": True})

    # Norte simple + escala (texto) dentro del DXF
    minlon, minlat, maxlon, maxlat = bbox_wgs84
    width_deg  = maxlon - minlon
    height_deg = maxlat - minlat
    r = 0.04 * width_deg
    cx = maxlon - 0.08*width_deg
    cy = minlat + 0.85*height_deg
    msp.add_circle((cx, cy), r, dxfattribs={"layer":"ANNOTATIONS"})
    tri = [(cx, cy+r*0.95), (cx-r*0.6, cy-r*0.3), (cx+r*0.6, cy-r*0.3), (cx, cy+r*0.95)]
    msp.add_lwpolyline(tri, dxfattribs={"layer":"ANNOTATIONS", "closed": True})
    txt = msp.add_text("N", dxfattribs={"height": r*0.9, "layer": "ANNOTATIONS"})
    txt.dxf.insert = (cx - r*0.25, cy - r*0.15)


    msp.add_mtext("CRS: WGS84 (EPSG:4326)", dxfattribs={"layer":"ANNOTATIONS"}).set_location((minlon, minlat))
    doc.saveas(dxf_path)

# ==================== PRJ (EPSG:4326) ====================
def write_prj_wgs84(prj_path):
    try:
        from pyproj import CRS
        wkt = CRS.from_epsg(4326).to_wkt()
    except Exception:
        wkt = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
               'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
    with open(prj_path, "w", encoding="utf-8") as f:
        f.write(wkt)

# ==================== DWG vía ODA (opcional) ====================
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

# ==================== PDF (UTM) ====================
def meters_per_degree(lat_deg):
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat) - 0.0023*math.cos(6*lat)
    m_per_deg_lon = 111412.84*math.cos(lat) - 93.5*math.cos(3*lat) + 0.118*math.cos(5*lat)
    return m_per_deg_lon, m_per_deg_lat

def choose_nice_scale(den):
    candidates = [500, 1000, 2000, 5000, 10000, 20000]
    return min(candidates, key=lambda c: abs(c - den))

def compute_scale_denominator_m(bbox_m, fig_width_in):
    minx, miny, maxx, maxy = bbox_m
    ground_w = maxx - minx
    paper_w = fig_width_in * 0.0254
    if paper_w <= 0 or ground_w <= 0: return 1000
    return max(1, int(round(ground_w / paper_w)))

def nice_scale_bar_length(den):
    # que ocupe ~3 cm en papel
    target = den * 0.03
    nice = [10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000]
    return min(nice, key=lambda x: abs(x - target))

def draw_map_elements_utm(ax, bbox_m, zone_str):
    """Leyenda pequeña, escala compacta y Norte nítido pegado a la esquina."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FormatStrFormatter
    from PIL import Image

    minx, miny, maxx, maxy = bbox_m
    dx = maxx - minx
    dy = maxy - miny

    # --- Ejes UTM con enteros (sin notación científica)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.set_xlabel(f"Easting (m) UTM {zone_str}")
    ax.set_ylabel(f"Northing (m) UTM {zone_str}")

        # ===================== LEYENDA (arriba-izq, corregida) =====================
    leg = ax.inset_axes([0.015, 0.855, 0.145, 0.115])  # [x,y,w,h]
    leg.axis("off")
    leg.add_patch(Rectangle((0,0), 1, 1, fill=False, linewidth=0.8))
    leg.text(0.5, 0.84, "LEYENDA", ha="center", va="center",
             fontsize=5.5, fontweight="bold")

    # línea azul (RED BITEL)
    leg.add_line(plt.Line2D([0.10, 0.32], [0.63, 0.63], linewidth=1.8, color="#1f4fff"))
    leg.text(0.38, 0.63, "RED BITEL", va="center", ha="left", fontsize=5.0)

    # rectángulo fucsia (ÁREA DE PROYECTO) — centrado correctamente
    # ahora el texto queda alineado al centro del cuadrado y con margen interno
    leg.add_patch(Rectangle((0.10, 0.36), 0.16, 0.18, fill=False,
                            linewidth=1.3, edgecolor="#ff00ff"))
    leg.text(0.38, 0.45, "ÁREA DE PROYECTO", va="center", ha="left", fontsize=5.0)


    # ===================== ESCALA (abajo-izq, compacta y separada) =====================
    den = choose_nice_scale(compute_scale_denominator_m(bbox_m, fig_width_in=8.27))
    bar_len_m = nice_scale_bar_length(den)

    # Barra (no se superpone con el texto "ESC")
    x0 = minx + 0.045 * dx
    y0 = miny + 0.055 * dy
    bar_h = 0.0020 * dy
    ax.add_patch(Rectangle((x0, y0), bar_len_m, bar_h, color="black", zorder=4))

    # textos de extremos (pequeños, con caja blanca)
    ax.text(x0, y0 + 0.0045*dy, "0 m",
            fontsize=5.0, ha="left", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=5)
    ax.text(x0 + bar_len_m, y0 + 0.0045*dy, f"{bar_len_m} m",
            fontsize=5.0, ha="right", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=5)

    # “ESC 1/xxxx” en su propio inset (separado de la barra)
    esc = ax.inset_axes([0.015, 0.020, 0.13, 0.050])  # más chico y alejado
    esc.axis("off")
    esc.add_patch(Rectangle((0.00, 0.25), 0.07, 0.50, color="black"))
    esc.text(0.09, 0.26, f"ESC 1/{den}", fontsize=6.0, va="bottom")

    # ===================== NORTE (arriba-derecha EXACTAMENTE en la esquina) =====================
    pad = 0.006
    w = h = 0.11
    x = 1.0 - pad - w
    y = 1.0 - pad - h

    if os.path.exists(NORTH_PNG):
        n_ax = ax.inset_axes([x, y, w, h])
        n_ax.axis("off")

        # Mejora de nitidez: reescala x3 con LANCZOS antes de insertar
        try:
            img = Image.open(NORTH_PNG)
            img = img.resize((img.width*3, img.height*3), Image.LANCZOS)
            n_ax.imshow(img)
        except Exception:
            n_ax.imshow(Image.open(NORTH_PNG))
    else:
        # fallback vectorial
        r = 0.09 * dx
        cx = maxx - pad*dx - 0.5*w*dx
        cy = maxy - pad*dy - 0.5*h*dy
        circ = plt.Circle((cx, cy), r, fill=False, linewidth=1.2, color="black")
        ax.add_patch(circ)
        tx = [cx, cx - r*0.6, cx + r*0.6, cx]
        ty = [cy + r*0.95, cy - r*0.3,  cy - r*0.3, cy + r*0.95]
        ax.plot(tx, ty, color="black", linewidth=1.2)
        ax.text(cx, cy - r*0.55, "N", ha="center", va="center",
                fontsize=9.0, fontweight="bold", color="black")


def safe_save_pdf(fig, path):
    base, ext = os.path.splitext(path)
    if ext.lower() != ".pdf":
        ext = ".pdf"
    tmp = f"{base}.tmp{ext}"
    try:
        fig.savefig(tmp, format="pdf", bbox_inches="tight")
        os.replace(tmp, path)
        print(f"[OK] PDF: {path}")
        return
    except PermissionError:
        for i in range(1, 50):
            alt = f"{base}_{i}{ext}"
            try:
                fig.savefig(alt, format="pdf", bbox_inches="tight")
                print(f"[WARN] {path} en uso. Guardado como: {alt}")
                return
            except PermissionError:
                continue
        print("[ERROR] No se pudo escribir el PDF (archivo bloqueado).")
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except Exception: pass

def render_pdf_utm(lines_ll, polys_ll, bbox_ll, pdf_path, title_text):
    # Proyectar a UTM y añadir margen 10%:
    lines_e, polys_e, bbox_m, zone_str = project_to_utm(lines_ll, polys_ll, bbox_ll)
    minx, miny, maxx, maxy = bbox_m

    fig = plt.figure(figsize=(11.69, 8.27))  # A4 apaisado para mayor área útil
    ax = fig.add_subplot(111)

    # Dibujar vectores UTM
    for pts in lines_e:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, linewidth=2.2, color="#1f4fff")
    for ring in polys_e:
        xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
        ax.plot(xs, ys, linewidth=1.6, color="#ff00ff")

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([minx, maxx]); ax.set_ylim([miny, maxy])
    ax.set_title(title_text)
    ax.grid(True, linestyle="--", linewidth=0.3)

    draw_map_elements_utm(ax, bbox_m, zone_str)

    safe_save_pdf(fig, pdf_path)
    plt.close(fig)

# ==================== MAIN ====================
def main():
    if not os.path.exists(KMZ_NAME):
        print(f"[ERROR] No encuentro {KMZ_NAME} en esta carpeta.")
        return

    print("[INFO] Leyendo geometrías del KMZ (WGS84)…")
    lines_ll, polys_ll = read_geometries_wgs84(KMZ_NAME)
    if not lines_ll and not polys_ll:
        print("[ERROR] El KMZ no contiene LineString ni Polygon utilizables.")
        return

    # BBox WGS84 (para DXF y para proyectar a UTM)
    lons = [lon for g in (lines_ll + polys_ll) for lon,lat in g]
    lats = [lat for g in (lines_ll + polys_ll) for lon,lat in g]
    bbox84 = bbox_wgs84(lons, lats, margin_ratio=0.06)

    print("[INFO] Creando DXF (WGS84)…")
    make_dxf_wgs84(lines_ll, polys_ll, DXF_OUT, PNG_W84, bbox84)
    print(f"[OK] DXF: {DXF_OUT}")

    print("[INFO] Escribiendo PRJ (EPSG:4326)…")
    write_prj_wgs84(PRJ_OUT)
    print(f"[OK] PRJ: {PRJ_OUT}")

    print("[INFO] Intentando convertir a DWG (si ODA/Teigha está instalado)…")
    converted = try_convert_dxf_to_dwg(DXF_OUT, DWG_OUT)
    print("[OK] DWG generado." if converted else "[AVISO] No se generó DWG automáticamente. Se entrega DXF.")

    print("[INFO] Renderizando PDF (UTM)…")
    render_pdf_utm(lines_ll, polys_ll, bbox84, PDF_OUT, "INFORMACIÓN DE RUTAS ACTUALES")
    print("[OK] Proceso finalizado.")

if __name__ == "__main__":
    main()
