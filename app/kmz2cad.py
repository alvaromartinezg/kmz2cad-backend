import os, zipfile, math, shutil, subprocess, xml.etree.ElementTree as ET

# backend sin GUI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

# ───────── CONFIG ─────────
KMZ_NAME         = "Exportado.kmz"
TEMPLATE_DXF_IN  = "PLANTILLA.dxf"        # tu plantilla (guardar como AutoCAD 2018 DXF)
FINAL_DXF_OUT    = "PLANTILLA_FINAL.dxf"  # plantilla + nuevos datos en Model
LAYOUT_NAME      = "I-01"                 # layout de salida
PDF_OUT          = "I-01.pdf"             # PDF del layout
CLEAN_DXF_OUT    = "exportado_wgs84.dxf"  # DXF limpio sin plantilla
DWG_OUT          = "exportado_wgs84.dwg"  # intentará si ODA está instalado
SHIFT_TO_LOCAL = True  # ← ponlo en True para mover el plano al origen (recomendado)
PAPER_FRAME_RECT = (7.8839, 6.1980, 252.5161, 201.1238)

# ───────── KML NS ─────────
NS = {"kml":"http://www.opengis.net/kml/2.2"}
for p,u in NS.items():
    ET.register_namespace("" if p=="kml" else p, u)

# ───────── Utilidades KML ─────────
# --- Admin meta (UBIGEO/DEP/PROV/DIST) -------------------
import json

def load_admin_meta(base_dir: str = "."):
    """
    Intenta leer ./meta.json con:
      { "ubigeo": "...", "departamento": "...", "provincia": "...", "distrito": "..." }
    Si no existe, usa variables de entorno (UBIGEO, DEPARTAMENTO, PROVINCIA, DISTRITO).
    Nunca lanza excepción; siempre devuelve un dict con strings.
    """
    meta = {
        "ubigeo": "",
        "departamento": "",
        "provincia": "",
        "distrito": "",
    }

    # 1) meta.json (preferencia)
    try:
        p = os.path.join(base_dir, "meta.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
                meta.update({k: (str(data.get(k, "") or "")).strip() for k in meta.keys()})
    except Exception as e:
        print(f"[WARN] meta.json no legible: {e}")

    # 2) ENV (rellena vacíos)
    meta["ubigeo"]       = (os.getenv("UBIGEO",       meta["ubigeo"]) or "").strip()
    meta["departamento"] = (os.getenv("DEPARTAMENTO", meta["departamento"]) or "").strip()
    meta["provincia"]    = (os.getenv("PROVINCIA",    meta["provincia"]) or "").strip()
    meta["distrito"]     = (os.getenv("DISTRITO",     meta["distrito"]) or "").strip()

    # Normaliza UBIGEO a 6 dígitos si es numérico
    if meta["ubigeo"].isdigit():
        meta["ubigeo"] = meta["ubigeo"].zfill(6)

    return meta

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
    roots = []
    with zipfile.ZipFile(kmz_path, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
        if not names:
            raise FileNotFoundError("El KMZ no contiene ningún .kml interno")
        for n in names:
            data = zf.read(n)
            root = _parse_kml_with_fallback(data, name=n)
            roots.append(root)
    return roots


def read_geometries_wgs84(kmz_path):
    lines, polys = [], []
    for root in read_kml_roots_from_kmz(kmz_path):
        for ls in root.findall(".//kml:LineString", NS):
            ce = ls.find("kml:coordinates", NS)
            pts = parse_coords(ce.text if ce is not None else "")
            if len(pts)>=2: lines.append(pts)
        for lr in root.findall(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing", NS):
            ce = lr.find("kml:coordinates", NS)
            ring = parse_coords(ce.text if ce is not None else "")
            if len(ring)>=3:
                if ring[0]!=ring[-1]: ring.append(ring[0])
                polys.append(ring)
    return lines, polys

def bbox_wgs84(lons, lats, margin_ratio=0.06):
    minlon, minlat = min(lons), min(lats)
    maxlon, maxlat = max(lons), max(lats)
    dx = (maxlon - minlon) * margin_ratio
    dy = (maxlat - minlat) * margin_ratio
    return (minlon - dx, minlat - dy, maxlon + dx, maxlat + dy)

# KML usa ABGR (aabbggrr). Ignoramos la 'aa' y devolvemos (r,g,b)
def kml_abgr_to_rgb(hexstr: str):
    try:
        s = hexstr.strip()
        if s.startswith("#"): s = s[1:]
        if len(s) != 8:  # debería ser aabbggrr
            return (0, 0, 0)
        aa = int(s[0:2], 16)  # alpha (no usado)
        bb = int(s[2:4], 16)
        gg = int(s[4:6], 16)
        rr = int(s[6:8], 16)
        return (rr, gg, bb)
    except Exception:
        return (0, 0, 0)

def build_kml_style_index(roots):
    """Devuelve (style_line, stylemap) donde:
       - style_line[id] = (r,g,b) de LineStyle/color
       - stylemap[id]   = id de Style 'normal' (si hay StyleMap)
    """
    style_line = {}
    stylemap = {}

    for root in roots:
        # <Style id="..."><LineStyle><color>aabbggrr</color>
        for st in root.findall(".//kml:Style", NS):
            sid = st.get("id")
            if not sid: continue
            ls = st.find("kml:LineStyle", NS)
            if ls is not None:
                ce = ls.find("kml:color", NS)
                if ce is not None and ce.text:
                    style_line[sid] = kml_abgr_to_rgb(ce.text)

        # <StyleMap id="..."><Pair><key>normal</key><styleUrl>#something</styleUrl>
        for sm in root.findall(".//kml:StyleMap", NS):
            sid = sm.get("id")
            if not sid: continue
            pairs = sm.findall(".//kml:Pair", NS)
            for pr in pairs:
                key_el = pr.find("kml:key", NS)
                url_el = pr.find("kml:styleUrl", NS)
                if key_el is not None and url_el is not None and (key_el.text or "").strip().lower() == "normal":
                    url = (url_el.text or "").strip()
                    if url.startswith("#"): url = url[1:]
                    stylemap[sid] = url
    return style_line, stylemap

def placemark_line_rgb(pm, style_line, stylemap):
    """Color de línea para un Placemark (prefiere inline Style; si no, styleUrl)."""
    # 1) Inline Style (si existe)
    st = pm.find(".//kml:Style/kml:LineStyle/kml:color", NS)
    if st is not None and st.text:
        return kml_abgr_to_rgb(st.text)

    # 2) styleUrl → id → StyleMap/Style
    su = pm.find("kml:styleUrl", NS)
    if su is not None and su.text:
        url = su.text.strip()
        if url.startswith("#"): url = url[1:]
        # ¿es un StyleMap?
        if url in stylemap:
            url = stylemap[url]
        # ¿hay Style con LineStyle?
        if url in style_line:
            return style_line[url]

    # 3) Color por defecto si no hay nada en KML
    return (0, 0, 0)

def read_geometries_wgs84_colored(kmz_path):
    """Retorna:
       lines = [ ([(lon,lat),...], (r,g,b)), ... ]
       polys = [ ([(lon,lat),...], (r,g,b)), ... ]  (color del contorno)
    """
    roots = read_kml_roots_from_kmz(kmz_path)
    style_line, stylemap = build_kml_style_index(roots)

    lines = []
    polys = []

    for root in roots:
        for pm in root.findall(".//kml:Placemark", NS):
            rgb = placemark_line_rgb(pm, style_line, stylemap)

            # LineString
            ls = pm.find(".//kml:LineString", NS)
            if ls is not None:
                ce = ls.find("kml:coordinates", NS)
                pts = parse_coords(ce.text if ce is not None else "")
                pts2 = [(lon, lat) for lon, lat in [(p[0], p[1]) for p in pts]]
                if len(pts2) >= 2:
                    lines.append((pts2, rgb))
                continue

            # Polygon (outerBoundary)
            lr = pm.find(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing", NS)
            if lr is not None:
                ce = lr.find("kml:coordinates", NS)
                ring = parse_coords(ce.text if ce is not None else "")
                ring2 = [(lon, lat) for lon, lat in [(p[0], p[1]) for p in ring]]
                if len(ring2) >= 3:
                    if ring2[0] != ring2[-1]:
                        ring2.append(ring2[0])
                    polys.append((ring2, rgb))

    return lines, polys

def shift_to_local_coords_colored(lines_col, polys_col):
    # bbox original (en lon/lat)
    lons = [x for (pts, _rgb) in (lines_col + polys_col) for x, y in pts]
    lats = [y for (pts, _rgb) in (lines_col + polys_col) for x, y in pts]
    minlon, minlat, maxlon, maxlat = min(lons), min(lats), max(lons), max(lats)
    cx = (minlon + maxlon) / 2.0
    cy = (minlat + maxlat) / 2.0

    def sh(seq): return [(x - cx, y - cy) for x, y in seq]

    lines_local = [(sh(pts), rgb) for (pts, rgb) in lines_col]
    polys_local = [(sh(r),   rgb) for (r,   rgb) in polys_col]

    w = (maxlon - minlon); h = (maxlat - minlat)
    bbox_local = (-w/2, -h/2, w/2, h/2)
    return lines_local, polys_local, bbox_local, (cx, cy)

from ezdxf.colors import rgb2int

def replace_model_content(doc, lines_col, polys_col, bbox84):
    """Ahora espera:
       lines_col = [ (pts, (r,g,b)), ... ]
       polys_col = [ (ring, (r,g,b)), ... ]
    """
    msp = doc.modelspace()
    for e in list(msp):
        try: e.destroy()
        except Exception: pass

    ensure_layers(doc)

    # Líneas
    for pts, rgb in lines_col:
        if len(pts) == 2:
            e = msp.add_line(pts[0], pts[1], dxfattribs={"layer":"LINES"})
        else:
            e = msp.add_lwpolyline(pts, dxfattribs={"layer":"LINES"})
        try:
            e.dxf.true_color = rgb2int(rgb)  # color real (RGB)
            e.dxf.color = 256                # BYLAYER → ignorado si hay true_color
        except Exception:
            pass

    # Polígonos (contorno)
    for ring, rgb in polys_col:
        if len(ring) >= 3:
            e = msp.add_lwpolyline(ring, dxfattribs={"layer":"POLYGONS", "closed": True})
            try:
                e.dxf.true_color = rgb2int(rgb)
                e.dxf.color = 256
            except Exception:
                pass

    # Extents / Unidades
    minlon, minlat, maxlon, maxlat = bbox84
    doc.header['$EXTMIN'] = (minlon, minlat, 0.0)
    doc.header['$EXTMAX'] = (maxlon, maxlat, 0.0)
    doc.header['$INSUNITS'] = 0

# ───────── DXF helper ─────────
def ensure_layers(doc):
    layers = doc.layers
    if "LINES" not in layers:     layers.add("LINES", color=1)      # rojo
    if "POLYGONS" not in layers:  layers.add("POLYGONS", color=6)   # magenta
    if "ANNOTATIONS" not in layers: layers.add("ANNOTATIONS", color=7)

def shift_to_local_coords(lines_ll, polys_ll):
    # centra el contenido en (0,0) restando el centroide del bbox
    lons = [x for g in (lines_ll + polys_ll) for x, y in g]
    lats = [y for g in (lines_ll + polys_ll) for x, y in g]
    minlon, minlat, maxlon, maxlat = min(lons), min(lats), max(lons), max(lats)
    cx = (minlon + maxlon) / 2.0
    cy = (minlat + maxlat) / 2.0

    def sh(seq): return [(x - cx, y - cy) for x, y in seq]

    lines_local = [sh(pts) for pts in lines_ll]
    polys_local = [sh(r) for r in polys_ll]
    # bbox en coords locales
    w = (maxlon - minlon)
    h = (maxlat - minlat)
    bbox_local = (-w/2, -h/2, w/2, h/2)
    return lines_local, polys_local, bbox_local, (cx, cy)

def _get_active_model_vport(doc):
    """
    Devuelve un VPORT '*ACTIVE' del Model. Si hay varios, toma el primero.
    Si no existe, lo crea.
    """
    try:
        vps = [v for v in doc.viewports if v.dxf.name.upper() == "*ACTIVE"]
        if vps:
            return vps[0]
        return doc.viewports.new("*ACTIVE")
    except Exception:
        return doc.viewports.new("*ACTIVE")

def center_model_view(doc, bbox84=None, pad=1.06):
    """
    Centra la vista de Model ajustando el VPORT '*ACTIVE' únicamente con
    atributos soportados: center y height. Usa el bbox que ya calculaste.
    """
    if bbox84 is None:
        return

    # obtener/crear el VPORT *ACTIVE*
    try:
        vps = [v for v in doc.viewports if v.dxf.name.upper() == "*ACTIVE"]
        vp = vps[0] if vps else doc.viewports.new("*ACTIVE")
    except Exception:
        vp = doc.viewports.new("*ACTIVE")

    minx, miny, maxx, maxy = bbox84
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    extra_margin = 0.01 * max(maxx - minx, maxy - miny)  # ~1%
    width  = (maxx - minx) + extra_margin
    height = (maxy - miny) + extra_margin

    # 'height' controla la altura visible en unidades de MODEL.
    # usa el mayor de (ancho, alto) para asegurar que todo quepa
    view_h = max(width, height, 1e-9)

    # SOLO estos dos atributos son necesarios y portables en VPORT:
    vp.dxf.center = (cx, cy)
    vp.dxf.height = view_h

def fit_layout_viewport(doc, layout_name, bbox, occupancy=0.58, safety=1.00,
                        frame_rect=None):
    if layout_name not in doc.layouts:
        print(f"[WARN] No existe el layout '{layout_name}'.")
        return
    layout = doc.layouts.get(layout_name)

    # 1) (opcional) recuadro del marco útil en PAPEL
    #    Si se da, el viewport ocupará EXACTAMENTE ese rectángulo.
    if frame_rect is not None:
        x0, y0, x1, y1 = map(float, frame_rect)
        paper_center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        paper_size   = (x1 - x0, y1 - y0)
        # borrar el viewport más grande si existe
        for e in [e for e in layout if e.dxftype()=="VIEWPORT"]:
            try: e.destroy()
            except Exception: pass
    else:
        # fallback: heredar del viewport más grande
        vps = [e for e in layout if e.dxftype()=="VIEWPORT"]
        paper_center = (210.0, 148.5)
        paper_size   = (200.0, 140.0)
        if vps:
            old = max(vps, key=lambda v: (v.dxf.width or 0)*(v.dxf.height or 0))
            paper_center = (float(old.dxf.center[0]), float(old.dxf.center[1]))
            paper_size   = (float(old.dxf.width or paper_size[0]),
                            float(old.dxf.height or paper_size[1]))
            try: old.destroy()
            except Exception: pass

    # 2) BBOX del MODEL → escala para que ocupe <= occupancy del viewport
    minx, miny, maxx, maxy = bbox
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    w_model = max(maxx - minx, 1e-9)
    h_model = max(maxy - miny, 1e-9)
    vp_aspect = (paper_size[0] or 1.0) / (paper_size[1] or 1.0)
    view_h = max(h_model/float(occupancy), w_model/(float(occupancy)*vp_aspect))
    view_h *= float(safety)

    # 3) Crear viewport centrado en el marco, tamaño = marco
    vp = layout.add_viewport(
        center=paper_center,
        size=paper_size,
        view_center_point=(cx, cy),
        view_height=max(view_h, 1e-9),
    )
    try: vp.dxf.status = 1
    except Exception: pass

    # 4) Asegurar capas
    for lname in ("LINES","POLYGONS"):
        try:
            L=doc.layers.get(lname); L.on(); L.thaw(); L.unlock()
            if hasattr(vp,"is_layer_frozen") and vp.is_layer_frozen(lname):
                vp.thaw_layer(lname)
        except Exception:
            pass

def export_layout_pdf_viewport(doc, layout_name, pdf_path):

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from ezdxf.math import Vec2, Vec3

    def _xy(val):
        if val is None:
            return None
        if isinstance(val, (Vec2, Vec3)):
            return float(val.x), float(val.y)
        try:
            return float(val[0]), float(val[1])
        except Exception:
            return None

    # --- mapeo simple de ACI (AutoCAD Color Index) a RGB ---
    _ACI_TABLE = {
        1:(255,0,0), 2:(255,255,0), 3:(0,255,0), 4:(0,255,255), 5:(0,0,255),
        6:(255,0,255), 7:(255,255,255), 8:(128,128,128), 9:(192,192,192)
    }
    def aci_to_rgb_local(aci:int):
        """Devuelve color RGB (0-255) aproximado para un índice ACI básico."""
        return _ACI_TABLE.get(int(aci), (0,0,0))

    def _rgb_from_entity(e):
        tc = int(getattr(e.dxf, "true_color", 0) or 0)
        if tc:
            r = (tc >> 16) & 0xFF
            g = (tc >> 8) & 0xFF
            b = tc & 0xFF
            return (r/255.0, g/255.0, b/255.0)
        aci = int(getattr(e.dxf, "color", 256) or 256)
        rgb = aci_to_rgb_local(aci)
        return (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)   

    # 1) Paper Space
    layout = doc.layouts.get(layout_name)
    ctx    = RenderContext(doc)

    fig = plt.figure(figsize=(11.69, 8.27))  # A4 apaisado
    ax  = fig.add_axes([0, 0, 1, 1])
    Frontend(ctx, MatplotlibBackend(ax)).draw_layout(layout)

    # 2) Viewport principal (mayor área en papel)
    vps = [e for e in layout if e.dxftype() == "VIEWPORT"]
    if not vps:
        fig.savefig(pdf_path, format="pdf", dpi=300)
        plt.close(fig)
        print(f"[OK] PDF (solo marco; sin viewports): {pdf_path}")
        return

    vp = max(vps, key=lambda v: float(v.dxf.width or 0) * float(v.dxf.height or 0))

    # Tamaño/posición del rectángulo del viewport en papel
    wp = float(vp.dxf.width or 0.0)
    hp = float(vp.dxf.height or 0.0)
    if wp <= 0.0 or hp <= 0.0:
        fig.savefig(pdf_path, format="pdf", dpi=300)
        plt.close(fig)
        print(f"[OK] PDF (solo marco; viewport sin tamaño): {pdf_path}")
        return

    cxp, cyp = float(vp.dxf.center[0]), float(vp.dxf.center[1])
    left, bottom = cxp - wp/2.0, cyp - hp/2.0

    # Ventana del MODEL (centro y altura visibles en unidades de Model)
    vcp = _xy(getattr(vp.dxf, "view_center_point", None))
    vh  = float(getattr(vp.dxf, "view_height", 0.0) or 0.0)
    if vcp is None or vh <= 0.0:
        fig.savefig(pdf_path, format="pdf", dpi=300)
        plt.close(fig)
        print(f"[OK] PDF (solo marco; viewport sin centro/altura): {pdf_path}")
        return
    cxm, cym = vcp

    # Transformación Model -> Papel (rectángulo del viewport)
    aspect = wp / hp
    vw     = vh * aspect               # ancho visible en model
    xmin   = cxm - vw/2.0; xmax = cxm + vw/2.0
    ymin   = cym - vh/2.0; ymax = cym + vh/2.0
    sx     = wp / (xmax - xmin)
    sy     = hp / (ymax - ymin)

    def to_paper(x, y):
        X = left + (x - xmin) * sx
        Y = bottom + (y - ymin) * sy
        return X, Y

    # Clip rectangular (Cohen–Sutherland simplificado)
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8
    def code(x, y):
        c = INSIDE
        if x < xmin: c |= LEFT
        elif x > xmax: c |= RIGHT
        if y < ymin: c |= BOTTOM
        elif y > ymax: c |= TOP
        return c

    def clip_segment(p1, p2):
        x1, y1 = p1; x2, y2 = p2
        c1, c2 = code(x1, y1), code(x2, y2)
        while True:
            if (c1 | c2) == 0:      # ambos dentro
                return (x1, y1), (x2, y2)
            if (c1 & c2) != 0:      # comparten zona fuera
                return None
            c_out = c1 or c2
            if c_out & TOP:
                x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1); y = ymax
            elif c_out & BOTTOM:
                x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1); y = ymin
            elif c_out & RIGHT:
                y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1); x = xmax
            else:  # LEFT
                y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1); x = xmin
            if c_out == c1:
                x1, y1 = x, y; c1 = code(x1, y1)
            else:
                x2, y2 = x, y; c2 = code(x2, y2)

    # 3) Dibuja LINE / LWPOLYLINE del Model en PAPEL, con color
    any_drawn = False
    msp = doc.modelspace()
    for e in msp:
        try:
            if e.dxftype() == "LINE":
                p1 = (float(e.dxf.start[0]), float(e.dxf.start[1]))
                p2 = (float(e.dxf.end[0]),   float(e.dxf.end[1]))
                seg = clip_segment(p1, p2)
                if not seg:
                    continue
                (x1, y1), (x2, y2) = seg
                X1, Y1 = to_paper(x1, y1)
                X2, Y2 = to_paper(x2, y2)
                ax.add_line(plt.Line2D([X1, X2], [Y1, Y2], linewidth=1, color=_rgb_from_entity(e))) #Antes era 1.2 width
                any_drawn = True

            elif e.dxftype() == "LWPOLYLINE":
                pts = [(float(p[0]), float(p[1])) for p in e.get_points()]
                if len(pts) < 2:
                    continue
                col = _rgb_from_entity(e)
                x_prev, y_prev = pts[0]
                for (x, y) in pts[1:]:
                    seg = clip_segment((x_prev, y_prev), (x, y))
                    if seg:
                        (xa, ya), (xb, yb) = seg
                        Xa, Ya = to_paper(xa, ya)
                        Xb, Yb = to_paper(xb, yb)
                        ax.add_line(plt.Line2D([Xa, Xb], [Ya, Yb], linewidth=1.2, color=col))
                        any_drawn = True
                    x_prev, y_prev = x, y
        except Exception:
            continue

    # 4) (opcional) dibujar el marco del viewport como guía visual
    ax.add_patch(Rectangle((left, bottom), wp, hp, fill=False, linewidth=0.6))

    # 5) Exporta
    fig.savefig(pdf_path, format="pdf", dpi=300)
    plt.close(fig)
    print(f"[OK] {'PDF con viewport aplanado' if any_drawn else 'PDF (solo Paper Space; sin geometría visible en viewport)'}: {pdf_path}")

def write_clean_dxf(lines_ll, polys_ll, bbox84, path):
    doc = ezdxf.new("R2018")
    ensure_layers(doc)
    msp = doc.modelspace()
    for pts in lines_ll:
        if len(pts)==2: msp.add_line(pts[0], pts[1], dxfattribs={"layer":"LINES"})
        else:           msp.add_lwpolyline(pts, dxfattribs={"layer":"LINES"})
    for ring in polys_ll:
        if len(ring)>=3: msp.add_lwpolyline(ring, dxfattribs={"layer":"POLYGONS", "closed": True})
    minlon, minlat, maxlon, maxlat = bbox84
    doc.header['$EXTMIN'] = (minlon, minlat, 0.0)
    doc.header['$EXTMAX'] = (maxlon, maxlat, 0.0)
    doc.header['$INSUNITS'] = 0
    doc.saveas(path)
    print(f"[OK] DXF limpio: {path}")

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
        print("[INFO] No se encontró ODA/Teigha. Se omite DWG.")
        return False

    out_dir = os.path.dirname(os.path.abspath(dxf_path)) or "."
    in_dir  = os.path.dirname(os.path.abspath(dxf_path)) or "."
    print(f"[INFO] Convirtiendo DXF→DWG con: {exe}")
    cmd = [exe, in_dir, out_dir, "ACAD2018", "1", "1", ""]
    try:
        subprocess.run(cmd, check=True)
        maybe = os.path.join(out_dir, os.path.splitext(os.path.basename(dxf_path))[0] + ".dwg")
        if os.path.exists(maybe):
            if os.path.abspath(maybe) != os.path.abspath(dwg_path):
                shutil.copy2(maybe, dwg_path)
            print(f"[OK] DWG: {dwg_path}")
            return True
    except Exception as e:
        print(f"[WARN] ODA falló: {e}")
    return False

# --- NORMALIZAR AL ORIGEN (garantiza que Model abra centrado) ---
def normalize_to_origin(lines_ll, polys_ll):
    # centra el contenido en (0,0) restando el centro del bbox
    xs = [x for g in (lines_ll + polys_ll) for x, y in g]
    ys = [y for g in (lines_ll + polys_ll) for x, y in g]
    if not xs or not ys:
        return lines_ll, polys_ll, (0,0,0,0)

    minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0

    def shift(seq): return [(x - cx, y - cy) for x, y in seq]

    lines0 = [shift(pts) for pts in lines_ll]
    polys0 = [shift(r)   for r   in polys_ll]

    w = (maxx - minx); h = (maxy - miny)
    bbox0 = (-w/2.0, -h/2.0, w/2.0, h/2.0)
    return lines0, polys0, bbox0

from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

def replace_placeholders(doc, meta=None):
    """
    Reemplaza tokens en la plantilla. Si 'meta' viene vacío o falta,
    usa 'Pendiente' para los nombres y deja UBIGEO tal cual (o vacío).
    Tokens soportados:
      - NOMBRE_DISTRITO
      - NOMBRE_PROVINCIA
      - NOMBRE_DEPARTAMENTO
      - UBIGEO
      - DD/MM/AAAA (fecha actual Lima)
    """
    if meta is None:
        meta = {}

    distrito     = (meta.get("distrito") or "Pendiente").strip()
    provincia    = (meta.get("provincia") or "Pendiente").strip()
    departamento = (meta.get("departamento") or "Pendiente").strip()
    ubigeo       = (meta.get("ubigeo") or "").strip()

    hoy = datetime.now(ZoneInfo("America/Lima")).strftime("%d/%m/%Y")
    tokens = {
        "NOMBRE_DISTRITO": distrito,
        "NOMBRE_PROVINCIA": provincia,
        "NOMBRE_DEPARTAMENTO": departamento,
        "UBIGEO": ubigeo,
        "DD/MM/AAAA": hoy,
    }

    for layout in doc.layouts:
        for e in layout:
            try:
                if e.dxftype() == "TEXT":
                    txt = e.dxf.text or ""
                    new_txt = txt
                    for k, v in tokens.items():
                        if k in new_txt:
                            new_txt = new_txt.replace(k, v)
                    if new_txt != txt:
                        e.dxf.text = new_txt

                elif e.dxftype() == "MTEXT":
                    txt = e.text or ""
                    new_txt = txt
                    for k, v in tokens.items():
                        if k in new_txt:
                            new_txt = new_txt.replace(k, v)
                    if new_txt != txt:
                        e.text = new_txt
            except Exception:
                continue


def upsert_utm_4lines(
    doc,
    layout_name,
    offset_lonlat,
    layer_name="GRID_UTM",
    lineweight=7,                 # mitad del grosor anterior (≈0.07 mm)
    text_h=2.5,
    label_gap=2.0,                # separación para rótulos X (arriba/abajo)
    y_label_inset=3.0,            # rótulos del eje Y un poco hacia adentro
    frame_rect=None,              # (xmin, ymin, xmax, ymax) del marco de tu plantilla
    epsg_override=None,
    zone_override=None,
    south_override=None,
):
    """
    Dibuja 4 líneas UTM verticales y 4 horizontales que
    se EXTiENDEN hasta el marco de la plantilla (frame_rect).
    - Rótulos X arriba/abajo del marco.
    - Rótulos Y dentro del marco (inset).
    Requiere: pyproj
    """
    try:
        from pyproj import Transformer
        from ezdxf.math import Vec2, Vec3
    except Exception:
        print("[WARN] Falta pyproj (pip install pyproj).")
        return

    def _xy(val):
        if val is None: return None
        if isinstance(val, (Vec2, Vec3)): return float(val.x), float(val.y)
        try: return float(val[0]), float(val[1])
        except Exception: return None

    if layout_name not in doc.layouts:
        print(f"[WARN] Layout '{layout_name}' no existe.")
        return
    layout = doc.layouts.get(layout_name)

    vps = [e for e in layout if e.dxftype() == "VIEWPORT"]
    if not vps:
        print("[WARN] Layout sin VIEWPORTs.")
        return
    vp = max(vps, key=lambda v: float(v.dxf.width or 0) * float(v.dxf.height or 0))

    # Rectángulo del viewport en PAPEL (mm)
    wp = float(vp.dxf.width or 0.0); hp = float(vp.dxf.height or 0.0)
    if wp <= 0.0 or hp <= 0.0:
        print("[WARN] VIEWPORT sin tamaño.")
        return
    cxp, cyp = float(vp.dxf.center[0]), float(vp.dxf.center[1])
    v_left, v_bottom = cxp - wp/2.0, cyp - hp/2.0
    v_right, v_top   = v_left + wp, v_bottom + hp

    # Marco de la plantilla (si no lo das, usa el rectángulo del viewport)
    if frame_rect is None:
        f_left, f_bottom, f_right, f_top = v_left, v_bottom, v_right, v_top
    else:
        f_left, f_bottom, f_right, f_top = map(float, frame_rect)

    # Ventana visible del MODEL
    vcp = _xy(getattr(vp.dxf, "view_center_point", None))
    vh  = float(getattr(vp.dxf, "view_height", 0.0) or 0.0)
    if vcp is None or vh <= 0.0:
        print("[WARN] VIEWPORT sin view_center_point/view_height válidos.")
        return
    cxm, cym = vcp
    aspect = wp / hp
    vw = vh * aspect
    xmin = cxm - vw/2.0; xmax = cxm + vw/2.0
    ymin = cym - vh/2.0; ymax = cym + vh/2.0

    # Revertir normalización a lon/lat reales
    off_lon, off_lat = offset_lonlat
    lon_center = cxm + off_lon
    lat_center = cym + off_lat

    # Selección de CRS
    if epsg_override is not None:
        epsg = int(epsg_override); zone = None; south = None
    else:
        if zone_override is not None:
            zone = int(zone_override)
            south = bool(south_override) if south_override is not None else (lat_center < 0)
        else:
            zone = int((lon_center + 180) // 6 + 1)
            south = (lat_center < 0)
        epsg = 32700 + zone if south else 32600 + zone

    tr = Transformer.from_crs(4326, epsg, always_xy=True)

    # Transformación Model→Papel (a coordenadas del viewport)
    sx = wp / (xmax - xmin)
    sy = hp / (ymax - ymin)
    def to_paper(x, y):
        return (v_left + (x - xmin) * sx, v_bottom + (y - ymin) * sy)

    # Limpia la capa previa
    if layer_name not in doc.layers:
        doc.layers.add(layer_name, color=8)  # gris
    for e in list(layout):
        try:
            if (e.dxf.layer or "") == layer_name:
                e.destroy()
        except Exception:
            pass

    # Fracciones internas: siempre 4
    fracs = [1/5, 2/5, 3/5, 4/5]

    # Utilidades de texto
    def add_mtext_center(pt, s, attach="TOP_CENTER"):
        ap_map = {
            "TOP_LEFT":1, "TOP_CENTER":2, "TOP_RIGHT":3,
            "MIDDLE_LEFT":4, "MIDDLE_CENTER":5, "MIDDLE_RIGHT":6,
            "BOTTOM_LEFT":7, "BOTTOM_CENTER":8, "BOTTOM_RIGHT":9,
        }
        m = layout.add_mtext(s, dxfattribs={"layer": layer_name, "char_height": text_h})
        m.set_location(pt, attachment_point=ap_map.get(attach, 2))

    def add_text_rotated(pt, s, deg=90.0):
        t = layout.add_text(s, dxfattribs={"layer": layer_name, "height": text_h, "rotation": float(deg)})
        t.dxf.insert = pt

    # 1) 4 verticales (Easting): prolongadas hasta el marco
    for f in fracs:
        x_model = xmin + f * (xmax - xmin)
        Xv, _ = to_paper(x_model, ymin)   # solo necesitamos X en papel

        # Línea desde el borde inferior del marco hasta el superior del marco
        layout.add_line((Xv, f_bottom), (Xv, f_top),
                        dxfattribs={"layer": layer_name, "lineweight": lineweight})

        # Etiquetas UTM (E) arriba/abajo del marco
        E, _N = tr.transform(x_model + off_lon, lat_center)
        label = f"{int(round(E))}"
        add_mtext_center((Xv, f_top + label_gap), label, attach="TOP_CENTER")
        add_mtext_center((Xv, f_bottom - label_gap), label, attach="BOTTOM_CENTER")

    # 2) 4 horizontales (Northing): prolongadas hasta el marco
    for f in fracs:
        y_model = ymin + f * (ymax - ymin)
        _, Yh = to_paper(xmin, y_model)   # solo necesitamos Y en papel

        layout.add_line((f_left, Yh), (f_right, Yh),
                        dxfattribs={"layer": layer_name, "lineweight": lineweight})

        # Etiquetas UTM (N) dentro del marco (inset)
        _E, N = tr.transform(lon_center, y_model + off_lat)
        label = f"{int(round(N))}"
        add_text_rotated((f_left + y_label_inset,  Yh), label, deg=90.0)
        add_text_rotated((f_right - y_label_inset, Yh), label, deg=90.0)

    # Leyenda de zona/epsg (opcional)
    if zone_override is not None:
        ztxt = f"UTM {zone}{'S' if south else 'N'} (EPSG:{epsg})"
    else:
        z = int((lon_center + 180) // 6 + 1); s = lat_center < 0
        ztxt = f"UTM {z}{'S' if s else 'N'} (EPSG:{epsg})"

    zone_text = f"{zone}{'S' if south else 'N'}" if zone else "?"
    set_zone_utm_in_layout(doc, layout_name, zone_text)


import re
from ezdxf.math import Vec2, Vec3

# --- PEGAR ARRIBA (junto a imports) ---
import re

def _ensure_ns_on_kml_root(xml_text: str, prefix: str, uri: str) -> str:
    """
    Si en el texto aparece 'prefix:' pero no existe xmlns:prefix en <kml ...>,
    inyecta xmlns:prefix="uri" en la etiqueta de apertura de <kml>.
    """
    if f"{prefix}:" in xml_text and f"xmlns:{prefix}=" not in xml_text:
        # Inserta antes del '>' del primer <kml ...>
        xml_text = re.sub(
            r'(<\s*kml\b[^>]*)(>)',
            rf'\1 xmlns:{prefix}="{uri}"\2',
            xml_text,
            count=1,
            flags=re.IGNORECASE
        )
    return xml_text

import re
import xml.etree.ElementTree as ET

_COMMON_NS = {
    "kml":  "http://www.opengis.net/kml/2.2",
    "gx":   "http://www.google.com/kml/ext/2.2",
    "atom": "http://www.w3.org/2005/Atom",
    "xsi":  "http://www.w3.org/2001/XMLSchema-instance",
    "xlink":"http://www.w3.org/1999/xlink",
    "gml":  "http://www.opengis.net/gml",
    "ogr":  "http://ogr.maptools.org/",
    "esri": "http://www.esri.com/",
}

def _inject_xmlns_in_root(xml_text: str, pfx: str, uri: str) -> str:
    # inserta xmlns:pfx="uri" en la etiqueta de apertura de <...> raíz
    # (respetando mayúsculas/minúsculas y posibles prefijos en el nombre)
    return re.sub(
        r'(<\s*[^?!][^>\s]*\b[^>]*)(>)',
        rf'\1 xmlns:{pfx}="{uri}"\2',
        xml_text,
        count=1
    )

def _declared_prefixes(xml_text: str) -> set[str]:
    return set(re.findall(r'xmlns:([A-Za-z_][\w\-\.]*)\s*=', xml_text))

def _used_tag_prefixes(xml_text: str) -> set[str]:
    return set(re.findall(r'</?\s*([A-Za-z_][\w\-\.]*):[A-Za-z_][\w\-\.]*', xml_text))

def _used_attr_prefixes(xml_text: str) -> set[str]:
    return set(re.findall(r'\s([A-Za-z_][\w\-\.]*):[A-Za-z_][\w\-\.]*\s*=', xml_text))

def _strip_prefix_from_tags(xml_text: str, prefixes: set[str]) -> str:
    # <pfx:Name ...> → <Name ...> y </pfx:Name> → </Name>
    for p in prefixes:
        xml_text = re.sub(rf'(<\/?)\s*{re.escape(p)}:', r'\1', xml_text)
    return xml_text

def _remove_prefixed_attributes(xml_text: str, prefixes: set[str]) -> str:
    # Quita atributos con prefijo no declarado:  pfx:attr="..."  o  pfx:attr='...'
    for p in prefixes:
        xml_text = re.sub(rf'\s{re.escape(p)}:[A-Za-z_][\w\-\.]*\s*=\s*"[^"]*"', '', xml_text)
        xml_text = re.sub(rf"\s{re.escape(p)}:[A-Za-z_][\w\-\.]*\s*=\s*'[^']*'", '', xml_text)
    return xml_text

def _normalize_root_prefix(xml_text: str) -> str:
    # Si la raíz es <kml:kml ...> o similar y NO está declarado xmlns:kml, lo inyecta.
    m = re.search(r'<\s*([A-Za-z_][\w\-\.]*):([A-Za-z_][\w\-\.]*)\b', xml_text)  # etiqueta de apertura con prefijo
    if m:
        pfx = m.group(1)
        decl = _declared_prefixes(xml_text)
        if pfx not in decl:
            uri = _COMMON_NS.get(pfx)
            if uri:
                xml_text = _inject_xmlns_in_root(xml_text, pfx, uri)
    return xml_text

def _parse_kml_with_fallback(raw_bytes: bytes, name: str = "?"):
    """
    Parse robusto para KML "sucios" con prefijos sin declarar:
    1) Intenta parseo directo.
    2) Si falla, normaliza raíz con prefijo (inyecta xmlns si es conocido).
    3) Inyecta xmlns para prefijos conocidos encontrados (kml, gx, atom, ...).
    4) Elimina prefijos NO declarados de etiquetas y elimina atributos con esos prefijos.
    """
    s = raw_bytes.decode("utf-8", errors="ignore")

    # 1) Intento directo
    try:
        return ET.fromstring(s)
    except ET.ParseError:
        pass

    # 2) Normaliza raíz con prefijo (ej. <kml:kml>)
    s = _normalize_root_prefix(s)
    try:
        return ET.fromstring(s)
    except ET.ParseError:
        pass

    # 3) Inyecta xmlns para prefijos comunes que se usen pero no estén declarados
    used = _used_tag_prefixes(s) | _used_attr_prefixes(s)
    decl = _declared_prefixes(s)
    missing_known = [p for p in used if p in _COMMON_NS and p not in decl]
    for p in missing_known:
        s = _inject_xmlns_in_root(s, p, _COMMON_NS[p])
    try:
        return ET.fromstring(s)
    except ET.ParseError:
        pass

    # 4) Quitar prefijos NO declarados de etiquetas y eliminar atributos con esos prefijos
    decl = _declared_prefixes(s)  # refresca
    used = _used_tag_prefixes(s) | _used_attr_prefixes(s)
    missing = {p for p in used if p not in decl}
    if missing:
        s = _strip_prefix_from_tags(s, missing)
        s = _remove_prefixed_attributes(s, missing)
        # También quita xmlns de prefijos que quitamos, por si había basura
        for p in missing:
            s = re.sub(rf'\sxmlns:{re.escape(p)}="[^"]*"', '', s)
        try:
            return ET.fromstring(s)
        except ET.ParseError as e:
            raise ValueError(f"KML inválido ({name}): {e}")

    # Si no había missing (raro), pero siguió fallando
    raise ValueError(f"KML inválido ({name}): no se pudo normalizar namespaces")


def _xy2(val):
    if val is None: return None
    if isinstance(val, (Vec2, Vec3)): return float(val.x), float(val.y)
    try: return float(val[0]), float(val[1])
    except Exception: return None

def compute_scale_1_to_n(doc, layout_name, offset_lonlat, epsg_override=None):
    """
    Devuelve (n, info) donde la escala es 1:n, calculada en UTM a partir
    del viewport principal del layout.
    """
    try:
        from pyproj import Transformer
    except Exception:
        print("[WARN] Falta pyproj (pip install pyproj) → no se puede calcular escala.")
        return None, {}

    if layout_name not in doc.layouts:
        print(f"[WARN] Layout '{layout_name}' no existe.")
        return None, {}

    layout = doc.layouts.get(layout_name)
    vps = [e for e in layout if e.dxftype() == "VIEWPORT"]
    if not vps:
        print("[WARN] Layout sin VIEWPORTs.")
        return None, {}

    # Viewport principal (mayor área)
    vp = max(vps, key=lambda v: float(v.dxf.width or 0) * float(v.dxf.height or 0))

    # Tamaño del rectángulo de viewport en papel (mm)
    wp = float(vp.dxf.width or 0.0)
    hp = float(vp.dxf.height or 0.0)
    if wp <= 0 or hp <= 0:
        print("[WARN] VIEWPORT sin tamaño.")
        return None, {}

    # Ventana en Model
    vcp = _xy2(getattr(vp.dxf, "view_center_point", None))
    vh  = float(getattr(vp.dxf, "view_height", 0.0) or 0.0)
    if vcp is None or vh <= 0.0:
        print("[WARN] VIEWPORT sin center/height.")
        return None, {}
    cxm, cym = vcp
    aspect = wp / hp
    vw     = vh * aspect

    xmin = cxm - vw/2.0; xmax = cxm + vw/2.0
    ymin = cym - vh/2.0; ymax = cym + vh/2.0

    # Pasar de coords locales (si SHIFT_TO_LOCAL=True) a lon/lat reales
    off_lon, off_lat = offset_lonlat
    # Usamos el centro para determinar zona UTM automáticamente (salvo override)
    lon_center = (xmin + xmax)/2.0 + off_lon
    lat_center = (ymin + ymax)/2.0 + off_lat

    if epsg_override:
        epsg   = int(epsg_override)
        zone   = None
        south  = lat_center < 0
    else:
        zone   = int((lon_center + 180.0)//6 + 1)
        south  = lat_center < 0
        epsg   = (32700 if south else 32600) + zone

    tr = Transformer.from_crs(4326, epsg, always_xy=True)

    # Proyectamos el rectángulo visible del MODEL a UTM (metros)
    lx, by = tr.transform(xmin + off_lon, ymin + off_lat)
    rx, ty = tr.transform(xmax + off_lon, ymax + off_lat)

    width_m  = abs(rx - lx)
    height_m = abs(ty - by)

    if width_m <= 0 or height_m <= 0:
        return None, {}

    # Escala 1:n (conservadora: usa el eje “más exigente”)
    n_w = (width_m  / wp) * 1000.0   # m/mm → *1000
    n_h = (height_m / hp) * 1000.0
    n   = int(max(n_w, n_h) + 0.5)   # redondeo al entero

    info = {"epsg": epsg, "zone": zone, "south": south,
            "width_m": width_m, "height_m": height_m,
            "wp_mm": wp, "hp_mm": hp}
    return n, info

import re

_ESC_RE   = re.compile(r"ESC\s*1\s*/\s*\d+", flags=re.IGNORECASE)
_ONLYVAL  = re.compile(r"^\s*1\s*/\s*\d+\s*$", flags=re.IGNORECASE)
_ESCALA_RE = re.compile(r"ESCALA", flags=re.IGNORECASE)  # para no tocar 'ESCALA:'

def apply_scale_text(doc, layout_name, scale_n):
    """
    Reemplaza de forma idempotente:
      - 'ESC 1/####' → 'ESC 1/<scale_n>'
      - Si un texto es EXACTAMENTE '1/####' → '1/<scale_n>'
    No toca nada que contenga 'ESCALA'.
    """
    if layout_name not in doc.layouts:
        return 0
    layout = doc.layouts.get(layout_name)

    new_full = f"ESC 1/{int(scale_n)}"
    new_val  = f"1/{int(scale_n)}"

    changed = 0

    def _patch_string(s):
        # No tocar rótulos “ESCALA: …”
        if _ESCALA_RE.search(s or ""):
            return s, False
        # 1) Sustituir solo la porción 'ESC 1/####' si existe
        if _ESC_RE.search(s or ""):
            s2 = _ESC_RE.sub(new_full, s)
            return s2, (s2 != s)
        # 2) Si el texto es exactamente '1/####', reemplazarlo
        if _ONLYVAL.match(s or ""):
            return new_val, True
        return s, False

    for e in layout:
        try:
            if e.dxftype() == "TEXT":
                s = e.dxf.text or ""
                s2, did = _patch_string(s)
                if did:
                    e.dxf.text = s2
                    changed += 1
            elif e.dxftype() == "MTEXT":
                s = e.text or ""
                s2, did = _patch_string(s)
                if did:
                    e.text = s2
                    changed += 1
        except Exception:
            pass

    # Atributos de bloques (INSERT/ATTRIB), muy común en cartelas
    try:
        for ins in layout.query("INSERT"):
            for att in getattr(ins, "attribs", []):
                s = att.dxf.text or ""
                s2, did = _patch_string(s)
                if did:
                    att.dxf.text = s2
                    changed += 1
    except Exception:
        pass

    return changed

import re

def set_zone_utm_in_layout(doc, layout_name, zone_text):
    """
    Reemplaza el token 'ZONA_UTM' por <zone_text> en:
      - TEXT (e.dxf.text)
      - MTEXT (e.text)
      - Atributos de INSERT (ATTRIB)
    Coincidencia insensible a mayúsculas y por substring.
    """
    token = "ZONA_UTM"
    pat = re.compile(re.escape(token), flags=re.IGNORECASE)

    if layout_name not in doc.layouts:
        return 0

    layout = doc.layouts.get(layout_name)
    changed = 0

    # TEXT / MTEXT en el Paper Space
    for e in layout:
        try:
            if e.dxftype() == "TEXT":
                s = e.dxf.text or ""
                s2 = pat.sub(zone_text, s)
                if s2 != s:
                    e.dxf.text = s2
                    changed += 1
            elif e.dxftype() == "MTEXT":
                s = e.text or ""
                s2 = pat.sub(zone_text, s)
                if s2 != s:
                    e.text = s2
                    changed += 1
        except Exception:
            pass

    # Atributos de bloques (muy típico en cartelas)
    try:
        for ins in layout.query("INSERT"):
            for att in getattr(ins, "attribs", []):
                s = att.dxf.text or ""
                s2 = pat.sub(zone_text, s)
                if s2 != s:
                    att.dxf.text = s2
                    changed += 1
    except Exception:
        pass

    return changed

# --- Escalas estándar ---
STD_SCALES = (500, 1000, 2000, 5000, 10000, 25000, 50000)

def _ensure_frame_viewport(doc, layout_name, frame_rect):
    """
    Garantiza que exista EXACTAMENTE 1 VIEWPORT que ocupe 'frame_rect' en papel.
    - Si ya hay viewports: reutiliza el mayor y AJÚSTALO al marco.
    - Si no hay: crea uno.
    NO crea un segundo viewport.
    """
    layout = doc.layouts.get(layout_name)

    # 1) Reutiliza el mayor si existe, o crea uno
    vps = [e for e in layout if e.dxftype() == "VIEWPORT"]
    if vps:
        vp = max(vps, key=lambda v: float(v.dxf.width or 0) * float(v.dxf.height or 0))
    else:
        # crea uno provisional; centraremos y escalaremos luego
        vp = layout.add_viewport(center=(210.0, 148.5), size=(200.0, 140.0))
        try: vp.dxf.status = 1
        except Exception: pass

    # 2) Ajusta ese MISMO viewport al rectángulo del marco
    x0, y0, x1, y1 = map(float, frame_rect)
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    size   = (x1 - x0, y1 - y0)
    vp.dxf.center = center
    vp.dxf.width  = size[0]
    vp.dxf.height = size[1]

    # (No tocamos view_center_point ni view_height aquí; eso lo hace apply_best_standard_scale)
    try: vp.dxf.status = 1
    except Exception: pass
    return vp



def apply_best_standard_scale(doc, layout_name, offset_lonlat, frame_rect,
                              standards=STD_SCALES, epsg_override=None):
    """
    1) Calcula la escala mínima necesaria (1:n_req) para que el bbox actual entre.
    2) Elige la escala estándar más cercana por arriba (la más grande que quepa).
    3) Ajusta el VIEWPORT para que esa escala sea exacta en el marco dado.
    4) Devuelve (n_elegida, info).
    """
    from pyproj import Transformer

    # Asegura un viewport exactamente del tamaño del marco
    vp = _ensure_frame_viewport(doc, layout_name, frame_rect)
    x0, y0, x1, y1 = map(float, frame_rect)
    wp, hp = (x1-x0), (y1-y0)  # mm de papel

    # Lee centro/altura actuales del model (si no hay, usa 0,0)
    try:
        cxm, cym = vp.dxf.view_center_point
    except Exception:
        cxm, cym = 0.0, 0.0

    # 1) escala mínima requerida con tu función existente
    n_req, info = compute_scale_1_to_n(doc, layout_name, offset_lonlat, epsg_override=epsg_override)
    if not n_req:
        n_req = standards[0]

    # 2) elige estándar
    choose = None
    for s in standards:
        if s >= n_req:
            choose = s
            break
    if choose is None:
        choose = standards[-1]

    # 3) fija el viewport a esa escala exacta (geom. en lon/lat → UTM → lon/lat)
    off_lon, off_lat = offset_lonlat
    lon_c = cxm + off_lon
    lat_c = cym + off_lat

    if epsg_override:
        epsg = int(epsg_override)
    else:
        zone  = int((lon_c + 180)//6 + 1)
        south = lat_c < 0
        epsg  = (32700 if south else 32600) + zone

    tr_fwd = Transformer.from_crs(4326, epsg, always_xy=True)   # lon/lat → UTM
    tr_inv = Transformer.from_crs(epsg, 4326, always_xy=True)   # UTM → lon/lat

    # ancho/alto reales que debe “ver” el viewport (en metros)
    width_m  = (wp/1000.0) * choose
    height_m = (hp/1000.0) * choose

    # construye rectángulo UTM alrededor del centro
    Ex, Ny   = tr_fwd.transform(lon_c, lat_c)
    Ex0, Ex1 = Ex - width_m/2.0,  Ex + width_m/2.0
    Ny0, Ny1 = Ny - height_m/2.0, Ny + height_m/2.0

    # vuelve a lon/lat para obtener altura en grados (AutoCAD usa “view_height”)
    lon_bot, lat_bot = tr_inv.transform(Ex,  Ny0)  # centro abajo
    lon_top, lat_top = tr_inv.transform(Ex,  Ny1)  # centro arriba
    view_h_deg = max(lat_top - lat_bot, 1e-9)

    # fija centro y altura (en coords del MODEL: restando el offset)
    vp.dxf.view_center_point = (lon_c - off_lon, lat_c - off_lat)
    vp.dxf.view_height       = view_h_deg

    # Asegura capas ON
    try:
        for lname in ("LINES","POLYGONS"):
            L = doc.layers.get(lname); L.on(); L.thaw(); L.unlock()
            if hasattr(vp,"is_layer_frozen") and vp.is_layer_frozen(lname):
                vp.thaw_layer(lname)
    except Exception:
        pass

    return choose, info

# ───────── MAIN ─────────
def main():
    if not os.path.exists(KMZ_NAME):
        print(f"[ERROR] Falta {KMZ_NAME}")
        return
    if not os.path.exists(TEMPLATE_DXF_IN):
        print(f"[ERROR] Falta {TEMPLATE_DXF_IN} (guarda tu plantilla como AutoCAD 2018 DXF)")
        return

    print("[INFO] Leyendo KMZ con colores…")
    lines_col, polys_col = read_geometries_wgs84_colored(KMZ_NAME)
    if not lines_col and not polys_col:
        print("[ERROR] KMZ sin geometrías utilizables.")
        return

    # BBOX WGS84 base (en lon/lat) para referencia
    lons = [x for (pts, _rgb) in (lines_col + polys_col) for x, y in pts]
    lats = [y for (pts, _rgb) in (lines_col + polys_col) for x, y in pts]
    bbox84 = bbox_wgs84(lons, lats, margin_ratio=0.06)

    # 1) Elegir sistema para insertar en la PLANTILLA:
    if SHIFT_TO_LOCAL:
        lines_ins, polys_ins, bbox_ins, offset = shift_to_local_coords_colored(lines_col, polys_col)
        print(f"[INFO] Normalizado al origen (offset lon/lat = {offset})")
    else:
        lines_ins, polys_ins, bbox_ins, offset = lines_col, polys_col, bbox84, (0.0, 0.0)

    # 2) Plantilla + reemplazo Model (respetando color)
    print("[INFO] Abriendo plantilla DXF…")
    doc = ezdxf.readfile(TEMPLATE_DXF_IN)

    # Lee meta desde ./meta.json o ENV; si no hay, no rompe
    meta = load_admin_meta(".")
    if any(meta.values()):
        print(f"[INFO] Meta recibida: UBIGEO={meta.get('ubigeo')}, "
            f"DEP={meta.get('departamento')}, PROV={meta.get('provincia')}, DIST={meta.get('distrito')}")
    else:
        print("[INFO] Sin meta (ejecución local). Se usarán valores 'Pendiente' en cartela.")

    replace_placeholders(doc, meta)  # <-- ahora con meta
    replace_model_content(doc, lines_ins, polys_ins, bbox_ins)


    # 3) Centrar la vista del Model (zoom extents portátil)
    center_model_view(doc, bbox_ins, pad=1.02)

    # 4) Encajar el VIEWPORT exactamente en el marco útil
    fit_layout_viewport(
        doc, LAYOUT_NAME, bbox_ins,
        occupancy=0.58, safety=1.00,
        frame_rect=PAPER_FRAME_RECT  # (xmin, ymin, xmax, ymax) en PAPEL
    )

    # 5) Buscar la mejor escala ESTÁNDAR que quepa y aplicarla al viewport
    best_n, info = apply_best_standard_scale(
        doc, LAYOUT_NAME, offset, frame_rect=PAPER_FRAME_RECT
    )
    print(f"[INFO] Escala estándar seleccionada: 1:{best_n}")

    # 6) Dibujar 4x4 líneas y rótulos UTM dentro del marco
    upsert_utm_4lines(doc, LAYOUT_NAME, offset, frame_rect=PAPER_FRAME_RECT)

    # 7) Sustituir ZONA_UTM en la cartela usando la zona detectada/derivada
    zone = info.get("zone"); south = info.get("south"); epsg = info.get("epsg")
    if zone is None and epsg:
        if 32601 <= epsg <= 32660: zone, south = epsg - 32600, False
        elif 32701 <= epsg <= 32760: zone, south = epsg - 32700, True
    zone_text = f"{zone}{'S' if south else 'N'}" if zone else "?"
    set_zone_utm_in_layout(doc, LAYOUT_NAME, zone_text)
    print(f"[INFO] Zona UTM: {zone_text} (EPSG:{epsg})")

    # 8) Escribir la escala en textos 'ESC 1/####' (solo una vez; no toca 'ESCALA:')
    cnt = apply_scale_text(doc, LAYOUT_NAME, best_n)
    print(f"[INFO] Textos 'ESC' actualizados: {cnt}")

 
    # 10) Reabrir solo para exportar PDF tal cual se ve el layout
    doc.saveas(FINAL_DXF_OUT)
    doc = ezdxf.readfile(FINAL_DXF_OUT)
    fit_layout_viewport(
        doc, LAYOUT_NAME, bbox_ins,
        occupancy=0.58, safety=1.00,
        frame_rect=PAPER_FRAME_RECT
    )
    upsert_utm_4lines(doc, LAYOUT_NAME, offset, frame_rect=PAPER_FRAME_RECT)
    set_zone_utm_in_layout(doc, LAYOUT_NAME, zone_text)  # idempotente

    export_layout_pdf_viewport(doc, LAYOUT_NAME, PDF_OUT)
    print(f"[OK] PDF: {PDF_OUT}")
    print("[OK] Proceso completo.")

    # 9) Guardar DXF final
    doc.saveas(FINAL_DXF_OUT)
    print(f"[OK] Plantilla final: {FINAL_DXF_OUT}")



if __name__ == "__main__":
    main()
