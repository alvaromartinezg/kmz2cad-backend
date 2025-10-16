#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===== Configuración estándar (no toques nada) =====
INPUT_NAME_KMZ = "TEST.kmz"
INPUT_NAME_KML = "TEST.kml"
OUTPUT_NAME    = "Exportado.kmz"
CLOSE_THRESHOLD_M = 30.0    # cerrar lineas a polígono si inicio-fin <= 30 m
NEAR_M            = 60.0   # radio de selección/recorte desde el polígono o línea
DENSIFY_STEP_M    = 30.0     # paso de muestreo para recorte (balance precisión/velocidad)

import os, sys, zipfile, math, re, xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError

# Trabajar siempre en la carpeta del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
print(f"[INFO] Carpeta de trabajo: {os.getcwd()}")

# Buscar input TEST y base Transmission Network
def find_input_path():
    if os.path.exists(INPUT_NAME_KMZ): return INPUT_NAME_KMZ
    if os.path.exists(INPUT_NAME_KML): return INPUT_NAME_KML
    return None

def find_base_kmz():
    cands = [f for f in os.listdir(".")
             if f.lower().endswith(".kmz")
             and "transmission" in f.lower()
             and "network" in f.lower()
             and "canaliz" not in f.lower()]
    if not cands: return None
    # elige el más grande (suele ser el correcto)
    cands.sort(key=lambda x: os.path.getsize(x), reverse=True)
    return cands[0]

def find_canalizado_kmz():
    # acepta cualquier KMZ que tenga transmission + network + canaliz*
    cands = [f for f in os.listdir(".")
             if f.lower().endswith(".kmz")
             and "transmission" in f.lower()
             and "network" in f.lower()
             and "canaliz" in f.lower()]
    # y también acepta el nombre estándar que deja main.py en /tmp
    if os.path.exists("Transmission Network Canalizado.kmz"):
        cands.append("Transmission Network Canalizado.kmz")

    # elimina duplicados
    cands = list({c.lower(): c for c in cands}.values())
    if not cands:
        return None
    cands.sort(key=lambda x: os.path.getsize(x), reverse=True)
    return cands[0]

NS = {"kml": "http://www.opengis.net/kml/2.2"}
for p,u in NS.items():
    ET.register_namespace(p if p!="kml" else "", u)

# ----------------- Utilidades geoespaciales -----------------
def haversine_m(p1, p2):
    lon1, lat1 = p1; lon2, lat2 = p2
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def equirect_xy(lon,lat,lon0,lat0):
    R=6371000.0
    x=math.radians(lon-lon0)*R*math.cos(math.radians(lat0))
    y=math.radians(lat-lat0)*R
    return (x,y)

def inv_equirect_xy(x,y,lon0,lat0):
    R=6371000.0
    lon = lon0 + math.degrees(x/(R*math.cos(math.radians(lat0))))
    lat = lat0 + math.degrees(y/R)
    return (lon,lat)

def point_in_poly(pt, ring):
    x,y=pt; inside=False
    n=len(ring)
    for i in range(n):
        x1,y1=ring[i]; x2,y2=ring[(i+1)%n]
        if ((y1>y)!=(y2>y)):
            xint=x1+(y-y1)*(x2-x1)/(y2-y1)
            if xint>x: inside=not inside
    return inside

def dist_pt_seg(p,a,b):
    (px,py),(ax,ay),(bx,by)=p,a,b
    vx,vy=bx-ax,by-ay; wx,wy=px-ax,py-ay
    v2=vx*vx+vy*vy
    if v2==0: return math.hypot(px-ax,py-ay)
    t=max(0,min(1,(wx*vx+wy*vy)/v2))
    projx,projy=ax+t*vx,ay+t*vy
    return math.hypot(px-projx,py-projy)

def dist_pt_poly(p, ring):
    if point_in_poly(p, ring): return 0.0
    dmin=float("inf")
    n=len(ring)
    for i in range(n):
        a=ring[i]; b=ring[(i+1)%n]
        d=dist_pt_seg(p,a,b)
        if d<dmin: dmin=d
    return dmin

def bbox_pts_xy(pts_xy):
    xs=[x for x,y in pts_xy]; ys=[y for x,y in pts_xy]
    return (min(xs),min(ys),max(xs),max(ys))

def bbox_expand(b, pad):
    x1,y1,x2,y2=b; return (x1-pad,y1-pad,x2+pad,y2+pad)

def bbox_overlap(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


# ======== NUEVO: distancia punto ↔ polilínea (lista de vértices) ========
def dist_pt_polyline(p, line_xy):
    dmin = float("inf")
    for i in range(len(line_xy)-1):
        d = dist_pt_seg(p, line_xy[i], line_xy[i+1])
        if d < dmin: dmin = d
    return dmin

# ----------------- Lectura/parseo KML/KMZ -----------------
def parse_coords(text):
    out=[]
    for tok in (text or "").strip().split():
        parts=tok.split(",")
        if len(parts)>=2:
            lon=float(parts[0]); lat=float(parts[1])
            alt=float(parts[2]) if len(parts)>2 and parts[2]!="" else 0.0
            out.append((lon,lat,alt))
    return out

def coords_to_text(coords):
    return " ".join(f"{lon},{lat},{alt:g}" for lon,lat,alt in coords)

def safe_parse_kml(data: bytes):
    # Parser robusto: agrega xmlns:* faltantes y evita prefijos reservados
    try:
        return ET.fromstring(data)
    except ParseError:
        text = data.decode("utf-8", errors="replace")
        m = re.search(r"<\s*kml\b[^>]*>", text, flags=re.IGNORECASE|re.DOTALL)
        if not m: raise
        open_tag = m.group(0)

        declared = set(re.findall(r'xmlns:([A-Za-z_][\w\-.]*)=', open_tag))
        used_in_tags  = set(re.findall(r'</?\s*([A-Za-z_][\w\-.]*):[A-Za-z_][\w\-.]*', text))
        used_in_attrs = set(re.findall(r'\s(?!xmlns:)([A-Za-z_][\w\-.]*):[A-Za-z_][\w\-.]*=', text))
        used = used_in_tags | used_in_attrs
        skip = {"kml","xml","xmlns"}

        # comunes
        if "gx" in used and "gx" not in declared:
            open_tag = open_tag[:-1] + ' xmlns:gx="http://www.google.com/kml/ext/2.2">'
            declared.add("gx")
        if "atom" in used and "atom" not in declared:
            open_tag = open_tag[:-1] + ' xmlns:atom="http://www.w3.org/2005/Atom">'
            declared.add("atom")
        # el resto dummy
        for pref in sorted(used):
            if pref in skip or pref in declared: continue
            open_tag = open_tag[:-1] + f' xmlns:{pref}="urn:autofix:{pref}">'
            declared.add(pref)

        fixed = text[:m.start()] + open_tag + text[m.end():]
        try:
            return ET.fromstring(fixed.encode("utf-8"))
        except ParseError:
            # Plan B: eliminar prefijos problemáticos
            fixed = re.sub(r'</?\s*([A-Za-z_][\w\-.]*):', lambda mo: mo.group(0).replace(mo.group(1)+":",""), fixed)
            fixed = re.sub(r'\s(?!xmlns:)([A-Za-z_][\w\-.]*):([A-Za-z_][\w\-.]*)=', r' \2=', fixed)
            return ET.fromstring(fixed.encode("utf-8"))

def read_kml_root(path):
    if path.lower().endswith(".kmz"):
        with zipfile.ZipFile(path,"r") as zf:
            kmls=[n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kmls: raise FileNotFoundError("KMZ sin KML interno")
            chosen=next((n for n in kmls if os.path.basename(n).lower()=="doc.kml"), kmls[0])
            data=zf.read(chosen)
            return safe_parse_kml(data)
    else:
        with open(path,"rb") as f:
            return safe_parse_kml(f.read())

# Polígonos desde input (usa outerBoundary; convierte líneas cerrables)
def polygons_from_input(path):
    root = read_kml_root(path)
    polys=[]

    # 1) Tomar polígonos existentes (outerBoundary)
    for lr in root.findall(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing", NS):
        coords_el = lr.find("kml:coordinates", NS)
        if coords_el is None: continue
        pts = [(lon,lat) for lon,lat,_ in parse_coords(coords_el.text)]
        if len(pts)>=3:
            if pts[0]!=pts[-1]: pts.append(pts[0])
            if len(pts)>=4: polys.append(pts)

    # 2) Convertir LineString cerrables (inicio-fin <= CLOSE_THRESHOLD_M) a polígono
    for ls in root.findall(".//kml:LineString", NS):
        coords_el = ls.find("kml:coordinates", NS)
        if coords_el is None: continue
        pts3 = parse_coords(coords_el.text)
        pts  = [(lon,lat) for lon,lat,_ in pts3]
        if len(pts) >= 2:
            d = haversine_m(pts[0], pts[-1])
            if d <= CLOSE_THRESHOLD_M:
                ring = list(pts)
                if ring[0]!=ring[-1]: ring.append(ring[0])
                if len(ring)>=4: polys.append(ring)

    return polys

# ======== NUEVO: Leer LÍNEAS desde el input (sin exigir 3 vértices) ========
def read_lines_from_input(path):
    roots = read_all_kml_roots(path)
    lines=[]
    for root in roots:
        for pm in root.findall(".//kml:Placemark", NS):
            ls = pm.find(".//kml:LineString", NS)
            if ls is None: continue
            ce = ls.find("kml:coordinates", NS)
            if ce is None: continue
            pts3 = parse_coords(ce.text)
            pts  = [(lon,lat) for lon,lat,_ in pts3]
            if len(pts) < 2:
                continue
            name_el=pm.find("kml:name", NS)
            name=name_el.text if name_el is not None else "linea_entrada"
            lines.append((name, pts))
    return lines

def read_polygons_only_from_input(path):
    roots = read_all_kml_roots(path)
    polys=[]
    for root in roots:
        for lr in root.findall(".//kml:Polygon/kml:outerBoundaryIs/kml:LinearRing", NS):
            coords_el = lr.find("kml:coordinates", NS)
            if coords_el is None: continue
            pts = [(lon,lat) for lon,lat,_ in parse_coords(coords_el.text)]
            if len(pts)>=3:
                if pts[0]!=pts[-1]: pts.append(pts[0])
                if len(pts)>=4: polys.append(pts)
    return polys

def close_lines_to_polys(lines, threshold_m):
    polys=[]
    for _name, pts in lines:
        if len(pts) >= 2:
            d = haversine_m(pts[0], pts[-1])
            if d <= threshold_m:
                ring = list(pts)
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                if len(ring) >= 4:
                    polys.append(ring)
    return polys


# Leer TODAS las LineString del KMZ base (sin importar carpetas)
def read_lines_from_kmz(kmz_path):
    with zipfile.ZipFile(kmz_path,"r") as zf:
        kmls=[n for n in zf.namelist() if n.lower().endswith(".kml")]
        if not kmls: return []
        chosen=next((n for n in kmls if os.path.basename(n).lower()=="doc.kml"), kmls[0])
        data=zf.read(chosen)
    root=safe_parse_kml(data)
    lines=[]
    for pm in root.findall(".//kml:Placemark", NS):
        ls = pm.find(".//kml:LineString", NS)
        if ls is None: continue
        ce = ls.find("kml:coordinates", NS)
        if ce is None: continue
        pts3 = parse_coords(ce.text)
        pts  = [(lon,lat) for lon,lat,_ in pts3]

        # ⛔ Excluir microondas: solo si el KML trae exactamente 2 puntos
        if len(pts) == 2:
            continue

        name_el = pm.find("kml:name", NS)
        name = name_el.text if name_el is not None else "sin_nombre"
        lines.append((name, pts))

    return lines

# -------------- Densificar + recorte por buffer --------------
def densify_line_lonlat_window(pts, fwd, inv, step_far, step_near, window_xy, ring_bboxes):
    """
    Densifica con paso grande (step_far) salvo cuando el segmento toca
    la ventana global o alguno de los bboxes de los anillos, donde usa
    step_near. Soporta window_xy=None y ring_bboxes=[].
    """
    out=[]
    for i in range(len(pts)-1):
        lon1,lat1=pts[i]; lon2,lat2=pts[i+1]
        x1,y1=fwd(lon1,lat1); x2,y2=fwd(lon2,lat2)
        seg_bb=(min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2))

        # --- Guardas: permite window_xy=None y ring_bboxes vacía
        overlaps_window = True if window_xy is None else bbox_overlap(seg_bb, window_xy)
        overlaps_any_ring = any(bbox_overlap(seg_bb, rb) for rb in (ring_bboxes or []))

        # Si no hay window ni ring_bboxes, nunca "toca"; usa step_far (en refline pasas far=near)
        touch = overlaps_window and (not ring_bboxes or overlaps_any_ring)

        step = step_near if touch else step_far
        dx,dy=x2-x1,y2-y1
        L=(dx*dx+dy*dy)**0.5

        if not out: out.append((lon1,lat1))
        if L>0:
            n = int(L//step)
            for k in range(1, n+1):
                t=(k*step)/L
                if t>=1: break
                out.append(inv(x1+t*dx, y1+t*dy))
        out.append((lon2,lat2))
    return out

def _equirect_funcs(lon0, lat0):
    R=6371000.0; c=math.cos(math.radians(lat0))
    def fwd(lon,lat):  return (math.radians(lon-lon0)*R*c, math.radians(lat-lat0)*R)
    def inv(x,y):      return (lon0+math.degrees(x/(R*c)), lat0+math.degrees(y/R))
    return fwd,inv

def clip_line_by_polygons(pts, polygons, near_m):
    """
    Recorta una línea devolviendo subtramos donde
    distancia(polígono) <= near_m o está dentro.
    Optimizado: proyección cacheada, bbox global/anillo, densificado adaptativo.
    """
    if not polygons or len(pts) < 2:
        return []

    # Centro de proyección
    lons = [lon for ring in polygons for lon, lat in ring]
    lats = [lat for ring in polygons for lon, lat in ring]
    lon0 = sum(lons) / len(lons); lat0 = sum(lats) / len(lats)
    fwd, inv = _equirect_funcs(lon0, lat0)

    # Anillos proyectados + bboxes
    rings_xy = [[fwd(lon, lat) for lon, lat in ring] for ring in polygons]
    ring_bbs = [bbox_expand(bbox_pts_xy(r), near_m) for r in rings_xy]

    # Ventana global (bbox unión)
    gx1 = min(b[0] for b in ring_bbs); gy1 = min(b[1] for b in ring_bbs)
    gx2 = max(b[2] for b in ring_bbs); gy2 = max(b[3] for b in ring_bbs)
    window = (gx1, gy1, gx2, gy2)

    # Rechazo grosero (solo extremos)
    x1, y1 = fwd(*pts[0]); x2, y2 = fwd(*pts[-1])
    line_bb = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    if not bbox_overlap(line_bb, window):
        return []

    # Densificar SOLO donde importa (adaptativo)
    dense = densify_line_lonlat_window(
        pts, fwd, inv,
        step_far=max(24.0, DENSIFY_STEP_M * 1.5),  # p.ej. 24–40 m
        step_near=DENSIFY_STEP_M,                  # tu valor actual (30 m)
        window_xy=window,
        ring_bboxes=ring_bbs
    )
    if len(dense) < 2:
        return []

    # ⬅️ FALTABA ESTA LÍNEA:
    dense_xy = [fwd(lon, lat) for lon, lat in dense]

    # Clasificado rápido (bbox por anillo → corta cálculos caros)
    mask = []
    for (x, y) in dense_xy:
        keep = False
        for ring, bb in zip(rings_xy, ring_bbs):
            if not (bb[0] <= x <= bb[2] and bb[1] <= y <= bb[3]):
                continue
            if point_in_poly((x, y), ring) or dist_pt_poly((x, y), ring) <= near_m:
                keep = True
                break
        mask.append(keep)

    # Agrupar por segmentos (usa extremos y punto medio)
    segments, cur = [], []
    for i in range(len(dense) - 1):
        mx = (dense_xy[i][0] + dense_xy[i + 1][0]) * 0.5
        my = (dense_xy[i][1] + dense_xy[i + 1][1]) * 0.5

        mid_keep = False
        for ring, bb in zip(rings_xy, ring_bbs):
            if not (bb[0] <= mx <= bb[2] and bb[1] <= my <= bb[3]):
                continue
            if point_in_poly((mx, my), ring) or dist_pt_poly((mx, my), ring) <= near_m:
                mid_keep = True
                break

        use = mask[i] or mask[i + 1] or mid_keep
        if use:
            if not cur:
                cur.append(dense[i])
            cur.append(dense[i + 1])
        else:
            if cur and len(cur) >= 2:
                segments.append(cur)
            cur = []

    if cur and len(cur) >= 2:
        segments.append(cur)

    return segments


# ======== NUEVO: Recorte por LÍNEA de referencia ========
def clip_line_by_refline(pts, ref_pts, near_m):
    """
    Recorta una línea devolviendo subtramos donde distancia a la línea de
    referencia <= near_m.
    """
    # Centro de proyección: basado en la línea de referencia
    lon0 = sum(lon for lon, lat in ref_pts) / len(ref_pts)
    lat0 = sum(lat for lon, lat in ref_pts) / len(ref_pts)
    fwd, inv = _equirect_funcs(lon0, lat0)

    ref_xy = [fwd(lon, lat) for lon, lat in ref_pts]

    # Densificar candidato (usa la misma función; pasos iguales y sin ventana)
    dense = densify_line_lonlat_window(
        pts, fwd, inv,
        step_far=DENSIFY_STEP_M,
        step_near=DENSIFY_STEP_M,
        window_xy=None,
        ring_bboxes=[]
    )
    if len(dense) < 2:
        return []

    dense_xy = [fwd(lon, lat) for lon, lat in dense]

    # Clasificar puntos por distancia a la polilínea de referencia
    mask = []
    for (x, y) in dense_xy:
        keep = dist_pt_polyline((x, y), ref_xy) <= near_m
        mask.append(keep)

    # Subtramos contiguos True
    segments = []
    cur = []
    for idx, keep in enumerate(mask):
        if keep:
            cur.append(dense[idx])
        elif cur:
            if len(cur) >= 2:
                segments.append(cur)
            cur = []
    if cur and len(cur) >= 2:
        segments.append(cur)

    return segments


# -------------- Filtrado (clip) de todas las líneas --------------
def filter_and_clip_lines(lines, polygons, near_m):
    selected=[]
    for name, pts in lines:
        segs = clip_line_by_polygons(pts, polygons, near_m)
        for j, seg in enumerate(segs, start=1):
            out_name = name if len(segs)==1 else f"{name} (parte {j})"
            selected.append((out_name, seg))
    return selected

# ======== NUEVO: Filtrar/recortar por LÍNEAS de referencia ========
def filter_and_clip_lines_near_ref(lines, ref_lines, near_m):
    selected=[]
    for name, pts in lines:
        parts=[]
        for ref_name, ref_pts in ref_lines:
            segs = clip_line_by_refline(pts, ref_pts, near_m)
            parts.extend(segs)
        for j, seg in enumerate(parts, start=1):
            out_name = name if len(parts)==1 else f"{name} (parte {j})"
            selected.append((out_name, seg))
    return selected

# -------------- Exportar KMZ con estilos y polígonos --------------
def write_kmz(lines, polygons, out_path, highlight_lines=None, canalizado_lines=None):
    # Estilos:
    kml = ET.Element("kml", xmlns=NS["kml"])
    doc = ET.SubElement(kml,"Document")

    # Estilo líneas (azules)
    st_line = ET.SubElement(doc, "Style", id="lineBlue")
    ls = ET.SubElement(st_line, "LineStyle")
    ET.SubElement(ls, "color").text = "ffff0000"
    ET.SubElement(ls, "width").text = "3"

    # Estilo polígonos (fucsia 50%)
    st_poly = ET.SubElement(doc, "Style", id="polyFuchsia")
    pls = ET.SubElement(st_poly, "PolyStyle")
    ET.SubElement(pls, "color").text = "80FF00FF"  # 50% fucsia
    lsp = ET.SubElement(st_poly, "LineStyle")
    ET.SubElement(lsp, "color").text = "ffFF00FF"
    ET.SubElement(lsp, "width").text = "2"

    # NUEVO: Estilo línea fucsia grosor 10
    st_line_fx = ET.SubElement(doc, "Style", id="lineFuchsia10")
    lsf = ET.SubElement(st_line_fx, "LineStyle")
    ET.SubElement(lsf, "color").text = "80FF00FF"
    ET.SubElement(lsf, "width").text = "10"

    #Estilo linea canalizado: Verde
    st_line_green = ET.SubElement(doc, "Style", id="lineGreen")
    lsg = ET.SubElement(st_line_green, "LineStyle")
    ET.SubElement(lsg, "color").text = "ff00ff00"  # verde (aabbggrr)
    ET.SubElement(lsg, "width").text = "3"

    # Polígonos (si hubiera)
    for idx, ring in enumerate(polygons, start=1):
        pm = ET.SubElement(doc, "Placemark")
        ET.SubElement(pm,"name").text = "Area de impacto"
        ET.SubElement(pm,"styleUrl").text = "#polyFuchsia"
        poly = ET.SubElement(pm,"Polygon")
        obi  = ET.SubElement(poly,"outerBoundaryIs")
        lr   = ET.SubElement(obi,"LinearRing")
        ET.SubElement(lr,"coordinates").text = coords_to_text([(lon,lat,0.0) for lon,lat in ring])

    # NUEVO: Dibujar línea(s) de entrada destacadas (fucsia 10)
    if highlight_lines:
        for name, pts in highlight_lines:
            pm = ET.SubElement(doc,"Placemark")
            ET.SubElement(pm,"name").text = "Area de impacto"
            ET.SubElement(pm,"styleUrl").text = "#lineFuchsia10"
            ls = ET.SubElement(pm,"LineString")
            ET.SubElement(ls,"coordinates").text = coords_to_text([(lon,lat,0.0) for lon,lat in pts])

        # NUEVO: Líneas recortadas CANALIZADAS (verdes)
    if canalizado_lines:
        for idx, (_orig_name, pts) in enumerate(canalizado_lines, start=1):
            pm = ET.SubElement(doc,"Placemark")
            ET.SubElement(pm,"name").text = f"ruta canalizada {idx}"
            ET.SubElement(pm,"styleUrl").text = "#lineGreen"
            ls = ET.SubElement(pm,"LineString")
            ET.SubElement(ls,"coordinates").text = coords_to_text([(lon,lat,0.0) for lon,lat in pts])

    # Líneas recortadas (seleccionadas)
    for idx, (_orig_name, pts) in enumerate(lines, start=1):
        pm = ET.SubElement(doc,"Placemark")
        ET.SubElement(pm,"name").text = f"ruta {idx}"
        ET.SubElement(pm,"styleUrl").text = "#lineBlue"
        ls = ET.SubElement(pm,"LineString")
        ET.SubElement(ls,"coordinates").text = coords_to_text([(lon,lat,0.0) for lon,lat in pts])


    kml_bytes = ET.tostring(kml, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
     zf.writestr("doc.kml", kml_bytes)


def _unique_vertices_count(pts):
    return len({(round(lon, 7), round(lat, 7)) for lon, lat in pts})

def read_all_kml_roots(path):
    """Devuelve una lista de roots KML. Si es KMZ, incluye *todos* los .kml internos."""
    roots = []
    if path.lower().endswith(".kmz"):
        with zipfile.ZipFile(path, "r") as zf:
            kmls = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kmls:
                raise FileNotFoundError("KMZ sin KML interno")
            for name in kmls:
                roots.append(safe_parse_kml(zf.read(name)))
    else:
        with open(path, "rb") as f:
            roots.append(safe_parse_kml(f.read()))
    return roots

# =========================== MAIN ===========================
def main():
    input_path = find_input_path()
    if not input_path:
        print("No se encontró TEST.kmz ni TEST.kml en esta carpeta."); sys.exit(0)
    base_kmz = find_base_kmz()
    if not base_kmz:
        print("No se encontró ningún KMZ tipo 'Transmission Network' en esta carpeta."); sys.exit(0)

    base_kmz_canal = find_canalizado_kmz()

    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Base:  {base_kmz}")

    if base_kmz_canal:
        print(f"[INFO] Base CANALIZADA: {base_kmz_canal}")

    # 1) Leer TODO del input (en todos los .kml internos):
    ref_lines      = read_lines_from_input(input_path)               # todas las LineString del TEST
    closed_polys   = close_lines_to_polys(ref_lines, CLOSE_THRESHOLD_M)  # polígonos creados desde líneas (≤50 m)
    existing_polys = read_polygons_only_from_input(input_path)       # polígonos que ya vienen en TEST

    print(f"[INFO] Input trae: lineas={len(ref_lines)} | poligonos_existentes={len(existing_polys)} | lineas_cerrables→poligonos={len(closed_polys)}")

    # 2) Usar SIEMPRE todas las áreas de impacto disponibles:
    #    - Unimos los polígonos existentes + los generados desde líneas cerrables.
    polys = []
    polys.extend(existing_polys)
    polys.extend(closed_polys)

    if polys:
        print(f"[INFO] Usando {len(polys)} área(s) de impacto (existentes + derivadas de líneas).")
        print("[INFO] Leyendo líneas del KMZ base…")
        lines = read_lines_from_kmz(base_kmz)   # anti-microondas activo
        print(f"[INFO] Total líneas en base: {len(lines)}")

        print(f"[INFO] Filtrando y recortando a ≤{NEAR_M} m del(los) área(s) de impacto…")
        clipped_poly = filter_and_clip_lines(lines, polys, NEAR_M)
        clipped_ref  = filter_and_clip_lines_near_ref(lines, ref_lines, NEAR_M) if ref_lines else []
        clipped      = clipped_poly + clipped_ref
        print(f"[OK] Tramos seleccionados (base): {len(clipped)}")

        # NUEVO: procesar base canalizada si existe
        clipped_canal = []
        if base_kmz_canal:
            print("[INFO] Leyendo líneas del KMZ base CANALIZADA…")
            lines_canal = read_lines_from_kmz(base_kmz_canal)
            print(f"[INFO] Total líneas en base canalizada: {len(lines_canal)}")
            clipped_poly_c = filter_and_clip_lines(lines_canal, polys, NEAR_M)
            clipped_ref_c  = filter_and_clip_lines_near_ref(lines_canal, ref_lines, NEAR_M) if ref_lines else []
            clipped_canal  = clipped_poly_c + clipped_ref_c
            print(f"[OK] Tramos seleccionados (canalizada): {len(clipped_canal)}")

        # Exportar ambas capas (negra + verde)
        write_kmz(
            clipped, polys, OUTPUT_NAME,
            highlight_lines=ref_lines if ref_lines else None,
            canalizado_lines=clipped_canal if clipped_canal else None
        )
        print(f"[OK] Exportado: {OUTPUT_NAME}")
        return

    # 3) Si no hay polígonos de ningún tipo, usar la(s) LÍNEA(s) de entrada como referencia (fallback)
    if ref_lines:
        print("[AVISO] No hay polígonos; usaré la(s) LÍNEA(s) de entrada como referencia.")
        print("[INFO] Leyendo líneas del KMZ base…")
        lines = read_lines_from_kmz(base_kmz)
        print(f"[INFO] Total líneas en base: {len(lines)}")

        print(f"[INFO] Filtrando y recortando a ≤{NEAR_M} m de la(s) línea(s) de entrada…")
        clipped = filter_and_clip_lines_near_ref(lines, ref_lines, NEAR_M)
        print(f"[OK] Tramos seleccionados (base): {len(clipped)}")

        # NUEVO: procesar base canalizada si existe
        clipped_canal = []
        if base_kmz_canal:
            print("[INFO] Leyendo líneas del KMZ base CANALIZADA…")
            lines_canal = read_lines_from_kmz(base_kmz_canal)
            print(f"[INFO] Total líneas en base canalizada: {len(lines_canal)}")
            clipped_canal = filter_and_clip_lines_near_ref(lines_canal, ref_lines, NEAR_M)
            print(f"[OK] Tramos seleccionados (canalizada): {len(clipped_canal)}")

        # Exporta: highlight de entrada + capas recortadas
        write_kmz(
            clipped, [], OUTPUT_NAME,
            highlight_lines=ref_lines,
            canalizado_lines=clipped_canal if clipped_canal else None
        )
        print(f"[OK] Exportado: {OUTPUT_NAME}")
        return


    print("[ERROR] El input no contiene polígonos ni líneas utilizables."); sys.exit(0)


if __name__=="__main__":
    main()
