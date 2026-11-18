"""
AgroScan - Dashboard interactivo con navegación lateral (Streamlit)
Tema: Dark Neon Purple (inspirado en dashboard de referencia).

Ejecutar:  python -m streamlit run app.py
Datos:     carpeta data/ con los CSV exportados de PostgreSQL (db_detector_maiz).
Opcional:  logo.png junto a app.py para usar tu propio logotipo.
"""
import base64
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="AgroScan · Dashboard",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===== CONFIGURACIÓN =====
NOMBRE = "AgroScan"
SUBTITULO = "Detector de gusano cogollero y roya del maíz"
BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
LOGO = BASE / "logo.png"

UMBRAL = 80
META_EXACTITUD = 90
COLUMNAS_GRAFICAS = 2
ALTO = 360

# (clave, icono, etiqueta, título de página)
PAGINAS = [
    ("inicio",     "⌂", "Inicio",                    "Inicio"),
    ("r1",         "◈", "R1 · Exactitud",            "R1 · Exactitud por clase y mes"),
    ("r2",         "◎", "R2 · Incidencia",           "R2 · Incidencia por parcela"),
    ("r3",         "⚠", "R3 · Confianza baja",       "R3 · Escaneos con confianza baja"),
    ("r4",         "⊞", "R4 · Matriz de confusión",  "R4 · Matriz de confusión"),
    ("r5",         "↗", "R5 · Evolución",            "R5 · Evolución temporal"),
    ("r6",         "⚙", "R6 · Algoritmos",           "R6 · Algoritmos y versiones"),
    ("decisiones", "◉", "Toma de decisiones",        "Toma de decisiones"),
]

# ===== Paleta dark neon =====
COLOR_BG_1     = "#1a0a2e"
COLOR_BG_2     = "#2d0d4f"
COLOR_SIDEBAR  = "#150826"
COLOR_CARD     = "#25163d"
COLOR_CARD_2   = "#2d1b4a"
COLOR_BORDER   = "rgba(168, 85, 247, 0.22)"
COLOR_TEXT     = "#f0e6ff"
COLOR_TEXTO    = COLOR_TEXT
COLOR_MUTED    = "#8b7ba5"

COLOR_CYAN     = "#00f0ff"
COLOR_MAGENTA  = "#ff00c8"
COLOR_GREEN    = "#00ff88"
COLOR_YELLOW   = "#ffe14d"
COLOR_PURPLE   = "#a855f7"
COLOR_ORANGE   = "#ff7b00"

COLOR_META     = COLOR_YELLOW
COLOR_BARRA    = COLOR_CYAN
COLOR_LINEA    = COLOR_MAGENTA

PALETA = [COLOR_CYAN, COLOR_MAGENTA, COLOR_GREEN, COLOR_YELLOW,
          COLOR_PURPLE, COLOR_ORANGE, "#4d7cff", "#ff4d6d"]

MIN_VERIF = 5


# ===== DATOS =====
def _csv(nombre):
    return pd.read_csv(DATA_DIR / nombre, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def cargar_datos():
    fact = _csv("fact_escaneos.csv")
    tiempo = _csv("dim_tiempo.csv")
    clase = _csv("dim_clase.csv")
    parcela = _csv("dim_parcela.csv")
    modelo = _csv("dim_modelo.csv")

    tiempo["fecha"] = pd.to_datetime(tiempo["fecha"], errors="coerce")
    tiempo["periodo"] = tiempo["fecha"].dt.strftime("%Y-%m")
    tiempo["dia"] = tiempo["fecha"].dt.strftime("%Y-%m-%d")
    tiempo["anio"] = tiempo["anio"].astype(str)
    tiempo["trimestre"] = "T" + tiempo["trimestre"].astype(str)
    tiempo["anio_trim"] = tiempo["anio"] + "-" + tiempo["trimestre"]

    ver = modelo["version_modelo"].astype(str)
    alg = modelo["algoritmo"].astype(str)
    modelo["modelo"] = np.where(ver.str.lower() == "desconocido", alg, alg + " " + ver)
    dup = modelo["modelo"].duplicated(keep=False)
    modelo.loc[dup, "modelo"] = modelo.loc[dup, "modelo"] + " (id " + modelo.loc[dup, "id_modelo"].astype(str) + ")"

    c_real = clase.rename(columns={"id_clase": "id_clase_real", "nombre_clase": "clase_real", "categoria": "categoria_real"})
    c_diag = clase.rename(columns={"id_clase": "id_diagnostico", "nombre_clase": "clase_diag", "categoria": "categoria_diag"})

    df = (
        fact.merge(tiempo[["id_tiempo", "fecha", "anio", "mes", "nombre_mes", "trimestre", "anio_trim", "periodo", "dia"]],
                   on="id_tiempo", how="left")
        .merge(c_real, on="id_clase_real", how="left")
        .merge(c_diag, on="id_diagnostico", how="left")
        .merge(parcela, on="id_parcela", how="left")
        .merge(modelo[["id_modelo", "algoritmo", "modelo"]], on="id_modelo", how="left")
        .rename(columns={"nombre_parcela": "parcela"})
    )

    for col in ["clase_real", "categoria_real", "clase_diag", "categoria_diag", "parcela", "algoritmo", "modelo"]:
        df[col] = df[col].fillna("desconocido").astype(str)

    df["confianza_pct"] = pd.to_numeric(df["confianza_pct"], errors="coerce")
    df["acierto"] = pd.to_numeric(df["acierto"], errors="coerce")
    df["cantidad_escaneos"] = pd.to_numeric(df["cantidad_escaneos"], errors="coerce").fillna(1)
    df["conf_baja"] = (df["confianza_pct"] < UMBRAL).astype(float).where(df["confianza_pct"].notna())
    return df


def resumen(df, por):
    n = df["cantidad_escaneos"]
    d = df.assign(
        _ok=df["acierto"] * n,
        _n=n.where(df["acierto"].notna(), 0),
        _b=df["conf_baja"] * n,
        _nb=n.where(df["conf_baja"].notna(), 0),
    )
    g = d.groupby(por, as_index=False).agg(
        escaneos=("cantidad_escaneos", "sum"),
        _ok=("_ok", "sum"),
        _n=("_n", "sum"),
        _b=("_b", "sum"),
        _nb=("_nb", "sum"),
        confianza=("confianza_pct", "mean"),
    )
    g["acierto_pct"] = g["_ok"] / g["_n"].replace(0, np.nan) * 100
    g["baja_pct"] = g["_b"] / g["_nb"].replace(0, np.nan) * 100
    g = g.rename(columns={"_n": "verificados", "_b": "n_baja"})
    return g.drop(columns=["_ok", "_nb"])


def precision(df):
    v = df[df["acierto"].notna()]
    n = v["cantidad_escaneos"].sum()
    if n == 0:
        return np.nan
    return (v["acierto"] * v["cantidad_escaneos"]).sum() / n * 100


def pct_baja(df):
    v = df[df["conf_baja"].notna()]
    n = v["cantidad_escaneos"].sum()
    if n == 0:
        return np.nan
    return (v["conf_baja"] * v["cantidad_escaneos"]).sum() / n * 100


def sin_desc(df, col):
    return df[df[col].astype(str).str.lower() != "desconocido"]


def cat_es(df, valores, col="categoria_diag"):
    return df[col].astype(str).str.lower().isin(valores)


def es_plaga(df, col="categoria_diag"):
    return cat_es(df, ["plaga"], col)


def es_incidencia(df, col="categoria_diag"):
    return cat_es(df, ["plaga", "enfermedad"], col)


def fmt_int(x):
    return f"{x:,.0f}"


def fmt_pct(x):
    return "s/d" if pd.isna(x) else f"{x:.1f}%"


def _mejor_peor(r, col="acierto_pct"):
    x = r.dropna(subset=[col])
    y = x[x["verificados"] >= MIN_VERIF]
    if y.empty:
        y = x
    if y.empty:
        return None, None
    return y.loc[y[col].idxmax()], y.loc[y[col].idxmin()]


def matriz_confusion(df):
    d = sin_desc(sin_desc(df, "clase_real"), "clase_diag")
    if d.empty:
        return None, None, None
    cm = d.pivot_table(index="clase_real", columns="clase_diag", values="cantidad_escaneos",
                       aggfunc="sum", fill_value=0)
    clases = sorted(set(cm.index) | set(cm.columns))
    cm = cm.reindex(index=clases, columns=clases, fill_value=0)
    tp = np.diag(cm.values).astype(float)
    pred = cm.sum(axis=0).values.astype(float)
    real = cm.sum(axis=1).values.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        met = pd.DataFrame({
            "clase": clases,
            "precision": np.where(pred > 0, tp / pred * 100, np.nan),
            "sensibilidad": np.where(real > 0, tp / real * 100, np.nan),
            "real": real,
        })
    pares = d[d["clase_real"] != d["clase_diag"]].groupby(["clase_real", "clase_diag"], as_index=False)["cantidad_escaneos"].sum()
    pares = pares.sort_values("cantidad_escaneos", ascending=False)
    return cm, met, pares


def calcular_hallazgos(df):
    hall, recs = [], []
    total = df["cantidad_escaneos"].sum()
    if total == 0:
        return hall, recs

    prec = precision(df)
    if pd.notna(prec):
        cumple = prec >= META_EXACTITUD
        hall.append(("Exactitud del modelo",
                     f"Exactitud global de <b>{prec:.1f}%</b> frente a la meta de {META_EXACTITUD}%: "
                     + ("<b>cumple</b> la meta." if cumple else f"faltan <b>{META_EXACTITUD - prec:.1f}</b> puntos.")))
        recs.append("Mantener el modelo actual y monitorear su exactitud." if cumple else
                    f"Reentrenar el modelo: la exactitud ({prec:.1f}%) está por debajo de la meta de {META_EXACTITUD}%.")

    rm = resumen(sin_desc(df, "modelo"), "modelo")
    mejor, peor = _mejor_peor(rm)
    if mejor is not None:
        m = escape(str(mejor["modelo"]))
        hall.append(("Mejor versión del modelo", f"<b>{m}</b> alcanza {mejor['acierto_pct']:.1f}% de exactitud ({mejor['verificados']:.0f} escaneos verificados)."))
        recs.append(f"Priorizar la versión <b>{m}</b> en producción.")
        if peor is not None and peor["modelo"] != mejor["modelo"]:
            p = escape(str(peor["modelo"]))
            hall.append(("Versión con menor exactitud", f"<b>{p}</b> obtiene {peor['acierto_pct']:.1f}% ({peor['verificados']:.0f} escaneos verificados)."))
            recs.append(f"Revisar o reentrenar <b>{p}</b>, o retirarla si no mejora.")

    dp = sin_desc(df, "parcela")
    inc = dp[es_incidencia(dp)]
    if not inc.empty:
        pp = inc.groupby("parcela")["cantidad_escaneos"].sum().sort_values(ascending=False)
        top, n = escape(str(pp.index[0])), pp.iloc[0]
        hall.append(("Parcela más afectada", f"<b>{top}</b> concentra {fmt_int(n)} escaneos con plaga o enfermedad ({n / pp.sum() * 100:.0f}% del total de incidencias)."))
        recs.append(f"Priorizar el monitoreo y el control en <b>{top}</b>.")
        pm = inc.groupby("periodo")["cantidad_escaneos"].sum()
        if not pm.empty:
            hall.append(("Mes con mayor presión", f"<b>{escape(str(pm.idxmax()))}</b> fue el mes con más incidencias ({fmt_int(pm.max())})."))

    _, met, pares = matriz_confusion(df)
    if pares is not None and not pares.empty:
        r = pares.iloc[0]
        a, b = escape(str(r["clase_real"])), escape(str(r["clase_diag"]))
        hall.append(("Confusión más frecuente", f"El modelo diagnostica <b>{b}</b> cuando la clase real es <b>{a}</b> ({fmt_int(r['cantidad_escaneos'])} escaneos)."))
        recs.append(f"Agregar imágenes de <b>{a}</b> y <b>{b}</b> al banco de entrenamiento para que el modelo las distinga mejor.")

    baja = pct_baja(df)
    if pd.notna(baja):
        rb = resumen(sin_desc(df, "modelo"), "modelo").dropna(subset=["baja_pct"])
        txt = f"<b>{baja:.1f}%</b> de los escaneos tiene confianza menor a {UMBRAL}%."
        if not rb.empty:
            w = rb.loc[rb["baja_pct"].idxmax()]
            txt += f" La versión con más casos es <b>{escape(str(w['modelo']))}</b> ({w['baja_pct']:.1f}%)."
        hall.append(("Confianza baja", txt))
        recs.append(f"Revisar manualmente los escaneos con confianza menor a {UMBRAL}%.")

    sin_verif = df.loc[df["acierto"].isna(), "cantidad_escaneos"].sum() / total * 100
    if sin_verif > 0:
        hall.append(("Calidad de datos", f"<b>{sin_verif:.0f}%</b> de los escaneos no tiene la clase real del experto, así que no entra en la exactitud."))
        recs.append("Completar la verificación del experto para tener métricas más confiables.")
    return hall, recs


def tabla_decision(df):
    prec = precision(df)
    baja = pct_baja(df)
    total = df["cantidad_escaneos"].sum()
    inc = df.loc[es_incidencia(df), "cantidad_escaneos"].sum()
    sin_verif = df.loc[df["acierto"].isna(), "cantidad_escaneos"].sum() / total * 100 if total else np.nan
    filas = [
        ("Exactitud global", fmt_pct(prec), f"≥ {META_EXACTITUD}%",
         "—" if pd.isna(prec) else ("✓ Cumple la meta" if prec >= META_EXACTITUD else "✗ Por debajo · reentrenar")),
        (f"Escaneos con confianza < {UMBRAL}%", fmt_pct(baja), "Lo más bajo posible", "→ Revisar manualmente"),
        ("Incidencia (plaga + enfermedad)", fmt_pct(inc / total * 100 if total else np.nan), "Lo más bajo posible", "→ Priorizar parcelas afectadas"),
        ("Escaneos sin verificar", fmt_pct(sin_verif), "0%", "→ Completar verificación" if sin_verif > 0 else "✓ Completo"),
    ]
    return pd.DataFrame(filas, columns=["Indicador", "Valor actual", "Meta", "Acción sugerida"])


# ===== CSS =====
CSS = """
<style>
header[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.stApp {
    background: radial-gradient(circle at 20% 0%, #2d0d4f 0%, @@BG1@@ 45%, #120720 100%);
    min-height: 100vh;
}
.block-container { padding: 1rem 1.6rem 1.6rem 1.6rem; max-width: 100%; }
[data-testid="stMain"] :is(h1, h2, h3, h4, h5, p, label, li) { color: @@TEXT@@; }

/* ---------- TOPBAR ---------- */
.topbar {
    display: flex; align-items: center; gap: 1.1rem;
    background: linear-gradient(90deg, rgba(21,8,38,0.95), rgba(37,22,61,0.85));
    border: 1px solid @@BORDER@@;
    border-radius: 14px;
    padding: 0.9rem 1.3rem;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.4);
}
.topbar-brand {
    display: flex; align-items: center; gap: 0.9rem;
    flex: 0 0 auto;
    min-width: 260px;
}
.topbar-logo {
    display: inline-flex; align-items: center; justify-content: center;
    width: 54px; height: 54px; border-radius: 14px;
    background: linear-gradient(135deg, @@CYAN@@, @@MAGENTA@@);
    box-shadow: 0 0 22px rgba(0,240,255,0.5);
    overflow: hidden;
    flex-shrink: 0;
}
.topbar-logo img {
    width: 40px; height: 40px;
    object-fit: contain;
    filter: drop-shadow(0 0 4px rgba(0,0,0,0.3));
}
.topbar-logo .fallback-icon {
    color: #1a0a2e; font-weight: 900; font-size: 1.6rem;
    line-height: 1;
}
.topbar-name {
    color: #ffffff; font-weight: 800;
    font-size: 1.3rem; letter-spacing: 0.01em;
    white-space: nowrap;
}
.topbar-name span { color: @@CYAN@@; font-weight: 900; }
.topbar-search {
    flex: 1 1 auto;
    min-width: 200px;
    max-width: 560px;
    display: flex; align-items: center; gap: 0.65rem;
    background: rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    color: @@MUTED@@; font-size: 0.9rem;
    margin: 0 auto;
}
.topbar-search .search-icon { color: @@CYAN@@; font-size: 1.1rem; line-height: 1; }
.topbar-search .search-text {
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.topbar-icons {
    display: flex; align-items: center; gap: 0.55rem;
    flex: 0 0 auto;
}
.tb-btn {
    display: inline-flex; align-items: center; justify-content: center;
    width: 44px; height: 44px;
    background: rgba(168, 85, 247, 0.1);
    border: 1px solid rgba(168, 85, 247, 0.22);
    border-radius: 10px;
    color: #b8a8d0; font-size: 1.1rem;
    transition: all 0.2s;
}
.tb-btn:hover { background: rgba(0,240,255,0.12); color: @@CYAN@@; border-color: @@CYAN@@; }
.tb-avatar {
    width: 44px; height: 44px; border-radius: 50%;
    background: linear-gradient(135deg, @@PURPLE@@, @@CYAN@@);
    display: inline-flex; align-items: center; justify-content: center;
    color: #1a0a2e; font-weight: 900; font-size: 1.1rem;
    box-shadow: 0 0 14px rgba(168,85,247,0.5);
    margin-left: 0.35rem;
}

@media (max-width: 900px) {
    .topbar-search { display: none; }
    .topbar-brand { min-width: auto; }
    .topbar-name { font-size: 1.05rem; }
}

/* ---------- SIDEBAR ---------- */
.st-key-menu {
    background: linear-gradient(180deg, @@SIDEBAR@@ 0%, #1d0a35 100%);
    border-radius: 14px;
    padding: 1.2rem 0.9rem 1.4rem 0.9rem;
    border: 1px solid @@BORDER@@;
    box-shadow: 0 8px 24px rgba(0,0,0,0.5);
}
.st-key-menu [data-testid="stVerticalBlock"] { gap: 0.25rem; }
.st-key-menu .stButton button {
    width: 100%; min-height: 2.6rem; height: auto;
    background: transparent !important;
    border: none !important;
    border-left: 3px solid transparent !important;
    border-radius: 8px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.55rem 0.85rem !important;
    transition: all 0.2s ease;
}
.st-key-menu .stButton button p {
    font-size: 0.86rem;
    text-align: left;
    color: #b8a8d0 !important;
}
.st-key-menu button[kind="primary"],
.st-key-menu [data-testid="stBaseButton-primary"] {
    background: rgba(0, 240, 255, 0.09) !important;
    border-left: 3px solid @@CYAN@@ !important;
    box-shadow: inset 0 0 20px rgba(0,240,255,0.08) !important;
}
.st-key-menu button[kind="primary"] p,
.st-key-menu [data-testid="stBaseButton-primary"] p {
    color: @@CYAN@@ !important;
    font-weight: 700;
}
.st-key-menu button[kind="secondary"]:hover,
.st-key-menu [data-testid="stBaseButton-secondary"]:hover {
    background: rgba(168, 85, 247, 0.09) !important;
    border-left: 3px solid @@PURPLE@@ !important;
}
.st-key-menu button[kind="secondary"]:hover p,
.st-key-menu [data-testid="stBaseButton-secondary"]:hover p {
    color: #ffffff !important;
}
.st-key-menu :is(label, [data-testid="stWidgetLabel"] p) {
    color: #8b7ba5 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
}

.brand {
    text-align: center;
    padding: 0 0 1rem 0;
    border-bottom: 1px solid rgba(168, 85, 247, 0.15);
    margin-bottom: 0.9rem;
}
.brand img {
    width: 84px; height: 84px;
    object-fit: contain;
    filter: drop-shadow(0 0 16px rgba(0,240,255,0.55));
}
.brand-n {
    color: #ffffff;
    font-size: 1.2rem; font-weight: 800;
    letter-spacing: 0.18em; text-transform: uppercase;
    margin-top: 0.4rem;
    background: linear-gradient(90deg, @@CYAN@@, @@MAGENTA@@);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.brand-s {
    color: #8b7ba5; font-size: 0.72rem; letter-spacing: 0.04em;
    margin-top: 0.2rem; line-height: 1.3;
}
.filtros-t {
    color: #8b7ba5;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700;
    border-top: 1px solid rgba(168, 85, 247, 0.15);
    padding-top: 0.9rem;
    margin-top: 0.6rem;
}

/* ---------- BREADCRUMB ---------- */
.breadcrumb-page {
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-weight: 700;
    color: #8b7ba5;
    margin: 0 0 1.2rem 0;
}
.breadcrumb-page .bc-root { color: #b8a8d0; }
.breadcrumb-page .bc-sep { color: @@MAGENTA@@; }
.breadcrumb-page .bc-current { color: @@CYAN@@; text-shadow: 0 0 8px rgba(0,240,255,0.4); }

/* ---------- KPI ---------- */
.kpi-card {
    position: relative;
    background: linear-gradient(145deg, #2a1a4a 0%, #1f1235 100%);
    border: 1px solid @@BORDER@@;
    border-radius: 14px;
    padding: 1rem 1rem 0.85rem;
    text-align: center;
    margin-bottom: 0.8rem;
    overflow: hidden;
    box-shadow: 0 4px 18px rgba(0,0,0,0.35);
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.9;
}
.kpi-gauge-wrap {
    position: relative;
    width: 90px; height: 90px;
    margin: 0 auto 0.5rem;
}
.kpi-gauge { width: 100%; height: 100%; display: block; }
.kpi-number {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 1.15rem;
    font-weight: 800;
    color: var(--accent);
    text-shadow: 0 0 14px var(--accent);
    white-space: nowrap;
    letter-spacing: -0.02em;
}
.kpi-value-big {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--accent);
    text-shadow: 0 0 14px var(--accent);
    line-height: 1.15;
    margin: 0.55rem 0 0.4rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-title {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #b8a8d0;
    font-weight: 700;
    margin-bottom: 0.15rem;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #7a6b95;
}

/* ---------- INFO CARD ---------- */
.info-card {
    background: linear-gradient(145deg, #2a1a4a 0%, #1f1235 100%);
    border: 1px solid @@BORDER@@;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.info-card::after {
    content: ''; position: absolute; bottom: -40px; right: -40px;
    width: 100px; height: 100px; border-radius: 50%;
    background: radial-gradient(circle, rgba(0,240,255,0.15), transparent 70%);
    pointer-events: none;
}
.info-card .lbl {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: @@MAGENTA@@;
    font-weight: 800;
    margin-bottom: 0.5rem;
    text-shadow: 0 0 10px rgba(255,0,200,0.4);
}
.info-card .txt {
    color: #e0d5f0;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ---------- SECTION ---------- */
.section-h {
    display: flex;
    align-items: baseline;
    gap: 0.9rem;
    border-bottom: 1px solid @@BORDER@@;
    padding-bottom: 0.55rem;
    margin: 1.8rem 0 1rem 0;
}
.section-n {
    font-size: 1.3rem;
    color: @@CYAN@@;
    font-weight: 800;
    line-height: 1;
    text-shadow: 0 0 10px rgba(0,240,255,0.5);
}
.section-t {
    font-size: 1.15rem;
    color: #ffffff;
    font-weight: 600;
    letter-spacing: 0.01em;
}

/* ---------- HALLAZGO ---------- */
.hallazgo {
    background: rgba(0, 240, 255, 0.06);
    border-left: 3px solid @@CYAN@@;
    color: #e0d5f0;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.1rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
    box-shadow: inset 0 0 30px rgba(0,240,255,0.03);
}
.hallazgo b { color: @@CYAN@@; text-shadow: 0 0 8px rgba(0,240,255,0.4); }

/* ---------- CARD ---------- */
.card {
    background: linear-gradient(145deg, #2a1a4a 0%, #1f1235 100%);
    border: 1px solid @@BORDER@@;
    border-left: 3px solid @@PURPLE@@;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    min-height: 6rem;
}
.card-t {
    color: @@CYAN@@;
    font-weight: 800;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.card-x {
    color: #d5c9e8;
    font-size: 0.88rem;
    line-height: 1.55;
}
.card-x b { color: #ffffff; }

/* ---------- RECO ---------- */
.reco-box {
    background: rgba(0, 255, 136, 0.06);
    border-left: 3px solid @@GREEN@@;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.1rem;
    margin-bottom: 0.55rem;
    color: #d5c9e8;
    font-size: 0.9rem;
    line-height: 1.6;
    box-shadow: inset 0 0 30px rgba(0,255,136,0.03);
}
.reco-box b { color: @@GREEN@@; text-shadow: 0 0 8px rgba(0,255,136,0.35); }
.reco-box b.principal { color: @@CYAN@@; text-shadow: 0 0 10px rgba(0,240,255,0.5); }

/* ---------- PIE ---------- */
.pie {
    color: #7a6b95;
    font-size: 0.72rem;
    margin-top: 1.4rem;
    padding-top: 0.8rem;
    border-top: 1px solid rgba(168, 85, 247, 0.15);
    letter-spacing: 0.03em;
}

/* ---------- FORM ---------- */
[data-testid="stMain"] [data-testid="stWidgetLabel"] p {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b7ba5 !important;
    font-weight: 700;
}
[data-testid="stMain"] [data-testid="stRadio"] label p {
    color: #d5c9e8 !important;
    font-size: 0.85rem !important;
    text-transform: none;
    letter-spacing: 0;
    font-weight: 500;
}

/* ---------- ST.TABLE ---------- */
[data-testid="stTable"] table {
    background: rgba(255,255,255,0.02);
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid @@BORDER@@;
}
[data-testid="stTable"] thead tr th {
    background: rgba(168, 85, 247, 0.15) !important;
    color: @@CYAN@@ !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    padding: 0.65rem 0.85rem !important;
    border-bottom: 1px solid @@BORDER@@ !important;
    text-align: left !important;
}
[data-testid="stTable"] tbody tr td {
    color: #e0d5f0 !important;
    padding: 0.6rem 0.85rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    font-size: 0.85rem !important;
    background: transparent !important;
}
[data-testid="stTable"] tbody tr:hover td { background: rgba(0,240,255,0.04) !important; }

/* ---------- ALERT ---------- */
[data-testid="stAlert"] {
    background: rgba(168,85,247,0.08) !important;
    border-left: 3px solid @@PURPLE@@ !important;
    color: #d5c9e8 !important;
    border-radius: 8px !important;
}
[data-testid="stAlert"] p { color: #d5c9e8 !important; }

/* ---------- CAPTION ---------- */
[data-testid="stMain"] [data-testid="stCaptionContainer"] p {
    color: #7a6b95 !important;
    font-size: 0.75rem !important;
}
[data-testid="stMain"] .stMarkdown p { color: #e0d5f0; }

/* ---------- MULTISELECT ---------- */
.st-key-menu [data-baseweb="select"] > div {
    background: rgba(0,0,0,0.25) !important;
    border-color: rgba(168,85,247,0.25) !important;
    color: #d5c9e8 !important;
}
</style>
"""
for token, valor in {
    "@@BG1@@": COLOR_BG_1, "@@BG2@@": COLOR_BG_2,
    "@@SIDEBAR@@": COLOR_SIDEBAR,
    "@@TEXT@@": COLOR_TEXT,
    "@@MUTED@@": COLOR_MUTED,
    "@@BORDER@@": COLOR_BORDER,
    "@@CYAN@@": COLOR_CYAN, "@@MAGENTA@@": COLOR_MAGENTA,
    "@@GREEN@@": COLOR_GREEN, "@@YELLOW@@": COLOR_YELLOW,
    "@@PURPLE@@": COLOR_PURPLE,
}.items():
    CSS = CSS.replace(token, valor)
st.markdown(CSS, unsafe_allow_html=True)

LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="72" height="72" viewBox="0 0 64 64">
<circle cx="32" cy="32" r="26" fill="none" stroke="#00f0ff" stroke-width="2" opacity="0.9"/>
<circle cx="32" cy="32" r="18" fill="none" stroke="#a855f7" stroke-width="1.5" opacity="0.8"/>
<circle cx="32" cy="32" r="10" fill="none" stroke="#ff00c8" stroke-width="1.2" opacity="0.9"/>
<circle cx="32" cy="32" r="4" fill="#00f0ff"/>
</svg>"""


def logo_html():
    """Logo para sidebar: usa LOGO.png si existe, si no el SVG de anillos."""
    if LOGO.exists():
        datos, tipo = base64.b64encode(LOGO.read_bytes()).decode(), "image/png"
    else:
        datos, tipo = base64.b64encode(LOGO_SVG.encode()).decode(), "image/svg+xml"
    return f'<img src="data:{tipo};base64,{datos}" alt="logo">'


def logo_topbar():
    """Logo para el topbar: usa LOGO.png si existe, si no el glifo ◆."""
    if LOGO.exists():
        datos = base64.b64encode(LOGO.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{datos}" alt="logo">'
    return '<span class="fallback-icon">◆</span>'


# ----- utilidades de dibujo -----
def boton(label, **kw):
    try:
        return st.button(label, width="stretch", **kw)
    except TypeError:
        return st.button(label, use_container_width=True, **kw)


def mostrar(fig, key):
    try:
        st.plotly_chart(fig, width="stretch", theme=None, key=key,
                        config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None, key=key,
                        config={"displayModeBar": False})


def estilo(fig, titulo, x="", y=""):
    fig.update_layout(
        title=dict(text=titulo, x=0, xanchor="left",
                   font=dict(size=14, color=COLOR_TEXT, family="Segoe UI, Inter, sans-serif")),
        height=ALTO, margin=dict(l=60, r=20, t=55, b=55),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_TEXT, family="Segoe UI, Inter, sans-serif", size=11),
        legend=dict(title=None, font=dict(size=10, color=COLOR_TEXT),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#25163d", font=dict(color="#f0e6ff", size=11),
                        bordercolor=COLOR_CYAN),
    )
    fig.update_xaxes(title_text=x, showgrid=False, automargin=True,
                     title_font=dict(size=11, color=COLOR_MUTED),
                     tickfont=dict(size=10, color=COLOR_MUTED),
                     linecolor="rgba(255,255,255,0.15)")
    fig.update_yaxes(title_text=y, gridcolor="rgba(255,255,255,0.06)", griddash="dot",
                     zeroline=False, automargin=True,
                     title_font=dict(size=11, color=COLOR_MUTED),
                     tickfont=dict(size=10, color=COLOR_MUTED),
                     linecolor="rgba(255,255,255,0.15)")
    return fig


def rejilla(items):
    n = max(1, COLUMNAS_GRAFICAS)
    for i in range(0, len(items), n):
        cols = st.columns(n, gap="large")
        for col, (clave, fig) in zip(cols, items[i:i + n]):
            with col:
                if isinstance(fig, str):
                    st.info(fig)
                else:
                    mostrar(fig, clave)


def sin_datos(titulo):
    return f"{titulo}: sin datos con los filtros actuales."


def meta_h(fig, valor=META_EXACTITUD):
    fig.add_hline(y=valor, line_dash="dash", line_color=COLOR_META,
                  line_width=1.5,
                  annotation_text=f"Meta {valor}%",
                  annotation_position="top left",
                  annotation_font=dict(color=COLOR_META, size=10))
    return fig


def _auto_pct(valor):
    if isinstance(valor, str) and valor.strip().endswith('%'):
        try:
            return float(valor.strip().rstrip('%'))
        except ValueError:
            return None
    return None


def kpi(col, titulo, valor, sub="", color=COLOR_CYAN):
    pct = _auto_pct(valor)
    if pct is not None:
        pct_c = max(0.0, min(100.0, pct))
        dash = pct_c * 2.513
        inner = f"""
        <div class='kpi-gauge-wrap'>
            <svg class='kpi-gauge' viewBox='0 0 100 100'>
                <circle cx='50' cy='50' r='40' stroke='rgba(255,255,255,0.06)' stroke-width='9' fill='none'/>
                <circle cx='50' cy='50' r='40' stroke='{color}' stroke-width='9' fill='none'
                        stroke-dasharray='{dash} 251' stroke-linecap='round'
                        transform='rotate(-90 50 50)'
                        style='filter: drop-shadow(0 0 6px {color});'/>
            </svg>
            <div class='kpi-number'>{valor}</div>
        </div>
        """
    else:
        inner = f"<div class='kpi-value-big' style='color:{color};'>{valor}</div>"
    col.markdown(
        f"<div class='kpi-card' style='--accent:{color};'>{inner}"
        f"<div class='kpi-title'>{titulo}</div>"
        f"<div class='kpi-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def hallazgo(texto):
    st.markdown(f"<div class='hallazgo'><b>Hallazgo.</b> {texto}</div>", unsafe_allow_html=True)


def encabezado(clave, titulo):
    st.markdown(
        f"<div class='breadcrumb-page'>"
        f"<span class='bc-root'>{NOMBRE}</span>"
        f"<span class='bc-sep'>/</span>"
        f"<span class='bc-current'>{escape(titulo)}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def seccion(num, titulo):
    st.markdown(
        f"<div class='section-h'><div class='section-n'>{num}</div>"
        f"<div class='section-t'>{escape(titulo)}</div></div>",
        unsafe_allow_html=True,
    )


def nota_desc(df, d):
    n = df["cantidad_escaneos"].sum() - d["cantidad_escaneos"].sum()
    if n > 0:
        st.caption(f"{fmt_int(n)} escaneos con valor «desconocido» no se muestran en estas gráficas.")


def barra_h(r, x, y, titulo, fmt, xt, color=None):
    r = r.sort_values(x, ascending=True)
    fig = px.bar(r, x=x, y=y, orientation="h", text_auto=fmt, color=color,
                 color_discrete_sequence=PALETA if color else [COLOR_BARRA])
    fig.update_traces(textfont=dict(color=COLOR_TEXT, size=10), marker_line_width=0)
    return estilo(fig, titulo, x=xt)


# ===== TOPBAR =====
def topbar():
    st.markdown(
        f"""
        <div class="topbar">
            <div class="topbar-brand">
                <span class="topbar-logo">{logo_topbar()}</span>
                <span class="topbar-name">{NOMBRE}<span> · </span>Dashboard</span>
            </div>
            <div class="topbar-search">
                <span class="search-icon">⌕</span>
                <span class="search-text">Buscar parcela, clase o modelo…</span>
            </div>
            <div class="topbar-icons">
                <span class="tb-btn">◉</span>
                <span class="tb-btn">⚙</span>
                <span class="tb-btn">✉</span>
                <span class="tb-avatar">A</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===== PÁGINAS =====
def pagina_inicio(df):
    total = df["cantidad_escaneos"].sum()
    inc = df.loc[es_incidencia(df), "cantidad_escaneos"].sum()

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            "<div class='info-card'><div class='lbl'>Objetivo</div>"
            "<div class='txt'>Analizar a lo largo del tiempo los diagnósticos del detector inteligente del gusano "
            "cogollero y la roya del maíz del Instituto Tecnológico de Tizimín, para decidir en qué parcelas y "
            "periodos intervenir y cuándo conviene reentrenar el modelo.</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='info-card'><div class='lbl'>Usuarios</div>"
            "<div class='txt'>Docente y estudiantes de Agronomía, y el equipo de desarrollo del proyecto.</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='info-card'><div class='lbl'>Cómo usarlo</div>"
            "<div class='txt'>Elija una sección en el menú. Las secciones R1 a R6 responden los seis requerimientos "
            "del data mart. «Toma de decisiones» concentra los hallazgos y las acciones sugeridas.</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    k = st.columns(4)
    kpi(k[0], "Total de escaneos", fmt_int(total), f"{fmt_int(len(df))} registros", color=COLOR_CYAN)
    kpi(k[1], "Exactitud del modelo", fmt_pct(precision(df)), f"meta {META_EXACTITUD}%", color=COLOR_MAGENTA)
    kpi(k[2], "Confianza promedio", fmt_pct(df["confianza_pct"].mean()), "reportada por el modelo", color=COLOR_GREEN)
    kpi(k[3], "Plaga o enfermedad", fmt_int(inc), f"{inc / total * 100:.1f}% de los escaneos", color=COLOR_YELLOW)

    por_mes = resumen(df, "periodo")
    fig1 = px.line(por_mes, x="periodo", y="escaneos", markers=True,
                   color_discrete_sequence=[COLOR_LINEA])
    fig1.update_traces(line_width=2.5, marker=dict(size=6, line=dict(width=0)))
    fig1.update_xaxes(type="category")
    por_cat = resumen(df, "categoria_diag").sort_values("escaneos", ascending=False)
    fig2 = px.bar(por_cat, x="categoria_diag", y="escaneos", text_auto=".0f",
                  color_discrete_sequence=PALETA)
    fig2.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
    rejilla([
        ("i_mes", estilo(fig1, "Escaneos por mes", x="Mes", y="Escaneos")),
        ("i_cat", estilo(fig2, "Escaneos por categoría diagnosticada", y="Escaneos")),
    ])

    top_mes = por_mes.loc[por_mes["escaneos"].idxmax()]
    top_cat = por_cat.iloc[0]
    hallazgo(
        f"El mes con más actividad fue <b>{escape(str(top_mes['periodo']))}</b> ({fmt_int(top_mes['escaneos'])} escaneos) "
        f"y la categoría más diagnosticada es <b>{escape(str(top_cat['categoria_diag']))}</b> "
        f"({top_cat['escaneos'] / total * 100:.0f}% del total)."
    )


def pagina_r1(df):
    d = sin_desc(df, "clase_real")
    if d["acierto"].notna().sum() == 0:
        st.info(sin_datos("R1"))
        return
    prec = precision(d)
    rc = resumen(d, "clase_real")
    mejor, peor = _mejor_peor(rc)
    verif = d.loc[d["acierto"].notna(), "cantidad_escaneos"].sum()

    k = st.columns(4)
    kpi(k[0], "Exactitud global", fmt_pct(prec),
        "✓ cumple la meta" if prec >= META_EXACTITUD else f"faltan {META_EXACTITUD - prec:.1f} pts",
        color=COLOR_CYAN)
    kpi(k[1], "Clase con mayor exactitud", escape(str(mejor["clase_real"])) if mejor is not None else "s/d",
        fmt_pct(mejor["acierto_pct"]) if mejor is not None else "", color=COLOR_GREEN)
    kpi(k[2], "Clase con menor exactitud", escape(str(peor["clase_real"])) if peor is not None else "s/d",
        fmt_pct(peor["acierto_pct"]) if peor is not None else "", color=COLOR_MAGENTA)
    kpi(k[3], "Escaneos verificados", fmt_int(verif), "con clase real del experto", color=COLOR_PURPLE)

    items = []
    x = rc.dropna(subset=["acierto_pct"]).sort_values("acierto_pct", ascending=False)
    fig = px.bar(x, x="clase_real", y="acierto_pct", text_auto=".1f", color_discrete_sequence=[COLOR_BARRA])
    fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
    items.append(("r1_clase", meta_h(estilo(fig, "Exactitud por clase (%)", y="Exactitud %"))))

    x = resumen(d, "periodo").dropna(subset=["acierto_pct"])
    if x.empty:
        items.append(("r1_mes", sin_datos("Exactitud por mes")))
    else:
        fig = px.line(x, x="periodo", y="acierto_pct", markers=True, color_discrete_sequence=[COLOR_LINEA])
        fig.update_traces(line_width=2.5, marker=dict(size=6))
        fig.update_xaxes(type="category")
        items.append(("r1_mes", meta_h(estilo(fig, "Exactitud por mes (%)", x="Mes", y="Exactitud %"))))

    x = resumen(d, "anio_trim").dropna(subset=["acierto_pct"])
    if x.empty:
        items.append(("r1_trim", sin_datos("Exactitud por trimestre")))
    else:
        fig = px.bar(x, x="anio_trim", y="acierto_pct", text_auto=".1f", color_discrete_sequence=[COLOR_BARRA])
        fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
        fig.update_xaxes(type="category")
        items.append(("r1_trim", meta_h(estilo(fig, "Exactitud por trimestre (%)", y="Exactitud %"))))

    x = resumen(sin_desc(d, "modelo"), "modelo").dropna(subset=["acierto_pct"])
    if x.empty:
        items.append(("r1_mod", sin_datos("Exactitud por versión del modelo")))
    else:
        fig = barra_h(x, "acierto_pct", "modelo", "Exactitud por versión del modelo (%)", ".1f", "Exactitud %")
        fig.add_vline(x=META_EXACTITUD, line_dash="dash", line_color=COLOR_META, line_width=1.5)
        items.append(("r1_mod", fig))
    rejilla(items)
    nota_desc(df, d)

    texto = (f"La exactitud global es <b>{prec:.1f}%</b>: " +
             ("cumple la meta de " if prec >= META_EXACTITUD else "no alcanza la meta de ") + f"{META_EXACTITUD}%.")
    if peor is not None and mejor is not None and peor["clase_real"] != mejor["clase_real"]:
        texto += (f" Va de {peor['acierto_pct']:.1f}% en <b>{escape(str(peor['clase_real']))}</b> "
                  f"a {mejor['acierto_pct']:.1f}% en <b>{escape(str(mejor['clase_real']))}</b>.")
    hallazgo(texto)


def pagina_r2(df):
    d = sin_desc(df, "parcela")
    if d.empty:
        st.info(sin_datos("R2"))
        return
    total = d["cantidad_escaneos"].sum()
    plaga = d.loc[es_plaga(d), "cantidad_escaneos"].sum()
    enf = d.loc[cat_es(d, ["enfermedad"]), "cantidad_escaneos"].sum()
    inc = d.loc[es_incidencia(d), "cantidad_escaneos"].sum()

    g = d.assign(_inc=np.where(es_incidencia(d), d["cantidad_escaneos"], 0)).groupby("parcela", as_index=False).agg(
        escaneos=("cantidad_escaneos", "sum"), incidencias=("_inc", "sum"))
    g["tasa"] = g["incidencias"] / g["escaneos"] * 100
    top = g.loc[g["incidencias"].idxmax()]

    k = st.columns(4)
    kpi(k[0], "Plaga (cogollero)", fmt_int(plaga), f"{plaga / total * 100:.1f}% de los escaneos", color=COLOR_CYAN)
    kpi(k[1], "Enfermedad (roya)", fmt_int(enf), f"{enf / total * 100:.1f}% de los escaneos", color=COLOR_MAGENTA)
    kpi(k[2], "Incidencia total", f"{inc / total * 100:.1f}%", f"{fmt_int(inc)} escaneos", color=COLOR_YELLOW)
    kpi(k[3], "Parcela más afectada", escape(str(top["parcela"])), f"{fmt_int(top['incidencias'])} incidencias", color=COLOR_ORANGE)

    items = []
    rcat = resumen(d, ["parcela", "categoria_diag"])
    fig = px.bar(rcat, x="parcela", y="escaneos", color="categoria_diag", barmode="stack",
                 color_discrete_sequence=PALETA)
    fig.update_layout(barnorm="percent")
    fig.update_traces(marker_line_width=0)
    items.append(("r2_stack", estilo(fig, "Diagnóstico por parcela (% de escaneos)", y="% de escaneos")))

    fig = px.bar(g.sort_values("tasa", ascending=False), x="parcela", y="tasa", text_auto=".1f",
                 color_discrete_sequence=[COLOR_BARRA])
    fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
    items.append(("r2_tasa", estilo(fig, "Incidencia por parcela (% de escaneos)", y="Incidencia %")))

    di = d[es_incidencia(d)]
    if di.empty:
        items.append(("r2_mes", sin_datos("Incidencia por mes")))
        items.append(("r2_heat", sin_datos("Incidencia por parcela y trimestre")))
    else:
        rm = resumen(di, ["periodo", "categoria_diag"])
        fig = px.line(rm, x="periodo", y="escaneos", color="categoria_diag", markers=True,
                      color_discrete_sequence=PALETA)
        fig.update_traces(line_width=2.5, marker=dict(size=6))
        fig.update_xaxes(type="category")
        items.append(("r2_mes", estilo(fig, "Plaga y enfermedad por mes", x="Mes", y="Escaneos")))
        mat = di.pivot_table(index="parcela", columns="anio_trim", values="cantidad_escaneos", aggfunc="sum", fill_value=0)
        fig = px.imshow(mat, text_auto=True, aspect="auto",
                        color_continuous_scale=[[0, "#1f1235"], [0.5, "#4a1d7d"], [1, "#00f0ff"]],
                        labels=dict(x="Trimestre", y="Parcela"))
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textfont=dict(color=COLOR_TEXT))
        items.append(("r2_heat", estilo(fig, "Incidencias por parcela y trimestre", x="Trimestre", y="Parcela")))
    rejilla(items)
    nota_desc(df, d)
    st.caption("Incidencia = escaneos diagnosticados como plaga o enfermedad. La zona de la parcela no está en las tablas exportadas, por eso se analiza por parcela.")

    hallazgo(f"<b>{escape(str(top['parcela']))}</b> es la parcela con más incidencias ({fmt_int(top['incidencias'])}; "
             f"{top['tasa']:.1f}% de sus escaneos). En total, {inc / total * 100:.1f}% de los escaneos resultó en plaga o enfermedad.")


def pagina_r3(df):
    d = df[df["conf_baja"].notna()]
    if d.empty:
        st.info(sin_datos("R3"))
        return
    baja = pct_baja(d)
    n_baja = (d["conf_baja"] * d["cantidad_escaneos"]).sum()
    rm = resumen(sin_desc(d, "modelo"), "modelo").dropna(subset=["baja_pct"])
    peor = rm.loc[rm["baja_pct"].idxmax()] if not rm.empty else None

    k = st.columns(4)
    kpi(k[0], "Umbral de confianza", f"{UMBRAL}%", "debajo = confianza baja", color=COLOR_YELLOW)
    kpi(k[1], "Escaneos con confianza baja", fmt_int(n_baja), f"{baja:.1f}% de los escaneos", color=COLOR_ORANGE)
    kpi(k[2], "Versión con más casos", escape(str(peor["modelo"])) if peor is not None else "s/d",
        fmt_pct(peor["baja_pct"]) if peor is not None else "", color=COLOR_MAGENTA)
    kpi(k[3], "Confianza promedio", fmt_pct(d["confianza_pct"].mean()), "de todos los escaneos", color=COLOR_CYAN)

    items = []
    if rm.empty:
        items.append(("r3_mod", sin_datos("Confianza baja por versión")))
    else:
        items.append(("r3_mod", barra_h(rm, "baja_pct", "modelo", "Confianza baja por versión (%)", ".1f", "% con confianza baja")))
    ra = resumen(sin_desc(d, "algoritmo"), "algoritmo").dropna(subset=["baja_pct"])
    if ra.empty:
        items.append(("r3_alg", sin_datos("Confianza baja por algoritmo")))
    else:
        fig = px.bar(ra.sort_values("baja_pct", ascending=False), x="algoritmo", y="baja_pct", text_auto=".1f",
                     color_discrete_sequence=[COLOR_BARRA])
        fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
        items.append(("r3_alg", estilo(fig, "Confianza baja por algoritmo (%)", y="% con confianza baja")))
    fig = px.histogram(d, x="confianza_pct", nbins=20, color_discrete_sequence=[COLOR_MAGENTA])
    fig.update_traces(marker_line_width=0)
    fig.add_vline(x=UMBRAL, line_dash="dash", line_color=COLOR_META,
                  annotation_text=f"Umbral {UMBRAL}%",
                  annotation_font=dict(color=COLOR_META))
    items.append(("r3_hist", estilo(fig, "Distribución de la confianza", x="Confianza %", y="Escaneos")))
    pm = resumen(d, "periodo").dropna(subset=["baja_pct"])
    fig = px.line(pm, x="periodo", y="baja_pct", markers=True, color_discrete_sequence=[COLOR_LINEA])
    fig.update_traces(line_width=2.5, marker=dict(size=6))
    fig.update_xaxes(type="category")
    items.append(("r3_mes", estilo(fig, "Confianza baja por mes (%)", x="Mes", y="% con confianza baja")))
    rejilla(items)
    st.caption("La fuente de la imagen (prototipo, campo o banco) no está en las tablas exportadas, por eso se analiza por versión y algoritmo.")

    texto = f"<b>{baja:.1f}%</b> de los escaneos tiene confianza menor a {UMBRAL}%."
    if peor is not None:
        texto += f" La versión con más casos es <b>{escape(str(peor['modelo']))}</b> ({peor['baja_pct']:.1f}%): conviene revisar sus diagnósticos manualmente."
    hallazgo(texto)


def pagina_r4(df):
    cm, met, pares = matriz_confusion(df)
    if cm is None:
        st.info(sin_datos("R4"))
        return
    total = cm.values.sum()
    ok = np.trace(cm.values)
    m = met.dropna(subset=["sensibilidad"])
    m = m[m["real"] >= 1]
    mejor = m.loc[m["sensibilidad"].idxmax()] if not m.empty else None
    peor = m.loc[m["sensibilidad"].idxmin()] if not m.empty else None

    k = st.columns(4)
    kpi(k[0], "Escaneos comparados", fmt_int(total), "clase real y diagnóstico", color=COLOR_CYAN)
    kpi(k[1], "Diagnósticos correctos", f"{ok / total * 100:.1f}%", f"{fmt_int(ok)} de {fmt_int(total)}", color=COLOR_GREEN)
    kpi(k[2], "Mejor sensibilidad", escape(str(mejor["clase"])) if mejor is not None else "s/d",
        fmt_pct(mejor["sensibilidad"]) if mejor is not None else "", color=COLOR_YELLOW)
    kpi(k[3], "Peor sensibilidad", escape(str(peor["clase"])) if peor is not None else "s/d",
        fmt_pct(peor["sensibilidad"]) if peor is not None else "", color=COLOR_MAGENTA)

    items = []
    fig = px.imshow(cm, text_auto=True, aspect="auto",
                    color_continuous_scale=[[0, "#1f1235"], [0.5, "#4a1d7d"], [1, "#00f0ff"]],
                    labels=dict(x="Diagnóstico del modelo", y="Clase real"))
    fig.update_coloraxes(showscale=False)
    fig.update_traces(textfont=dict(color=COLOR_TEXT))
    items.append(("r4_heat", estilo(fig, "Clase real vs. diagnóstico", x="Diagnóstico del modelo", y="Clase real")))

    d = sin_desc(sin_desc(df, "categoria_real"), "categoria_diag")
    if d.empty:
        items.append(("r4_cat", sin_datos("Matriz por categoría")))
    else:
        mc = d.pivot_table(index="categoria_real", columns="categoria_diag", values="cantidad_escaneos",
                           aggfunc="sum", fill_value=0)
        fig = px.imshow(mc, text_auto=True, aspect="auto",
                        color_continuous_scale=[[0, "#1f1235"], [0.5, "#4a1d7d"], [1, "#00f0ff"]],
                        labels=dict(x="Categoría diagnosticada", y="Categoría real"))
        fig.update_coloraxes(showscale=False)
        fig.update_traces(textfont=dict(color=COLOR_TEXT))
        items.append(("r4_cat", estilo(fig, "Por categoría de enfermedad", x="Categoría diagnosticada", y="Categoría real")))

    largo = met.melt(id_vars="clase", value_vars=["precision", "sensibilidad"], var_name="métrica", value_name="valor")
    largo["métrica"] = largo["métrica"].map({"precision": "Precisión", "sensibilidad": "Sensibilidad"})
    fig = px.bar(largo, x="clase", y="valor", color="métrica", barmode="group", text_auto=".0f",
                 color_discrete_sequence=[COLOR_CYAN, COLOR_MAGENTA])
    fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
    items.append(("r4_met", estilo(fig, "Precisión y sensibilidad por clase (%)", y="%")))

    if pares is None or pares.empty:
        items.append(("r4_par", "No hay confusiones entre clases con los filtros actuales."))
    else:
        p = pares.head(6).copy()
        p["par"] = p["clase_real"] + " → " + p["clase_diag"]
        items.append(("r4_par", barra_h(p, "cantidad_escaneos", "par", "Confusiones más frecuentes (real → diagnóstico)", ".0f", "Escaneos")))
    rejilla(items)

    texto = f"<b>{ok / total * 100:.1f}%</b> de los diagnósticos coincide con la clase real."
    if pares is not None and not pares.empty:
        r = pares.iloc[0]
        texto += (f" La confusión más frecuente es <b>{escape(str(r['clase_real']))}</b> clasificada como "
                  f"<b>{escape(str(r['clase_diag']))}</b> ({fmt_int(r['cantidad_escaneos'])} escaneos).")
    hallazgo(texto)


NIVELES = {"Año": "anio", "Trimestre": "anio_trim", "Mes": "periodo", "Día": "dia"}


def pagina_r5(df):
    nivel = st.radio("Nivel de detalle (año → trimestre → mes → día)", list(NIVELES),
                     index=2, horizontal=True, key="nivel_tiempo")
    col = NIVELES[nivel]
    por = resumen(df, col)
    di = df[es_incidencia(df)]
    pico = por.loc[por["escaneos"].idxmax()]
    pico_inc = di.groupby(col)["cantidad_escaneos"].sum() if not di.empty else None

    k = st.columns(4)
    kpi(k[0], "Escaneos", fmt_int(df["cantidad_escaneos"].sum()), f"nivel: {nivel.lower()}", color=COLOR_CYAN)
    kpi(k[1], f"{nivel} con más escaneos", escape(str(pico[col])), f"{fmt_int(pico['escaneos'])} escaneos", color=COLOR_MAGENTA)
    kpi(k[2], "Confianza promedio", fmt_pct(df["confianza_pct"].mean()), "de todos los escaneos", color=COLOR_GREEN)
    kpi(k[3], f"Pico de incidencia", escape(str(pico_inc.idxmax())) if pico_inc is not None else "s/d",
        f"{fmt_int(pico_inc.max())} casos" if pico_inc is not None else "", color=COLOR_YELLOW)

    items = []
    fig = px.line(por, x=col, y="escaneos", markers=True, color_discrete_sequence=[COLOR_LINEA])
    fig.update_traces(line_width=2.5, marker=dict(size=6))
    fig.update_xaxes(type="category")
    items.append(("r5_line", estilo(fig, f"Escaneos por {nivel.lower()}", x=nivel, y="Escaneos")))

    rc = resumen(df, [col, "categoria_diag"])
    fig = px.bar(rc, x=col, y="escaneos", color="categoria_diag", barmode="stack", color_discrete_sequence=PALETA)
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(type="category")
    items.append(("r5_stack", estilo(fig, f"Diagnósticos por {nivel.lower()}", x=nivel, y="Escaneos")))

    pc = por.dropna(subset=["confianza"])
    if pc.empty:
        items.append(("r5_conf", sin_datos("Confianza promedio")))
    else:
        fig = px.line(pc, x=col, y="confianza", markers=True, color_discrete_sequence=[COLOR_CYAN])
        fig.update_traces(line_width=2.5, marker=dict(size=6))
        fig.update_xaxes(type="category")
        items.append(("r5_conf", estilo(fig, f"Confianza promedio por {nivel.lower()} (%)", x=nivel, y="Confianza %")))

    tm = resumen(df.dropna(subset=["anio", "trimestre", "nombre_mes"]), ["anio", "trimestre", "nombre_mes"])
    tm = tm[tm["escaneos"] > 0]
    if tm.empty:
        items.append(("r5_tree", sin_datos("Jerarquía de tiempo")))
    else:
        fig = px.treemap(tm, path=["anio", "trimestre", "nombre_mes"], values="escaneos",
                         color_discrete_sequence=PALETA)
        fig.update_traces(textinfo="label+value",
                          textfont=dict(color="#ffffff", size=12),
                          marker=dict(line=dict(color="#120720", width=1)))
        items.append(("r5_tree", estilo(fig, "Año → trimestre → mes")))
    rejilla(items)

    hallazgo(f"El {nivel.lower()} con más escaneos es <b>{escape(str(pico[col]))}</b> ({fmt_int(pico['escaneos'])})."
             + (f" La mayor incidencia de plaga o enfermedad ocurre en <b>{escape(str(pico_inc.idxmax()))}</b> ({fmt_int(pico_inc.max())})."
                if pico_inc is not None else ""))


def pagina_r6(df):
    d = sin_desc(df, "modelo")
    if d.empty:
        st.info(sin_datos("R6"))
        return
    rv = resumen(d, "modelo")
    ra = resumen(sin_desc(d, "algoritmo"), "algoritmo")
    mejor_v, _ = _mejor_peor(rv)
    mejor_a, _ = _mejor_peor(ra)

    k = st.columns(4)
    kpi(k[0], "Algoritmos", str(ra["algoritmo"].nunique()),
        ", ".join(escape(str(a)) for a in ra["algoritmo"].head(3)), color=COLOR_CYAN)
    kpi(k[1], "Versiones", str(rv["modelo"].nunique()), "sin contar «desconocido»", color=COLOR_PURPLE)
    kpi(k[2], "Mejor algoritmo", escape(str(mejor_a["algoritmo"])) if mejor_a is not None else "s/d",
        fmt_pct(mejor_a["acierto_pct"]) if mejor_a is not None else "", color=COLOR_GREEN)
    kpi(k[3], "Mejor versión", escape(str(mejor_v["modelo"])) if mejor_v is not None else "s/d",
        fmt_pct(mejor_v["acierto_pct"]) if mejor_v is not None else "", color=COLOR_MAGENTA)

    items = []
    rap = ra.dropna(subset=["acierto_pct"])
    if rap.empty:
        items.append(("r6_alg", sin_datos("Algoritmos")))
    else:
        largo = rap.melt(id_vars="algoritmo", value_vars=["acierto_pct", "confianza", "baja_pct"],
                         var_name="métrica", value_name="valor")
        largo["métrica"] = largo["métrica"].map({"acierto_pct": "Exactitud",
                                                  "confianza": "Confianza promedio",
                                                  "baja_pct": "Confianza baja"})
        fig = px.bar(largo, x="algoritmo", y="valor", color="métrica", barmode="group", text_auto=".0f",
                     color_discrete_sequence=[COLOR_CYAN, COLOR_MAGENTA, COLOR_ORANGE])
        fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=9))
        items.append(("r6_alg", estilo(fig, "Algoritmos: exactitud, confianza y confianza baja (%)", y="%")))

    rvv = rv.dropna(subset=["acierto_pct"]).merge(d[["modelo", "algoritmo"]].drop_duplicates(), on="modelo", how="left")
    if rvv.empty:
        items.append(("r6_ver", sin_datos("Versiones")))
    else:
        fig = barra_h(rvv, "acierto_pct", "modelo", "Exactitud por versión (%)", ".1f", "Exactitud %", color="algoritmo")
        fig.add_vline(x=META_EXACTITUD, line_dash="dash", line_color=COLOR_META, line_width=1.5)
        items.append(("r6_ver", fig))

    rs = rv.dropna(subset=["acierto_pct", "confianza"])
    if rs.empty:
        items.append(("r6_sc", sin_datos("Confianza vs. exactitud")))
    else:
        fig = px.scatter(rs, x="confianza", y="acierto_pct", size="escaneos", hover_name="modelo",
                         color_discrete_sequence=[COLOR_CYAN])
        fig.update_traces(marker=dict(line=dict(width=1, color="#ffffff")))
        items.append(("r6_sc", estilo(fig, "Confianza vs. exactitud por versión", x="Confianza %", y="Exactitud %")))

    rt = rv[rv["escaneos"] > 0]
    fig = px.treemap(rt, path=["modelo"], values="escaneos", color="modelo", color_discrete_sequence=PALETA)
    fig.update_traces(textinfo="label+value", textposition="top left",
                      textfont=dict(color="#ffffff", size=12),
                      marker=dict(line=dict(color="#120720", width=1)))
    items.append(("r6_tree", estilo(fig, "Escaneos por versión")))
    rejilla(items)

    if mejor_v is not None:
        texto = f"La versión con mayor exactitud es <b>{escape(str(mejor_v['modelo']))}</b> ({mejor_v['acierto_pct']:.1f}%)."
        if mejor_a is not None:
            texto += f" Entre algoritmos, destaca <b>{escape(str(mejor_a['algoritmo']))}</b> ({mejor_a['acierto_pct']:.1f}%)."
        texto += " Esta es la versión que conviene mantener en producción si cumple la meta."
        hallazgo(texto)


def pagina_decisiones(df):
    hall, recs = calcular_hallazgos(df)
    if not hall:
        st.info(sin_datos("Toma de decisiones"))
        return

    total = df["cantidad_escaneos"].sum()
    prec = precision(df)
    baja = pct_baja(df)
    inc = df.loc[es_incidencia(df), "cantidad_escaneos"].sum()
    inc_pct = inc / total * 100 if total else np.nan
    sin_verif = df.loc[df["acierto"].isna(), "cantidad_escaneos"].sum() / total * 100 if total else np.nan

    seccion("01", "Planteamiento del análisis")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            "<div class='info-card'><div class='lbl'>¿Qué queremos analizar?</div>"
            "<div class='txt'>El desempeño del detector inteligente del gusano cogollero y la roya del maíz a lo "
            "largo del tiempo: exactitud del modelo, incidencia por parcela, confianza de los diagnósticos y "
            "confiabilidad de las versiones desplegadas.</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='info-card'><div class='lbl'>¿Qué buscamos?</div>"
            "<div class='txt'>Identificar en qué parcelas y periodos intervenir, qué versión del modelo mantener "
            "en producción y cuándo conviene reentrenar, con base en evidencia estadística y no en percepciones.</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='info-card'><div class='lbl'>Decisión a tomar</div>"
            "<div class='txt'>Determinar si el modelo actual cumple la meta de exactitud, qué acciones tomar sobre "
            "las parcelas con mayor incidencia y si es necesario reentrenar o cambiar de versión del algoritmo.</div></div>",
            unsafe_allow_html=True,
        )

    seccion("02", "Análisis estadístico")
    k = st.columns(4)
    kpi(k[0], "Exactitud global", fmt_pct(prec), f"meta ≥ {META_EXACTITUD}%", color=COLOR_CYAN)
    kpi(k[1], "Incidencia (plaga + enfermedad)", fmt_pct(inc_pct),
        f"{fmt_int(inc)} de {fmt_int(total)}", color=COLOR_YELLOW)
    kpi(k[2], "Escaneos con confianza baja", fmt_pct(baja), f"umbral < {UMBRAL}%", color=COLOR_MAGENTA)
    kpi(k[3], "Escaneos sin verificar", fmt_pct(sin_verif), "sin clase real del experto", color=COLOR_ORANGE)

    st.markdown("**Indicadores de decisión**")
    st.table(tabla_decision(df))

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown("**Hallazgos clave**")
    for i in range(0, len(hall), 3):
        cols = st.columns(3, gap="medium")
        for col, (titulo, texto) in zip(cols, hall[i:i + 3]):
            col.markdown(
                f"<div class='card'><div class='card-t'>{titulo}</div><div class='card-x'>{texto}</div></div>",
                unsafe_allow_html=True,
            )

    seccion("03", "Decisión recomendada")
    if pd.isna(prec):
        principal = ("No hay datos verificados suficientes para calcular la exactitud del modelo. "
                     "Se recomienda completar la verificación del experto antes de tomar decisiones operativas.")
    elif prec >= META_EXACTITUD:
        principal = (f"El modelo <b>cumple</b> la meta de exactitud ({prec:.1f}% ≥ {META_EXACTITUD}%). "
                     "Se recomienda mantener la versión actual en producción y concentrar los esfuerzos en las "
                     "parcelas con mayor incidencia.")
    else:
        principal = (f"El modelo <b>no cumple</b> la meta de exactitud ({prec:.1f}% &lt; {META_EXACTITUD}%). "
                     "Se recomienda reentrenar antes de seguir usando el modelo como base para decisiones operativas.")
    st.markdown(f"<div class='reco-box'><b class='principal'>Decisión principal.</b> {principal}</div>",
                unsafe_allow_html=True)

    if recs:
        for n, texto in enumerate(recs, 1):
            st.markdown(f"<div class='reco-box'><b>Acción {n}.</b> {texto}</div>", unsafe_allow_html=True)


PAGINAS_FN = {
    "inicio": pagina_inicio, "r1": pagina_r1, "r2": pagina_r2, "r3": pagina_r3,
    "r4": pagina_r4, "r5": pagina_r5, "r6": pagina_r6, "decisiones": pagina_decisiones,
}

# ===== CARGA Y LAYOUT =====
try:
    df_all = cargar_datos()
except FileNotFoundError as e:
    st.error(f"No encuentro el archivo {Path(str(e.filename)).name}. Verifica que la carpeta data/ tenga los 5 CSV.")
    st.stop()

if "pagina" not in st.session_state:
    st.session_state.pagina = PAGINAS[0][0]


def ir_a(clave):
    st.session_state.pagina = clave


# Topbar full-width
topbar()

col_menu, col_main = st.columns([1.1, 4.8], gap="medium")

# Sidebar
with col_menu:
    with st.container(key="menu"):
        st.markdown(
            f"<div class='brand'>{logo_html()}"
            f"<div class='brand-n'>{NOMBRE}</div>"
            f"<div class='brand-s'>{SUBTITULO}</div></div>",
            unsafe_allow_html=True,
        )
        for clave, icono, etiqueta, _ in PAGINAS:
            boton(f"{icono}    {etiqueta}", key=f"nav_{clave}",
                  type="primary" if st.session_state.pagina == clave else "secondary",
                  on_click=ir_a, args=(clave,))
        st.markdown("<div class='filtros-t'>Filtros</div>", unsafe_allow_html=True)
        f_anio = st.multiselect("Año", sorted(df_all["anio"].dropna().unique().tolist()), key="f_anio")
        f_trim = st.multiselect("Trimestre", sorted(df_all["trimestre"].dropna().unique().tolist()), key="f_trim")
        f_parc = st.multiselect("Parcela", sorted(sin_desc(df_all, "parcela")["parcela"].unique().tolist()), key="f_parc")

df = df_all
if f_anio:
    df = df[df["anio"].isin(f_anio)]
if f_trim:
    df = df[df["trimestre"].isin(f_trim)]
if f_parc:
    df = df[df["parcela"].isin(f_parc)]

titulos = {clave: titulo for clave, _, _, titulo in PAGINAS}
with col_main:
    encabezado(st.session_state.pagina, titulos[st.session_state.pagina])
    if df.empty:
        st.warning("No hay datos con los filtros seleccionados. Quita algún filtro en el menú.")
    else:
        PAGINAS_FN[st.session_state.pagina](df)
    st.markdown(
        "<div class='pie'>Fuente: PostgreSQL · db_detector_maiz (modelo ROLAP) · Instituto Tecnológico de Tizimín</div>",
        unsafe_allow_html=True,
    )