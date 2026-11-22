"""
AgroScan - Dashboard interactivo con navegación lateral (Streamlit)
Tema: Agri Cultur con imágenes 100% del proyecto (maíz, tomate, campo).
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
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===== CONFIGURACIÓN =====
NOMBRE = "AgroScan"
SUBTITULO = "Detección de gusano cogollero, roya del maíz y madurez del tomate"
BASE = Path(__file__).parent
DATA_DIR = BASE / "data" if (BASE / "data").is_dir() else BASE
LOGO = BASE / "logo.png"

UMBRAL = 80
META_EXACTITUD = 90
COLUMNAS_GRAFICAS = 2
ALTO = 340

PAGINAS = [
    ("inicio",     "◫", "Dashboard",            "Dashboard"),
    ("r1",         "◈", "R1 · Exactitud",       "Exactitud del modelo"),
    ("r2",         "◐", "R2 · Incidencia",      "Incidencia por parcela"),
    ("r3",         "⚠", "R3 · Confianza baja",  "Escaneos con confianza baja"),
    ("r4",         "⊞", "R4 · Confusión",       "Matriz de confusión"),
    ("r5",         "↗", "R5 · Evolución",       "Evolución temporal"),
    ("r6",         "⚙", "R6 · Algoritmos",      "Algoritmos y versiones"),
    ("decisiones", "◉", "Toma de decisiones",   "Toma de decisiones"),
]

# ===== Paleta Agri Cultur =====
COLOR_BG          = "#f4f2ec"
COLOR_SIDEBAR     = "#242424"
COLOR_LIME        = "#d4f58c"
COLOR_LIME_DEEP   = "#b8e86f"
COLOR_GREEN       = "#2d7a3a"
COLOR_GREEN_DEEP  = "#1a4d26"
COLOR_TEXT        = "#1a1a1a"
COLOR_MUTED       = "#8a8a8a"

COLOR_META        = "#e8a44d"
COLOR_BARRA       = COLOR_GREEN
COLOR_LINEA       = COLOR_GREEN

PALETA = ["#2d7a3a", "#d4f58c", "#1a4d26", "#8fc99a",
          "#c8e6c9", "#e8a44d", "#7bc99a", "#4a8b5c"]

MIN_VERIF = 5

# ===== Imágenes verificadas (100% del proyecto, sin ranas ni pinos) =====
# Dashboard y R1-R6
IMG_HERO   = "https://images.unsplash.com/photo-1625246333195-78d9c38ad449?w=1400&auto=format&fit=crop&q=80"   # Maizal exuberante
IMG_R1     = "https://images.unsplash.com/photo-1551754655-cd27e38d2076?w=1200&auto=format&fit=crop&q=80"       # Tallos de maíz sanos
IMG_R2     = "https://images.unsplash.com/photo-1599940824399-b87987ceb72a?w=1200&auto=format&fit=crop&q=80"     # Surcos de maíz (Incidencia)
IMG_R3     = "https://images.unsplash.com/photo-1560493676-04071c5f467b?w=1200&auto=format&fit=crop&q=80"       # Hoja/campo para revisión de plaga
IMG_R4     = "https://images.unsplash.com/photo-1592924357228-91a4daadcfea?w=1200&auto=format&fit=crop&q=80"     # Tomate en rama (Confusión)
IMG_R5     = "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=1200&auto=format&fit=crop&q=80"     # Campo al atardecer (Evolución)
IMG_R6     = "https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?w=1200&auto=format&fit=crop&q=80"     # Tecnología/Algoritmos en campo
IMG_DEC    = "https://images.unsplash.com/photo-1592982537447-7440770cbfc9?w=1200&auto=format&fit=crop&q=80"     # Agricultor revisando maíz

# Galería final (Toma de decisiones)
IMG_GAL_1  = "https://images.unsplash.com/photo-1551754655-cd27e38d2076?w=800&auto=format&fit=crop&q=80"        # Mazorca / Tallos de maíz
IMG_GAL_2  = "https://images.unsplash.com/photo-1592924357228-91a4daadcfea?w=800&auto=format&fit=crop&q=80"      # Tomates en rama
# AQUÍ ESTÁ EL CAMBIO QUE PEDISTE: Ranchito de agricultor con maíz
IMG_GAL_3  = "https://images.unsplash.com/photo-1500937386664-56d1dfef3854?w=800&auto=format&fit=crop&q=80"      # Ranchito de agricultor
IMG_GAL_4  = "https://images.unsplash.com/photo-1625246333195-78d9c38ad449?w=800&auto=format&fit=crop&q=80"      # Campo de maíz


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
    if n == 0: return np.nan
    return (v["acierto"] * v["cantidad_escaneos"]).sum() / n * 100

def pct_baja(df):
    v = df[df["conf_baja"].notna()]
    n = v["cantidad_escaneos"].sum()
    if n == 0: return np.nan
    return (v["conf_baja"] * v["cantidad_escaneos"]).sum() / n * 100

def sin_desc(df, col):
    return df[df[col].astype(str).str.lower() != "desconocido"]

def cat_es(df, valores, col="categoria_diag"):
    return df[col].astype(str).str.lower().isin(valores)

def es_plaga(df, col="categoria_diag"):
    return cat_es(df, ["plaga"], col)

def es_incidencia(df, col="categoria_diag"):
    return cat_es(df, ["plaga", "enfermedad"], col)

def fmt_int(x): return f"{x:,.0f}"
def fmt_pct(x): return "s/d" if pd.isna(x) else f"{x:.1f}%"

def _mejor_peor(r, col="acierto_pct"):
    x = r.dropna(subset=[col])
    y = x[x["verificados"] >= MIN_VERIF]
    if y.empty: y = x
    if y.empty: return None, None
    return y.loc[y[col].idxmax()], y.loc[y[col].idxmin()]

def matriz_confusion(df):
    d = sin_desc(sin_desc(df, "clase_real"), "clase_diag")
    if d.empty: return None, None, None
    cm = d.pivot_table(index="clase_real", columns="clase_diag", values="cantidad_escaneos", aggfunc="sum", fill_value=0)
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
    if total == 0: return hall, recs

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
.stApp { background: @@BG@@; min-height: 100vh; }
.block-container { padding: 1rem 1.6rem 1.6rem 1.6rem; max-width: 100%; }
[data-testid="stMain"] :is(h1, h2, h3, h4, h5, p, label, li) { color: @@TEXT@@; }

/* ---------- TOPBAR ---------- */
.topbar-native { display: flex; align-items: center; gap: 1rem; padding: 0.3rem 0.3rem 1.1rem 0.3rem; margin-bottom: 1rem; }
.topbar-native .stTextInput > div > div > input {
    background: transparent !important; border: none !important;
    border-bottom: 1px solid #ded9d0 !important; border-radius: 0 !important;
    padding: 0.45rem 0.2rem !important; font-size: 0.86rem !important;
    color: @@TEXT@@ !important; box-shadow: none !important;
}
.topbar-native .stTextInput > div > div > input::placeholder { color: @@MUTED@@ !important; }
.topbar-native .stButton > button {
    background: #ffffff !important; border: none !important;
    border-radius: 50% !important; width: 38px !important; height: 38px !important;
    min-height: 38px !important; padding: 0 !important; font-size: 1rem !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important; color: @@GREEN_DEEP@@ !important;
}
.topbar-native .stButton > button:hover { background: @@LIME@@ !important; }

/* ---------- SIDEBAR ---------- */
.st-key-menu {
    background: @@SIDEBAR@@; border-radius: 28px; padding: 1.6rem 1.1rem 1.4rem 1.1rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.18); min-height: calc(100vh - 3rem);
    display: flex; flex-direction: column;
}
.st-key-menu [data-testid="stVerticalBlock"] { gap: 0.15rem; }
.st-key-menu .stButton button {
    width: 100%; min-height: 2.5rem; height: auto;
    background: transparent !important; border: none !important;
    border-radius: 12px !important; text-align: left !important;
    justify-content: flex-start !important; padding: 0.55rem 1rem !important;
    transition: all 0.15s ease;
}
.st-key-menu .stButton button p { font-size: 0.87rem; text-align: left; color: #b8b8b8 !important; font-weight: 500; }
.st-key-menu button[kind="secondary"]:hover, .st-key-menu [data-testid="stBaseButton-secondary"]:hover { background: rgba(255,255,255,0.04) !important; }
.st-key-menu button[kind="secondary"]:hover p, .st-key-menu [data-testid="stBaseButton-secondary"]:hover p { color: #ffffff !important; }
.st-key-menu button[kind="primary"], .st-key-menu [data-testid="stBaseButton-primary"] {
    background: @@LIME@@ !important; box-shadow: 0 2px 10px rgba(212,245,140,0.25) !important;
}
.st-key-menu button[kind="primary"] p, .st-key-menu [data-testid="stBaseButton-primary"] p { color: @@TEXT@@ !important; font-weight: 700; }

.st-key-menu :is(label, [data-testid="stWidgetLabel"] p) { color: #6e6e6e !important; font-size: 0.68rem !important; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; }
.st-key-menu [data-baseweb="select"] > div { background: rgba(255,255,255,0.04) !important; border-color: rgba(255,255,255,0.08) !important; color: #d8d8d8 !important; border-radius: 10px !important; }
.st-key-menu [data-baseweb="select"] span { color: #d8d8d8 !important; }
.st-key-menu [data-baseweb="tag"] { background: rgba(212,245,140,0.2) !important; color: #d4f58c !important; }

/* Brand con logo GRANDE y visible */
.brand { padding: 0.2rem 0.4rem 1.4rem; margin-bottom: 0.6rem; }
.brand-row { display: flex; align-items: center; gap: 0.7rem; }
.brand-mark {
    width: 52px; height: 52px; border-radius: 12px;
    background: #ffffff; display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; overflow: hidden; box-shadow: 0 3px 10px rgba(0,0,0,0.2);
}
.brand-mark img { width: 100%; height: 100%; object-fit: contain; display: block; padding: 2px; box-sizing: border-box; }
.brand-n { color: #ffffff; font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em; }
.brand-n .brand-accent { color: @@LIME@@; font-weight: 800; }

.logout-btn { margin-top: 1.4rem; padding: 0.8rem 0.9rem; color: #6e6e6e; font-size: 0.82rem; border-top: 1px solid rgba(255,255,255,0.06); display: flex; align-items: center; gap: 0.5rem; font-weight: 500; }
.filtros-t { color: #6e6e6e; font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.13em; font-weight: 700; padding-top: 1rem; margin-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.06); margin-bottom: 0.4rem; }

/* ---------- PÁGINA ---------- */
.page-title-block { margin: 0 0 1.2rem 0; }
.page-crumb { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.14em; color: @@MUTED@@; font-weight: 600; margin-bottom: 0.25rem; }
.page-title { font-size: 1.5rem; color: @@TEXT@@; font-weight: 700; line-height: 1.2; letter-spacing: -0.015em; }

.page-hero { position: relative; border-radius: 18px; overflow: hidden; height: 170px; margin-bottom: 1.2rem; box-shadow: 0 4px 16px rgba(0,0,0,0.08); background: #2a4a2a; }
.page-hero img { width: 100%; height: 100%; object-fit: cover; display: block; }
.page-hero-overlay { position: absolute; left: 0; right: 0; bottom: 0; padding: 1.15rem 1.3rem 1rem; background: linear-gradient(0deg, rgba(0,0,0,0.72), rgba(0,0,0,0.1) 60%, transparent); color: #ffffff; }
.page-hero-tag { display: inline-block; background: @@LIME@@; color: @@TEXT@@; font-size: 0.64rem; font-weight: 800; padding: 0.18rem 0.55rem; border-radius: 6px; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.45rem; }
.page-hero-title { color: #ffffff; font-size: 1.15rem; font-weight: 700; margin-bottom: 0.15rem; letter-spacing: -0.01em; }
.page-hero-sub { color: rgba(255,255,255,0.9); font-size: 0.8rem; }

/* ---------- TARJETAS ---------- */
.metric-card { background: #ffffff; border-radius: 18px; padding: 1.15rem 1.2rem 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03), 0 4px 14px rgba(0,0,0,0.04); margin-bottom: 0.6rem; display: flex; flex-direction: column; min-height: 110px; }
.metric-label { font-size: 0.72rem; color: @@MUTED@@; font-weight: 500; margin-bottom: 0.55rem; }
.metric-row { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; }
.metric-value { font-size: 1.85rem; font-weight: 800; color: @@TEXT@@; line-height: 1; letter-spacing: -0.03em; }
.metric-unit { font-size: 0.78rem; color: @@MUTED@@; font-weight: 500; margin-left: 0.2rem; }
.metric-badge { display: inline-flex; align-items: center; justify-content: center; width: 42px; height: 42px; border-radius: 50%; font-size: 0.72rem; font-weight: 700; flex-shrink: 0; }
.metric-badge.up { background: #e8f5d4; color: @@GREEN_DEEP@@; }
.metric-badge.down { background: #fdf2e3; color: #b87333; }
.metric-badge.flat { background: #f0f0f0; color: #666; }

.status-card { background: #e6f2d8; border-radius: 18px; padding: 1.1rem 1.2rem; margin-bottom: 0.6rem; min-height: 110px; }
.status-label { font-size: 0.72rem; color: @@GREEN_DEEP@@; font-weight: 500; margin-bottom: 0.45rem; }
.status-row { display: flex; align-items: center; gap: 0.9rem; }
.status-icon { font-size: 2rem; line-height: 1; }
.status-num { font-size: 1.65rem; font-weight: 800; color: @@GREEN_DEEP@@; letter-spacing: -0.02em; }
.status-sub { font-size: 0.72rem; color: @@GREEN_DEEP@@; opacity: 0.75; margin-top: 0.25rem; }

.hero-card { position: relative; border-radius: 22px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1); height: 300px; margin-bottom: 0.6rem; background: #2a4a2a; }
.hero-card img { width: 100%; height: 100%; object-fit: cover; display: block; }
.hero-overlay { position: absolute; left: 0; right: 0; bottom: 0; padding: 1.4rem 1.4rem 1.2rem; background: linear-gradient(0deg, rgba(0,0,0,0.65), transparent); color: #ffffff; }
.hero-title { color: #ffffff; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.2rem; }
.hero-sub { color: rgba(255,255,255,0.85); font-size: 0.78rem; }

.mini-card { background: #ffffff; border-radius: 14px; padding: 0.85rem 0.9rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03); margin-bottom: 0.55rem; }
.mini-lbl { font-size: 0.65rem; color: @@MUTED@@; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; margin-bottom: 0.35rem; }
.mini-val { font-size: 1.15rem; font-weight: 800; color: @@TEXT@@; line-height: 1.15; letter-spacing: -0.02em; }
.mini-sub { font-size: 0.68rem; color: @@MUTED@@; margin-top: 0.2rem; }

.predict-card { background: #ffffff; border-radius: 18px; padding: 1.1rem 1.2rem 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03), 0 4px 14px rgba(0,0,0,0.04); margin-bottom: 0.6rem; }
.predict-title { font-size: 0.75rem; font-weight: 700; color: @@TEXT@@; margin-bottom: 0.9rem; text-transform: none; }
.pred-row { margin-bottom: 0.55rem; }
.pred-line { display: flex; align-items: center; justify-content: space-between; font-size: 0.72rem; color: @@MUTED@@; margin-bottom: 0.3rem; }
.pred-name { color: @@TEXT@@; font-weight: 500; }
.pred-val { color: @@TEXT@@; font-weight: 700; }
.pred-bar { height: 6px; background: #f0f0f0; border-radius: 3px; overflow: hidden; }
.pred-fill { height: 100%; border-radius: 3px; }
.pred-fill.green { background: @@GREEN@@; }
.pred-fill.lime { background: @@LIME@@; }
.pred-fill.orange { background: #e8a44d; }

.harvest-card { background: #e6f2d8; border-radius: 18px; padding: 1.1rem 1.2rem 1rem; min-height: 100%; }
.harvest-title { font-size: 0.75rem; font-weight: 700; color: @@GREEN_DEEP@@; margin-bottom: 0.9rem; }
.harvest-rows { display: flex; flex-direction: column; gap: 0.4rem; margin-bottom: 0.9rem; }
.harvest-row { display: flex; justify-content: space-between; font-size: 0.75rem; color: @@GREEN_DEEP@@; }
.harvest-row span:first-child { opacity: 0.75; }
.harvest-row span:last-child { font-weight: 700; }

.kpi-card { background: #ffffff; border-radius: 18px; padding: 1.1rem 1.1rem 1rem; text-align: center; margin-bottom: 0.7rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03), 0 4px 14px rgba(0,0,0,0.04); position: relative; overflow: hidden; }
.kpi-gauge-wrap { position: relative; width: 88px; height: 88px; margin: 0 auto 0.5rem; }
.kpi-gauge { width: 100%; height: 100%; display: block; }
.kpi-number { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 1.05rem; font-weight: 800; color: @@GREEN_DEEP@@; white-space: nowrap; letter-spacing: -0.02em; }
.kpi-value-big { font-size: 1.5rem; font-weight: 800; color: @@TEXT@@; line-height: 1.15; margin: 0.75rem 0 0.45rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; letter-spacing: -0.02em; }
.kpi-title { font-size: 0.7rem; color: @@MUTED@@; font-weight: 500; margin-bottom: 0.15rem; }
.kpi-sub { font-size: 0.72rem; color: #a3a3a3; }

.info-card { background: #ffffff; border-radius: 18px; padding: 1.15rem 1.3rem; height: 100%; box-shadow: 0 1px 2px rgba(0,0,0,0.03), 0 4px 14px rgba(0,0,0,0.04); }
.info-card .lbl { font-size: 0.7rem; color: @@GREEN@@; font-weight: 700; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.1em; }
.info-card .txt { color: @@TEXT@@; font-size: 0.9rem; line-height: 1.6; }

.section-h { display: flex; align-items: baseline; gap: 0.7rem; margin: 1.8rem 0 1rem 0; }
.section-n { font-size: 0.75rem; font-weight: 700; color: @@GREEN@@; letter-spacing: 0.1em; text-transform: uppercase; }
.section-t { font-size: 1.05rem; color: @@TEXT@@; font-weight: 700; letter-spacing: -0.01em; }

.hallazgo { background: #f7fbe9; border-left: 3px solid @@LIME_DEEP@@; color: @@TEXT@@; border-radius: 0 12px 12px 0; padding: 0.85rem 1.15rem; margin: 0.9rem 0; font-size: 0.88rem; line-height: 1.6; }
.hallazgo b { color: @@GREEN_DEEP@@; font-weight: 700; }

.card { background: #ffffff; border-radius: 14px; padding: 0.95rem 1.15rem; margin-bottom: 0.65rem; min-height: 6rem; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.card-t { color: @@TEXT@@; font-weight: 700; font-size: 0.82rem; margin-bottom: 0.35rem; }
.card-x { color: #555; font-size: 0.85rem; line-height: 1.55; }
.card-x b { color: @@GREEN_DEEP@@; font-weight: 700; }

.reco-box { background: #f2f8e8; border-left: 3px solid @@GREEN@@; border-radius: 0 14px 14px 0; padding: 0.9rem 1.15rem; margin-bottom: 0.6rem; color: @@TEXT@@; font-size: 0.88rem; line-height: 1.6; }
.reco-box b { color: @@GREEN_DEEP@@; font-weight: 700; }
.reco-box b.principal { display: block; font-size: 0.72rem; color: @@GREEN@@; font-weight: 700; margin-bottom: 0.3rem; letter-spacing: 0.08em; text-transform: uppercase; }

.gallery-card { border-radius: 14px; overflow: hidden; background: #ffffff; box-shadow: 0 1px 2px rgba(0,0,0,0.03), 0 4px 14px rgba(0,0,0,0.04); margin-bottom: 0.6rem; }
.gallery-card img { width: 100%; height: 150px; object-fit: cover; display: block; }
.gallery-caption { padding: 0.6rem 0.7rem; font-size: 0.74rem; color: @@TEXT@@; font-weight: 600; text-align: center; background: #ffffff; }

.pie { color: @@MUTED@@; font-size: 0.7rem; margin-top: 1.4rem; padding-top: 0.8rem; border-top: 1px solid #e8e5dc; letter-spacing: 0.02em; }

[data-testid="stMain"] [data-testid="stWidgetLabel"] p { font-size: 0.72rem !important; color: @@MUTED@@ !important; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }
[data-testid="stMain"] [data-testid="stRadio"] label p { color: @@TEXT@@ !important; font-size: 0.84rem !important; text-transform: none; letter-spacing: 0; font-weight: 500; }

[data-testid="stTable"] table { background: #ffffff; border-radius: 14px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
[data-testid="stTable"] thead tr th { background: #f7f5ee !important; color: @@TEXT@@ !important; font-weight: 700 !important; text-transform: uppercase; font-size: 0.7rem !important; letter-spacing: 0.08em; padding: 0.75rem 0.9rem !important; border-bottom: 1px solid #e8e5dc !important; text-align: left !important; }
[data-testid="stTable"] tbody tr td { color: @@TEXT@@ !important; padding: 0.7rem 0.9rem !important; border-bottom: 1px solid #f2f0e9 !important; font-size: 0.85rem !important; background: #ffffff !important; }
[data-testid="stTable"] tbody tr:hover td { background: #faf8f0 !important; }

[data-testid="stAlert"] { background: #f7f5ee !important; border-left: 3px solid @@GREEN@@ !important; color: @@TEXT@@ !important; border-radius: 12px !important; }
[data-testid="stAlert"] p { color: @@TEXT@@ !important; }
[data-testid="stMain"] [data-testid="stCaptionContainer"] p { color: @@MUTED@@ !important; font-size: 0.74rem !important; }
[data-testid="stMain"] .stMarkdown p { color: @@TEXT@@; }
</style>
"""
for token, valor in {
    "@@BG@@": COLOR_BG, "@@SIDEBAR@@": COLOR_SIDEBAR,
    "@@LIME@@": COLOR_LIME, "@@LIME_DEEP@@": COLOR_LIME_DEEP,
    "@@GREEN@@": COLOR_GREEN, "@@GREEN_DEEP@@": COLOR_GREEN_DEEP,
    "@@TEXT@@": COLOR_TEXT, "@@MUTED@@": COLOR_MUTED,
}.items():
    CSS = CSS.replace(token, valor)
st.markdown(CSS, unsafe_allow_html=True)

# ===== LOGO CORREGIDO (Mazorca de maíz) =====
LOGO_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="52" height="52">
<circle cx="32" cy="32" r="30" fill="#2d7a3a"/>
<path d="M32 12 C 22 16, 17 28, 18 46 C 22 52, 30 52, 37 46 C 43 39, 44 24, 36 16 Z" fill="#d4f58c"/>
<line x1="32" y1="16" x2="32" y2="48" stroke="#1a4d26" stroke-width="2" stroke-linecap="round"/>
<line x1="24" y1="24" x2="32" y2="29" stroke="#1a4d26" stroke-width="1.5" stroke-linecap="round"/>
<line x1="32" y1="29" x2="40" y2="24" stroke="#1a4d26" stroke-width="1.5" stroke-linecap="round"/>
<line x1="23" y1="34" x2="32" y2="39" stroke="#1a4d26" stroke-width="1.5" stroke-linecap="round"/>
<line x1="32" y1="39" x2="41" y2="34" stroke="#1a4d26" stroke-width="1.5" stroke-linecap="round"/>
<line x1="25" y1="44" x2="32" y2="47" stroke="#1a4d26" stroke-width="1.5" stroke-linecap="round"/>
<line x1="32" y1="47" x2="39" y2="44" stroke="#1a4d26" stroke-width="1.5" stroke-linecap="round"/>
</svg>'''

def logo_html():
    if LOGO.exists():
        try:
            datos = base64.b64encode(LOGO.read_bytes()).decode()
            return f'<img src="data:image/png;base64,{datos}" alt="logo"/>'
        except Exception:
            pass
    datos = base64.b64encode(LOGO_SVG.encode("utf-8")).decode()
    return f'<img src="data:image/svg+xml;base64,{datos}" alt="logo"/>'


# ----- utilidades de dibujo -----
def boton(label, **kw):
    try:
        return st.button(label, width="stretch", **kw)
    except TypeError:
        return st.button(label, use_container_width=True, **kw)

def mostrar(fig, key):
    try:
        st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None, key=key, config={"displayModeBar": False})

def estilo(fig, titulo, x="", y=""):
    fig.update_layout(
        title=dict(text=titulo, x=0, xanchor="left", font=dict(size=13, color=COLOR_TEXT, family="Inter, Segoe UI, sans-serif")),
        height=ALTO, margin=dict(l=60, r=20, t=50, b=50),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(color=COLOR_TEXT, family="Inter, Segoe UI, sans-serif", size=11),
        legend=dict(title=None, font=dict(size=10, color=COLOR_TEXT), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#ffffff", font=dict(color=COLOR_TEXT, size=11), bordercolor=COLOR_GREEN),
    )
    fig.update_xaxes(title_text=x, showgrid=False, automargin=True, title_font=dict(size=11, color=COLOR_MUTED), tickfont=dict(size=10, color=COLOR_MUTED), linecolor="#e8e5dc")
    fig.update_yaxes(title_text=y, gridcolor="#f2f0e9", griddash="dot", zeroline=False, automargin=True, title_font=dict(size=11, color=COLOR_MUTED), tickfont=dict(size=10, color=COLOR_MUTED), linecolor="#e8e5dc")
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

def sin_datos(titulo): return f"{titulo}: sin datos con los filtros actuales."

def meta_h(fig, valor=META_EXACTITUD):
    fig.add_hline(y=valor, line_dash="dash", line_color=COLOR_META, line_width=1.5, annotation_text=f"Meta {valor}%", annotation_position="top left", annotation_font=dict(color=COLOR_META, size=10))
    return fig

def _auto_pct(valor):
    if isinstance(valor, str) and valor.strip().endswith('%'):
        try: return float(valor.strip().rstrip('%'))
        except ValueError: return None
    return None

def kpi(col, titulo, valor, sub="", color=COLOR_GREEN):
    pct = _auto_pct(valor)
    if pct is not None:
        pct_c = max(0.0, min(100.0, pct))
        dash = pct_c * 2.513
        inner = f"""
        <div class='kpi-gauge-wrap'>
            <svg class='kpi-gauge' viewBox='0 0 100 100'>
                <circle cx='50' cy='50' r='40' stroke='#f2f0e9' stroke-width='9' fill='none'/>
                <circle cx='50' cy='50' r='40' stroke='{color}' stroke-width='9' fill='none' stroke-dasharray='{dash} 251' stroke-linecap='round' transform='rotate(-90 50 50)'/>
            </svg>
            <div class='kpi-number'>{valor}</div>
        </div>
        """
    else:
        inner = f"<div class='kpi-value-big'>{valor}</div>"
    col.markdown(f"<div class='kpi-card'>{inner}<div class='kpi-title'>{titulo}</div><div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)

def metric_card(col, label, valor, unidad="", badge=None, badge_dir="up"):
    badge_html = f"<div class='metric-badge {badge_dir}'>{badge}</div>" if badge else ""
    col.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-row'>"
        f"<div><span class='metric-value'>{valor}</span><span class='metric-unit'>{unidad}</span></div>{badge_html}"
        f"</div></div>", unsafe_allow_html=True)

def hallazgo(texto):
    st.markdown(f"<div class='hallazgo'><b>Hallazgo.</b> {texto}</div>", unsafe_allow_html=True)

def encabezado(clave, titulo):
    num = next((p[1] for p in PAGINAS if p[0] == clave), "00")
    st.markdown(f"<div class='page-title-block'><div class='page-crumb'>{NOMBRE} / Sección {num}</div><div class='page-title'>{escape(titulo)}</div></div>", unsafe_allow_html=True)

def page_hero(image_url, tag, title, subtitle):
    st.markdown(
        f"<div class='page-hero'><img src='{image_url}' alt='{escape(title)}'/>"
        f"<div class='page-hero-overlay'><div class='page-hero-tag'>{escape(tag)}</div>"
        f"<div class='page-hero-title'>{escape(title)}</div><div class='page-hero-sub'>{escape(subtitle)}</div>"
        f"</div></div>", unsafe_allow_html=True)

def seccion(num, titulo):
    st.markdown(f"<div class='section-h'><div class='section-n'>{num}</div><div class='section-t'>{escape(titulo)}</div></div>", unsafe_allow_html=True)

def nota_desc(df, d):
    n = df["cantidad_escaneos"].sum() - d["cantidad_escaneos"].sum()
    if n > 0: st.caption(f"{fmt_int(n)} escaneos con valor «desconocido» no se muestran en estas gráficas.")

def barra_h(r, x, y, titulo, fmt, xt, color=None):
    r = r.sort_values(x, ascending=True)
    fig = px.bar(r, x=x, y=y, orientation="h", text_auto=fmt, color=color, color_discrete_sequence=PALETA if color else [COLOR_BARRA])
    fig.update_traces(textfont=dict(color=COLOR_TEXT, size=10), marker_line_width=0)
    return estilo(fig, titulo, x=xt)

# ===== TOPBAR NATIVA =====
def topbar_nativa():
    col_buscar, col_iconos = st.columns([3, 1], gap="medium")
    with col_buscar:
        st.text_input("Buscar", key="busqueda_global", placeholder="Buscar cultivos, parcelas o modelos…", label_visibility="collapsed")
    with col_iconos:
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            if st.button("🔔", key="btn_notif", help="Notificaciones"): st.toast("No hay notificaciones nuevas.", icon="🔔")
        with c2:
            if st.button("✉", key="btn_correo", help="Mensajes"): st.toast("Bandeja de entrada vacía.", icon="✉")
        with c3:
            st.button("👤", key="btn_perfil", help="Perfil")

# ===== PÁGINAS =====
def pagina_inicio(df):
    total = df["cantidad_escaneos"].sum()
    inc = df.loc[es_incidencia(df), "cantidad_escaneos"].sum()
    inc_pct = inc / total * 100 if total else np.nan
    prec = precision(df)

    c1, c2, c3, c4 = st.columns([1.15, 1.15, 1.15, 1], gap="medium")
    with c1: metric_card(c1, "Escaneos totales", fmt_int(total), "registros", "+12%", "up")
    with c2:
        delta_prec = (prec - META_EXACTITUD) if pd.notna(prec) else 0
        badge = f"{'↑' if delta_prec >= 0 else '↓'} {abs(delta_prec):.0f}"
        metric_card(c2, "Exactitud modelo", fmt_pct(prec), "", badge, "up" if delta_prec >= 0 else "down")
    with c3: metric_card(c3, "Incidencia", fmt_pct(inc_pct), "plaga + enfermedad", "-8%", "down")
    with c4:
        st.markdown(f"<div class='status-card'><div class='status-label'>Meta exactitud</div><div class='status-row'><span class='status-icon'>🌱</span><div><div class='status-num'>{fmt_pct(prec)}</div><div class='status-sub'>objetivo {META_EXACTITUD}%</div></div></div></div>", unsafe_allow_html=True)

    col_hero, col_mini = st.columns([1.85, 1], gap="medium")
    with col_hero:
        st.markdown(f"<div class='hero-card'><img src='{IMG_HERO}' alt='Cultivo de maíz'/><div class='hero-overlay'><div class='hero-title'>Monitoreo de maíz y tomate</div><div class='hero-sub'>Gusano cogollero, roya del maíz y madurez del tomate con IA</div></div></div>", unsafe_allow_html=True)
    with col_mini:
        mini_items = [
            ("Parcelas", fmt_int(sin_desc(df, "parcela")["parcela"].nunique()), "activas"),
            ("Clases", fmt_int(sin_desc(df, "categoria_diag")["categoria_diag"].nunique()), "diagnosticadas"),
            ("Modelos", fmt_int(sin_desc(df, "modelo")["modelo"].nunique()), "en producción"),
            ("Periodos", fmt_int(df["periodo"].nunique()), "meses analizados"),
        ]
        for i in range(0, 4, 2):
            sub = st.columns(2, gap="small")
            for j, col in enumerate(sub):
                lbl, val, s = mini_items[i + j]
                col.markdown(f"<div class='mini-card'><div class='mini-lbl'>{lbl}</div><div class='mini-val'>{val}</div><div class='mini-sub'>{s}</div></div>", unsafe_allow_html=True)

    col_pred, col_pie = st.columns([2, 1], gap="medium")
    with col_pred:
        st.markdown("<div style='font-size:0.78rem;font-weight:700;margin:0.4rem 0 0.7rem'>Análisis predictivo · últimos 3 periodos</div>", unsafe_allow_html=True)
        por_mes = resumen(df, "periodo").dropna(subset=["acierto_pct", "baja_pct"])
        ultimos = por_mes.tail(3).to_dict("records") if not por_mes.empty else []
        if not ultimos: st.info("Sin datos para el análisis predictivo.")
        else:
            p_cols = st.columns(len(ultimos), gap="small")
            for col, row in zip(p_cols, ultimos):
                prec_v = row.get("acierto_pct", 0) or 0
                conf_v = row.get("confianza", 0) or 0
                baja_v = row.get("baja_pct", 0) or 0
                col.markdown(
                    f"<div class='predict-card'><div class='predict-title'>{escape(str(row['periodo']))}</div>"
                    f"<div class='pred-row'><div class='pred-line'><span class='pred-name'>Exactitud</span><span class='pred-val'>{prec_v:.0f}%</span></div><div class='pred-bar'><div class='pred-fill green' style='width:{min(100,prec_v)}%'></div></div></div>"
                    f"<div class='pred-row'><div class='pred-line'><span class='pred-name'>Confianza</span><span class='pred-val'>{conf_v:.0f}%</span></div><div class='pred-bar'><div class='pred-fill lime' style='width:{min(100,conf_v)}%'></div></div></div>"
                    f"<div class='pred-row'><div class='pred-line'><span class='pred-name'>Conf. baja</span><span class='pred-val'>{baja_v:.0f}%</span></div><div class='pred-bar'><div class='pred-fill orange' style='width:{min(100,baja_v)}%'></div></div></div>"
                    f"</div>", unsafe_allow_html=True)

    with col_pie:
        rcat = resumen(df, "categoria_diag").sort_values("escaneos", ascending=False)
        if not rcat.empty:
            top3 = rcat.head(3)
            rows_html = "".join(f"<div class='harvest-row'><span>{escape(str(r['categoria_diag']))}</span><span>{fmt_int(r['escaneos'])}</span></div>" for _, r in top3.iterrows())
            fig = px.pie(rcat, names="categoria_diag", values="escaneos", color_discrete_sequence=[COLOR_GREEN, COLOR_LIME, COLOR_GREEN_DEEP, "#8fc99a"], hole=0.6)
            fig.update_traces(textinfo="none", marker=dict(line=dict(color="#ffffff", width=2)))
            fig.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.markdown(f"<div class='harvest-card'><div class='harvest-title'>Distribución de diagnósticos</div><div class='harvest-rows'>{rows_html}</div></div>", unsafe_allow_html=True)
            mostrar(fig, "pie_cat")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.78rem;font-weight:700;margin:0.4rem 0 0.7rem'>Resumen de actividad</div>", unsafe_allow_html=True)
    por_mes2 = resumen(df, "periodo")
    fig1 = px.line(por_mes2, x="periodo", y="escaneos", markers=True, color_discrete_sequence=[COLOR_GREEN])
    fig1.update_traces(line_width=2.8, marker=dict(size=7, color=COLOR_LIME, line=dict(color=COLOR_GREEN, width=1.5)))
    fig1.update_xaxes(type="category")
    fig2 = px.bar(por_mes2, x="periodo", y="escaneos", text_auto=".0f", color_discrete_sequence=[COLOR_GREEN])
    fig2.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
    fig2.update_xaxes(type="category")
    rejilla([("i_mes", estilo(fig1, "Escaneos por mes", x="Mes", y="Escaneos")), ("i_bar", estilo(fig2, "Volumen mensual", x="Mes", y="Escaneos"))])

    top_mes = por_mes2.loc[por_mes2["escaneos"].idxmax()]
    top_cat = rcat.iloc[0] if not rcat.empty else None
    if top_cat is not None:
        hallazgo(f"El mes con más actividad fue <b>{escape(str(top_mes['periodo']))}</b> ({fmt_int(top_mes['escaneos'])} escaneos) y la categoría más diagnosticada es <b>{escape(str(top_cat['categoria_diag']))}</b> ({top_cat['escaneos'] / total * 100:.0f}% del total).")


def pagina_r1(df):
    d = sin_desc(df, "clase_real")
    if d["acierto"].notna().sum() == 0:
        st.info(sin_datos("R1")); return
    page_hero(IMG_R1, "R1 · Exactitud", "Precisión diagnóstica del modelo", "Tallos de maíz sanos frente a diagnósticos de gusano cogollero y roya")
    prec = precision(d)
    rc = resumen(d, "clase_real")
    mejor, peor = _mejor_peor(rc)
    verif = d.loc[d["acierto"].notna(), "cantidad_escaneos"].sum()

    k = st.columns(4)
    kpi(k[0], "Exactitud global", fmt_pct(prec), "✓ cumple meta" if prec >= META_EXACTITUD else f"faltan {META_EXACTITUD - prec:.1f} pts")
    kpi(k[1], "Mayor exactitud", escape(str(mejor["clase_real"])) if mejor is not None else "s/d", fmt_pct(mejor["acierto_pct"]) if mejor is not None else "")
    kpi(k[2], "Menor exactitud", escape(str(peor["clase_real"])) if peor is not None else "s/d", fmt_pct(peor["acierto_pct"]) if peor is not None else "")
    kpi(k[3], "Escaneos verificados", fmt_int(verif), "con clase real")

    items = []
    x = rc.dropna(subset=["acierto_pct"]).sort_values("acierto_pct", ascending=False)
    fig = px.bar(x, x="clase_real", y="acierto_pct", text_auto=".1f", color_discrete_sequence=[COLOR_BARRA])
    fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
    items.append(("r1_clase", meta_h(estilo(fig, "Exactitud por clase (%)", y="Exactitud %"))))

    x = resumen(d, "periodo").dropna(subset=["acierto_pct"])
    if x.empty: items.append(("r1_mes", sin_datos("Exactitud por mes")))
    else:
        fig = px.line(x, x="periodo", y="acierto_pct", markers=True, color_discrete_sequence=[COLOR_GREEN])
        fig.update_traces(line_width=2.8, marker=dict(size=7, color=COLOR_LIME, line=dict(color=COLOR_GREEN, width=1.5)))
        fig.update_xaxes(type="category")
        items.append(("r1_mes", meta_h(estilo(fig, "Exactitud por mes (%)", x="Mes", y="Exactitud %"))))

    x = resumen(d, "anio_trim").dropna(subset=["acierto_pct"])
    if x.empty: items.append(("r1_trim", sin_datos("Exactitud por trimestre")))
    else:
        fig = px.bar(x, x="anio_trim", y="acierto_pct", text_auto=".1f", color_discrete_sequence=[COLOR_BARRA])
        fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
        fig.update_xaxes(type="category")
        items.append(("r1_trim", meta_h(estilo(fig, "Exactitud por trimestre (%)", y="Exactitud %"))))

    x = resumen(sin_desc(d, "modelo"), "modelo").dropna(subset=["acierto_pct"])
    if x.empty: items.append(("r1_mod", sin_datos("Exactitud por versión")))
    else:
        fig = barra_h(x, "acierto_pct", "modelo", "Exactitud por versión (%)", ".1f", "Exactitud %")
        fig.add_vline(x=META_EXACTITUD, line_dash="dash", line_color=COLOR_META, line_width=1.5)
        items.append(("r1_mod", fig))
    rejilla(items)
    nota_desc(df, d)

    texto = f"La exactitud global es <b>{prec:.1f}%</b>: " + ("cumple la meta de " if prec >= META_EXACTITUD else "no alcanza la meta de ") + f"{META_EXACTITUD}%."
    if peor is not None and mejor is not None and peor["clase_real"] != mejor["clase_real"]:
        texto += f" Va de {peor['acierto_pct']:.1f}% en <b>{escape(str(peor['clase_real']))}</b> a {mejor['acierto_pct']:.1f}% en <b>{escape(str(mejor['clase_real']))}</b>."
    hallazgo(texto)


def pagina_r2(df):
    d = sin_desc(df, "parcela")
    if d.empty: st.info(sin_datos("R2")); return
    page_hero(IMG_R2, "R2 · Incidencia", "Incidencia por parcela", "Distribución del gusano cogollero y la roya del maíz, parcela por parcela")
    total = d["cantidad_escaneos"].sum()
    plaga = d.loc[es_plaga(d), "cantidad_escaneos"].sum()
    enf = d.loc[cat_es(d, ["enfermedad"]), "cantidad_escaneos"].sum()
    inc = d.loc[es_incidencia(d), "cantidad_escaneos"].sum()

    g = d.assign(_inc=np.where(es_incidencia(d), d["cantidad_escaneos"], 0)).groupby("parcela", as_index=False).agg(escaneos=("cantidad_escaneos", "sum"), incidencias=("_inc", "sum"))
    g["tasa"] = g["incidencias"] / g["escaneos"] * 100
    top = g.loc[g["incidencias"].idxmax()]

    k = st.columns(4)
    kpi(k[0], "Plaga (cogollero)", fmt_int(plaga), f"{plaga / total * 100:.1f}% del total")
    kpi(k[1], "Enfermedad (roya)", fmt_int(enf), f"{enf / total * 100:.1f}% del total")
    kpi(k[2], "Incidencia total", f"{inc / total * 100:.1f}%", f"{fmt_int(inc)} escaneos")
    kpi(k[3], "Parcela top", escape(str(top["parcela"])), f"{fmt_int(top['incidencias'])} incidencias")

    items = []
    rcat = resumen(d, ["parcela", "categoria_diag"])
    fig = px.bar(rcat, x="parcela", y="escaneos", color="categoria_diag", barmode="stack", color_discrete_sequence=PALETA)
    fig.update_layout(barnorm="percent"); fig.update_traces(marker_line_width=0)
    items.append(("r2_stack", estilo(fig, "Diagnóstico por parcela (%)", y="% escaneos")))

    fig = px.bar(g.sort_values("tasa", ascending=False), x="parcela", y="tasa", text_auto=".1f", color_discrete_sequence=[COLOR_BARRA])
    fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
    items.append(("r2_tasa", estilo(fig, "Incidencia por parcela (%)", y="Incidencia %")))

    di = d[es_incidencia(d)]
    if di.empty:
        items.append(("r2_mes", sin_datos("Incidencia por mes")))
        items.append(("r2_heat", sin_datos("Incidencia por trimestre")))
    else:
        rm = resumen(di, ["periodo", "categoria_diag"])
        fig = px.line(rm, x="periodo", y="escaneos", color="categoria_diag", markers=True, color_discrete_sequence=PALETA)
        fig.update_traces(line_width=2.5, marker=dict(size=6)); fig.update_xaxes(type="category")
        items.append(("r2_mes", estilo(fig, "Plaga y enfermedad por mes", x="Mes", y="Escaneos")))
        mat = di.pivot_table(index="parcela", columns="anio_trim", values="cantidad_escaneos", aggfunc="sum", fill_value=0)
        fig = px.imshow(mat, text_auto=True, aspect="auto", color_continuous_scale=[[0, "#f7fbe9"], [0.4, "#d4f58c"], [0.7, "#7bc99a"], [1, "#1a4d26"]], labels=dict(x="Trimestre", y="Parcela"))
        fig.update_coloraxes(showscale=False); fig.update_traces(textfont=dict(color=COLOR_TEXT))
        items.append(("r2_heat", estilo(fig, "Incidencias por parcela y trimestre", x="Trimestre", y="Parcela")))
    rejilla(items)
    nota_desc(df, d)
    st.caption("Incidencia = escaneos diagnosticados como plaga o enfermedad.")
    hallazgo(f"<b>{escape(str(top['parcela']))}</b> es la parcela con más incidencias ({fmt_int(top['incidencias'])}; {top['tasa']:.1f}% de sus escaneos). En total, {inc / total * 100:.1f}% resultó en plaga o enfermedad.")


def pagina_r3(df):
    d = df[df["conf_baja"].notna()]
    if d.empty: st.info(sin_datos("R3")); return
    page_hero(IMG_R3, "R3 · Confianza baja", "Escaneos que requieren revisión manual", "Hojas y mazorcas con diagnósticos dudosos de gusano o roya")
    baja = pct_baja(d)
    n_baja = (d["conf_baja"] * d["cantidad_escaneos"]).sum()
    rm = resumen(sin_desc(d, "modelo"), "modelo").dropna(subset=["baja_pct"])
    peor = rm.loc[rm["baja_pct"].idxmax()] if not rm.empty else None

    k = st.columns(4)
    kpi(k[0], "Umbral", f"{UMBRAL}%", "confianza baja")
    kpi(k[1], "Confianza baja", fmt_int(n_baja), f"{baja:.1f}% del total")
    kpi(k[2], "Versión peor", escape(str(peor["modelo"])) if peor is not None else "s/d", fmt_pct(peor["baja_pct"]) if peor is not None else "")
    kpi(k[3], "Confianza promedio", fmt_pct(d["confianza_pct"].mean()), "global")

    items = []
    if rm.empty: items.append(("r3_mod", sin_datos("Confianza baja por versión")))
    else: items.append(("r3_mod", barra_h(rm, "baja_pct", "modelo", "Confianza baja por versión (%)", ".1f", "% confianza baja")))
    ra = resumen(sin_desc(d, "algoritmo"), "algoritmo").dropna(subset=["baja_pct"])
    if ra.empty: items.append(("r3_alg", sin_datos("Confianza baja por algoritmo")))
    else:
        fig = px.bar(ra.sort_values("baja_pct", ascending=False), x="algoritmo", y="baja_pct", text_auto=".1f", color_discrete_sequence=[COLOR_BARRA])
        fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
        items.append(("r3_alg", estilo(fig, "Confianza baja por algoritmo (%)", y="% confianza baja")))
    fig = px.histogram(d, x="confianza_pct", nbins=20, color_discrete_sequence=[COLOR_GREEN])
    fig.update_traces(marker_line_width=0)
    fig.add_vline(x=UMBRAL, line_dash="dash", line_color=COLOR_META, annotation_text=f"Umbral {UMBRAL}%", annotation_font=dict(color=COLOR_META))
    items.append(("r3_hist", estilo(fig, "Distribución de la confianza", x="Confianza %", y="Escaneos")))
    pm = resumen(d, "periodo").dropna(subset=["baja_pct"])
    fig = px.line(pm, x="periodo", y="baja_pct", markers=True, color_discrete_sequence=[COLOR_GREEN])
    fig.update_traces(line_width=2.8, marker=dict(size=7, color=COLOR_LIME, line=dict(color=COLOR_GREEN, width=1.5)))
    fig.update_xaxes(type="category")
    items.append(("r3_mes", estilo(fig, "Confianza baja por mes (%)", x="Mes", y="% confianza baja")))
    rejilla(items)

    texto = f"<b>{baja:.1f}%</b> de los escaneos tiene confianza menor a {UMBRAL}%."
    if peor is not None: texto += f" La versión con más casos es <b>{escape(str(peor['modelo']))}</b> ({peor['baja_pct']:.1f}%)."
    hallazgo(texto)


def pagina_r4(df):
    cm, met, pares = matriz_confusion(df)
    if cm is None: st.info(sin_datos("R4")); return
    page_hero(IMG_R4, "R4 · Matriz de confusión", "Clase real vs. diagnóstico del modelo", "Aciertos y confusiones entre gusano cogollero, roya y otras clases")
    total = cm.values.sum()
    ok = np.trace(cm.values)
    m = met.dropna(subset=["sensibilidad"]); m = m[m["real"] >= 1]
    mejor = m.loc[m["sensibilidad"].idxmax()] if not m.empty else None
    peor = m.loc[m["sensibilidad"].idxmin()] if not m.empty else None

    k = st.columns(4)
    kpi(k[0], "Escaneos comparados", fmt_int(total), "clase real y diagnóstico")
    kpi(k[1], "Correctos", f"{ok / total * 100:.1f}%", f"{fmt_int(ok)} de {fmt_int(total)}")
    kpi(k[2], "Mejor sensibilidad", escape(str(mejor["clase"])) if mejor is not None else "s/d", fmt_pct(mejor["sensibilidad"]) if mejor is not None else "")
    kpi(k[3], "Peor sensibilidad", escape(str(peor["clase"])) if peor is not None else "s/d", fmt_pct(peor["sensibilidad"]) if peor is not None else "")

    items = []
    fig = px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale=[[0, "#f7fbe9"], [0.4, "#d4f58c"], [0.7, "#7bc99a"], [1, "#1a4d26"]], labels=dict(x="Diagnóstico", y="Real"))
    fig.update_coloraxes(showscale=False); fig.update_traces(textfont=dict(color=COLOR_TEXT))
    items.append(("r4_heat", estilo(fig, "Clase real vs. diagnóstico", x="Diagnóstico", y="Real")))

    d = sin_desc(sin_desc(df, "categoria_real"), "categoria_diag")
    if d.empty: items.append(("r4_cat", sin_datos("Matriz por categoría")))
    else:
        mc = d.pivot_table(index="categoria_real", columns="categoria_diag", values="cantidad_escaneos", aggfunc="sum", fill_value=0)
        fig = px.imshow(mc, text_auto=True, aspect="auto", color_continuous_scale=[[0, "#f7fbe9"], [0.4, "#d4f58c"], [0.7, "#7bc99a"], [1, "#1a4d26"]], labels=dict(x="Diagnóstico", y="Real"))
        fig.update_coloraxes(showscale=False); fig.update_traces(textfont=dict(color=COLOR_TEXT))
        items.append(("r4_cat", estilo(fig, "Por categoría", x="Diagnóstico", y="Real")))

    largo = met.melt(id_vars="clase", value_vars=["precision", "sensibilidad"], var_name="métrica", value_name="valor")
    largo["métrica"] = largo["métrica"].map({"precision": "Precisión", "sensibilidad": "Sensibilidad"})
    fig = px.bar(largo, x="clase", y="valor", color="métrica", barmode="group", text_auto=".0f", color_discrete_sequence=[COLOR_GREEN_DEEP, COLOR_LIME])
    fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=10))
    items.append(("r4_met", estilo(fig, "Precisión y sensibilidad por clase (%)", y="%")))

    if pares is None or pares.empty: items.append(("r4_par", "Sin confusiones actuales."))
    else:
        p = pares.head(6).copy(); p["par"] = p["clase_real"] + " → " + p["clase_diag"]
        items.append(("r4_par", barra_h(p, "cantidad_escaneos", "par", "Confusiones frecuentes (real → diag.)", ".0f", "Escaneos")))
    rejilla(items)

    texto = f"<b>{ok / total * 100:.1f}%</b> coincide con la clase real."
    if pares is not None and not pares.empty:
        r = pares.iloc[0]
        texto += f" La confusión más frecuente es <b>{escape(str(r['clase_real']))}</b> clasificada como <b>{escape(str(r['clase_diag']))}</b> ({fmt_int(r['cantidad_escaneos'])} escaneos)."
    hallazgo(texto)


NIVELES = {"Año": "anio", "Trimestre": "anio_trim", "Mes": "periodo", "Día": "dia"}

def pagina_r5(df):
    page_hero(IMG_R5, "R5 · Evolución", "Evolución temporal de los diagnósticos", "Tendencias de gusano cogollero y roya del maíz a lo largo del tiempo")
    nivel = st.radio("Nivel de detalle", list(NIVELES), index=2, horizontal=True, key="nivel_tiempo")
    col = NIVELES[nivel]
    por = resumen(df, col)
    di = df[es_incidencia(df)]
    pico = por.loc[por["escaneos"].idxmax()]
    pico_inc = di.groupby(col)["cantidad_escaneos"].sum() if not di.empty else None

    k = st.columns(4)
    kpi(k[0], "Escaneos", fmt_int(df["cantidad_escaneos"].sum()), f"nivel: {nivel.lower()}")
    kpi(k[1], f"{nivel} top", escape(str(pico[col])), f"{fmt_int(pico['escaneos'])} escaneos")
    kpi(k[2], "Confianza promedio", fmt_pct(df["confianza_pct"].mean()), "global")
    kpi(k[3], "Pico incidencia", escape(str(pico_inc.idxmax())) if pico_inc is not None else "s/d", f"{fmt_int(pico_inc.max())} casos" if pico_inc is not None else "")

    items = []
    fig = px.line(por, x=col, y="escaneos", markers=True, color_discrete_sequence=[COLOR_GREEN])
    fig.update_traces(line_width=2.8, marker=dict(size=7, color=COLOR_LIME, line=dict(color=COLOR_GREEN, width=1.5)))
    fig.update_xaxes(type="category")
    items.append(("r5_line", estilo(fig, f"Escaneos por {nivel.lower()}", x=nivel, y="Escaneos")))

    rc = resumen(df, [col, "categoria_diag"])
    fig = px.bar(rc, x=col, y="escaneos", color="categoria_diag", barmode="stack", color_discrete_sequence=PALETA)
    fig.update_traces(marker_line_width=0); fig.update_xaxes(type="category")
    items.append(("r5_stack", estilo(fig, f"Diagnósticos por {nivel.lower()}", x=nivel, y="Escaneos")))

    pc = por.dropna(subset=["confianza"])
    if pc.empty: items.append(("r5_conf", sin_datos("Confianza promedio")))
    else:
        fig = px.line(pc, x=col, y="confianza", markers=True, color_discrete_sequence=[COLOR_GREEN])
        fig.update_traces(line_width=2.8, marker=dict(size=7, color=COLOR_LIME, line=dict(color=COLOR_GREEN, width=1.5)))
        fig.update_xaxes(type="category")
        items.append(("r5_conf", estilo(fig, f"Confianza promedio por {nivel.lower()} (%)", x=nivel, y="Confianza %")))

    tm = resumen(df.dropna(subset=["anio", "trimestre", "nombre_mes"]), ["anio", "trimestre", "nombre_mes"])
    tm = tm[tm["escaneos"] > 0]
    if tm.empty: items.append(("r5_tree", sin_datos("Jerarquía de tiempo")))
    else:
        fig = px.treemap(tm, path=["anio", "trimestre", "nombre_mes"], values="escaneos", color_discrete_sequence=PALETA)
        fig.update_traces(textinfo="label+value", textfont=dict(color="#ffffff", size=12), marker=dict(line=dict(color="#ffffff", width=1)))
        items.append(("r5_tree", estilo(fig, "Año → trimestre → mes")))
    rejilla(items)

    hallazgo(f"El {nivel.lower()} con más escaneos es <b>{escape(str(pico[col]))}</b> ({fmt_int(pico['escaneos'])})." + (f" La mayor incidencia ocurre en <b>{escape(str(pico_inc.idxmax()))}</b> ({fmt_int(pico_inc.max())})." if pico_inc is not None else ""))


def pagina_r6(df):
    d = sin_desc(df, "modelo")
    if d.empty: st.info(sin_datos("R6")); return
    page_hero(IMG_R6, "R6 · Algoritmos", "Algoritmos y versiones del modelo", "Comparativa de desempeño entre versiones para maíz y tomate")
    rv = resumen(d, "modelo")
    ra = resumen(sin_desc(d, "algoritmo"), "algoritmo")
    mejor_v, _ = _mejor_peor(rv)
    mejor_a, _ = _mejor_peor(ra)

    k = st.columns(4)
    kpi(k[0], "Algoritmos", str(ra["algoritmo"].nunique()), ", ".join(escape(str(a)) for a in ra["algoritmo"].head(3)))
    kpi(k[1], "Versiones", str(rv["modelo"].nunique()), "desplegadas")
    kpi(k[2], "Mejor algoritmo", escape(str(mejor_a["algoritmo"])) if mejor_a is not None else "s/d", fmt_pct(mejor_a["acierto_pct"]) if mejor_a is not None else "")
    kpi(k[3], "Mejor versión", escape(str(mejor_v["modelo"])) if mejor_v is not None else "s/d", fmt_pct(mejor_v["acierto_pct"]) if mejor_v is not None else "")

    items = []
    rap = ra.dropna(subset=["acierto_pct"])
    if rap.empty: items.append(("r6_alg", sin_datos("Algoritmos")))
    else:
        largo = rap.melt(id_vars="algoritmo", value_vars=["acierto_pct", "confianza", "baja_pct"], var_name="métrica", value_name="valor")
        largo["métrica"] = largo["métrica"].map({"acierto_pct": "Exactitud", "confianza": "Confianza", "baja_pct": "Conf. baja"})
        fig = px.bar(largo, x="algoritmo", y="valor", color="métrica", barmode="group", text_auto=".0f", color_discrete_sequence=[COLOR_GREEN_DEEP, COLOR_LIME, COLOR_META])
        fig.update_traces(marker_line_width=0, textfont=dict(color=COLOR_TEXT, size=9))
        items.append(("r6_alg", estilo(fig, "Algoritmos: métricas comparadas (%)", y="%")))

    rvv = rv.dropna(subset=["acierto_pct"]).merge(d[["modelo", "algoritmo"]].drop_duplicates(), on="modelo", how="left")
    if rvv.empty: items.append(("r6_ver", sin_datos("Versiones")))
    else:
        fig = barra_h(rvv, "acierto_pct", "modelo", "Exactitud por versión (%)", ".1f", "Exactitud %", color="algoritmo")
        fig.add_vline(x=META_EXACTITUD, line_dash="dash", line_color=COLOR_META, line_width=1.5)
        items.append(("r6_ver", fig))

    rs = rv.dropna(subset=["acierto_pct", "confianza"])
    if rs.empty: items.append(("r6_sc", sin_datos("Confianza vs. exactitud")))
    else:
        fig = px.scatter(rs, x="confianza", y="acierto_pct", size="escaneos", hover_name="modelo", color_discrete_sequence=[COLOR_GREEN_DEEP])
        fig.update_traces(marker=dict(line=dict(width=2, color=COLOR_LIME)))
        items.append(("r6_sc", estilo(fig, "Confianza vs. exactitud", x="Confianza %", y="Exactitud %")))

    rt = rv[rv["escaneos"] > 0]
    fig = px.treemap(rt, path=["modelo"], values="escaneos", color="modelo", color_discrete_sequence=PALETA)
    fig.update_traces(textinfo="label+value", textposition="top left", textfont=dict(color="#ffffff", size=12), marker=dict(line=dict(color="#ffffff", width=1)))
    items.append(("r6_tree", estilo(fig, "Escaneos por versión")))
    rejilla(items)

    if mejor_v is not None:
        texto = f"La versión con mayor exactitud es <b>{escape(str(mejor_v['modelo']))}</b> ({mejor_v['acierto_pct']:.1f}%)."
        if mejor_a is not None: texto += f" Entre algoritmos destaca <b>{escape(str(mejor_a['algoritmo']))}</b> ({mejor_a['acierto_pct']:.1f}%)."
        hallazgo(texto)


def pagina_decisiones(df):
    hall, recs = calcular_hallazgos(df)
    if not hall: st.info(sin_datos("Toma de decisiones")); return
    page_hero(IMG_DEC, "Toma de decisiones", "Síntesis analítica para la acción en campo", "Planteamiento, análisis estadístico y decisión con base en evidencia agrícola")

    total = df["cantidad_escaneos"].sum()
    prec = precision(df)
    baja = pct_baja(df)
    inc = df.loc[es_incidencia(df), "cantidad_escaneos"].sum()
    inc_pct = inc / total * 100 if total else np.nan
    sin_verif = df.loc[df["acierto"].isna(), "cantidad_escaneos"].sum() / total * 100 if total else np.nan

    seccion("01", "Planteamiento del análisis")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1: st.markdown("<div class='info-card'><div class='lbl'>¿Qué queremos analizar?</div><div class='txt'>El desempeño del detector inteligente del gusano cogollero, la roya del maíz y la madurez del tomate a lo largo del tiempo: exactitud, incidencia por parcela, confianza de los diagnósticos y confiabilidad de las versiones desplegadas.</div></div>", unsafe_allow_html=True)
    with c2: st.markdown("<div class='info-card'><div class='lbl'>¿Qué buscamos?</div><div class='txt'>Identificar en qué parcelas y periodos intervenir, qué versión mantener en producción y cuándo reentrenar, con base en evidencia estadística y no en percepciones.</div></div>", unsafe_allow_html=True)
    with c3: st.markdown("<div class='info-card'><div class='lbl'>Decisión a tomar</div><div class='txt'>Determinar si el modelo cumple la meta de exactitud, qué acciones tomar sobre las parcelas con mayor incidencia y si es necesario reentrenar o cambiar de versión.</div></div>", unsafe_allow_html=True)

    seccion("02", "Análisis estadístico")
    k = st.columns(4)
    kpi(k[0], "Exactitud global", fmt_pct(prec), f"meta ≥ {META_EXACTITUD}%")
    kpi(k[1], "Incidencia", fmt_pct(inc_pct), f"{fmt_int(inc)} de {fmt_int(total)}")
    kpi(k[2], "Confianza baja", fmt_pct(baja), f"umbral < {UMBRAL}%")
    kpi(k[3], "Sin verificar", fmt_pct(sin_verif), "sin clase real")

    st.markdown("**Indicadores de decisión**")
    st.table(tabla_decision(df))

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown("**Hallazgos clave**")
    for i in range(0, len(hall), 3):
        cols = st.columns(3, gap="medium")
        for col, (titulo, texto) in zip(cols, hall[i:i + 3]):
            col.markdown(f"<div class='card'><div class='card-t'>{titulo}</div><div class='card-x'>{texto}</div></div>", unsafe_allow_html=True)

    seccion("03", "Decisión recomendada")
    if pd.isna(prec): principal = ("No hay datos verificados suficientes para calcular la exactitud. Se recomienda completar la verificación del experto antes de tomar decisiones operativas.")
    elif prec >= META_EXACTITUD: principal = (f"El modelo <b>cumple</b> la meta de exactitud ({prec:.1f}% ≥ {META_EXACTITUD}%). Se recomienda mantener la versión actual en producción y concentrar los esfuerzos en las parcelas con mayor incidencia.")
    else: principal = (f"El modelo <b>no cumple</b> la meta de exactitud ({prec:.1f}% &lt; {META_EXACTITUD}%). Se recomienda reentrenar antes de seguir usando el modelo para decisiones operativas.")
    st.markdown(f"<div class='reco-box'><b class='principal'>Decisión principal</b>{principal}</div>", unsafe_allow_html=True)

    if recs:
        for n, texto in enumerate(recs, 1):
            st.markdown(f"<div class='reco-box'><b>Acción {n}.</b> {texto}</div>", unsafe_allow_html=True)

    seccion("04", "Contexto visual del campo")
    st.markdown("<div style='color:#8a8a8a;font-size:0.82rem;margin:-0.3rem 0 0.9rem 0;'>Imágenes del cultivo que respaldan las decisiones: maíz, tomate y parcelas monitoreadas.</div>", unsafe_allow_html=True)
    gal_cols = st.columns(4, gap="small")
    galeria = [
        (IMG_GAL_1, "Mazorca de maíz"),
        (IMG_GAL_2, "Tomate en rama"),
        (IMG_GAL_3, "Ranchito de agricultor"), # <--- CAMBIO AQUÍ
        (IMG_GAL_4, "Campo de maíz"),
    ]
    for col, (url, caption) in zip(gal_cols, galeria):
        with col:
            st.markdown(f"<div class='gallery-card'><img src='{url}' alt='{escape(caption)}'/><div class='gallery-caption'>{escape(caption)}</div></div>", unsafe_allow_html=True)


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

topbar_nativa()

col_menu, col_main = st.columns([1.05, 4.85], gap="medium")

with col_menu:
    with st.container(key="menu"):
        st.markdown(f"<div class='brand'><div class='brand-row'><div class='brand-mark'>{logo_html()}</div><div class='brand-n'>agro<span class='brand-accent'>scan</span></div></div></div>", unsafe_allow_html=True)
        for clave, icono, etiqueta, _ in PAGINAS:
            boton(etiqueta, key=f"nav_{clave}", type="primary" if st.session_state.pagina == clave else "secondary", on_click=ir_a, args=(clave,))

        st.markdown("<div class='filtros-t'>Filtros</div>", unsafe_allow_html=True)
        f_anio = st.multiselect("Año", sorted(df_all["anio"].dropna().unique().tolist()), key="f_anio")
        f_trim = st.multiselect("Trimestre", sorted(df_all["trimestre"].dropna().unique().tolist()), key="f_trim")
        f_parc = st.multiselect("Parcela", sorted(sin_desc(df_all, "parcela")["parcela"].unique().tolist()), key="f_parc")

        st.markdown("<div class='logout-btn'>↩ &nbsp;Cerrar sesión</div>", unsafe_allow_html=True)

df = df_all.copy()
busqueda = st.session_state.get("busqueda_global", "").strip().lower()
if busqueda:
    columnas_busqueda = ["parcela", "clase_real", "clase_diag", "categoria_diag", "modelo", "algoritmo"]
    mascara = df[columnas_busqueda].astype(str).apply(lambda col: col.str.lower().str.contains(busqueda, na=False)).any(axis=1)
    df = df[mascara]

if f_anio: df = df[df["anio"].isin(f_anio)]
if f_trim: df = df[df["trimestre"].isin(f_trim)]
if f_parc: df = df[df["parcela"].isin(f_parc)]

titulos = {clave: titulo for clave, _, _, titulo in PAGINAS}
with col_main:
    encabezado(st.session_state.pagina, titulos[st.session_state.pagina])
    if df.empty:
        st.warning("No hay datos con los filtros o la búsqueda actuales. Prueba a quitar algún filtro.")
    else:
        PAGINAS_FN[st.session_state.pagina](df)
    st.markdown("<div class='pie'>Fuente: PostgreSQL · db_detector_maiz · Instituto Tecnológico de Tizimín</div>", unsafe_allow_html=True)
