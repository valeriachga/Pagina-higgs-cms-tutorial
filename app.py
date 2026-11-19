"""
app.py: tutorial interactivo "Buscando el bosón de Higgs con datos abiertos de CMS"
(nivel preparatoria).

QUÉ HACE
--------
Es una página web hecha con Streamlit (solo Python). Explica los conceptos básicos
de la física de partículas y deja que el estudiante aplique los mismos cortes que
usa un científico de CMS (los de 02_cutflow.py) para "limpiar" los datos y ver
aparecer el pico del Higgs en la masa invariante de 4 leptones (m4l).

QUÉ ARCHIVOS LEE (dentro del mismo repositorio)
-----------------------------------------------
  web_data/muestra_eventos.csv.gz   -> generado con scripts/preparar_muestra.py
                                       (una fila por evento, unos pocos MB).
  resultados/grafica_final.png      -> tu gráfica final oficial (opcional).
Los CSV completos (GB) NO se usan aquí: viven en Google Drive.

CÓMO SE EJECUTA EN TU COMPUTADORA
---------------------------------
    pip install -r requirements.txt
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure   # sin pyplot: evita que se acumulen figuras en memoria

# =====================================================================================
# 0. CONFIGURACIÓN (lo único que normalmente tendrás que editar)
# =====================================================================================
REPO_URL = "https://github.com/valeriachga/higgs-4l-cms-opendata"
DRIVE_URL = "https://drive.google.com/drive/folders/1EeJIrFuOe8qOKbwBo9QCEP9_YRfOAyIt?usp=sharing"

RAIZ = Path(__file__).parent
RUTA_MUESTRA = RAIZ / "web_data" / "muestra_eventos.csv.gz"
RUTA_FINAL = RAIZ / "resultados" / "grafica_final.png"

BINS = np.linspace(70, 180, 37)               # 36 bins entre 70 y 180 GeV (igual que 03_grafica_final.py)
MUESTRAS_ZZ = ["ZZ→4μ", "ZZ→4e", "ZZ→2e2μ"]
HIGGS, DATOS = "Higgs", "Datos"

st.set_page_config(page_title="El Higgs en CMS Open Data", page_icon="⚛️", layout="wide")

# =====================================================================================
# 1. LOS CORTES (mismos valores y mismo orden que 02_cutflow.py)
# =====================================================================================
# Valores "oficiales" (selección del artículo de CMS arXiv:1207.7235)
OFICIAL = {
    "pt_e": 7.0, "pt_mu": 5.0,        # pT mínimo electrones / muones (GeV)
    "eta_e": 2.5, "eta_mu": 2.4,      # |eta| máximo electrones / muones
    "dxy": 0.5, "dz": 1.0,            # parámetros de impacto máximos (cm)
    "sip": 4.0,                       # SIP3D máximo
    "iso": 0.4,                       # aislamiento relativo máximo
    "pt1": 20.0, "pt2": 10.0,         # pT mínimo de los dos leptones más duros (GeV)
    "mz1_min": 40.0, "mz1_max": 120.0,   # ventana de masa de Z1 (GeV)
    "mz2_min": 12.0, "mz2_max": 120.0,   # ventana de masa de Z2 (GeV)
    "mll": 4.0,                       # masa mínima de cualquier par de carga opuesta (GeV)
    "m4l_min": 70.0,                  # masa mínima de los 4 leptones (GeV)
}
# Orden de las etapas (cada una incluye a las anteriores)
ORDEN = ["pt", "eta", "ip", "sip", "iso", "carga", "pt12", "pares", "mz1", "mz2", "mll", "m4l"]
NOMBRES = {
    "pt": "pT de los leptones", "eta": "|η| de los leptones", "ip": "|dxy| y |dz|",
    "sip": "SIP3D", "iso": "Aislamiento relativo", "carga": "Carga total = 0",
    "pt12": "pT1 y pT2 mínimos", "pares": "Existe emparejamiento Z+Z",
    "mz1": "Masa de Z1 en ventana", "mz2": "Masa de Z2 en ventana",
    "mll": "mll mínima (pares de carga opuesta)", "m4l": "m4l mínima",
}


def mascaras(d, p):
    """Para cada corte devuelve un arreglo True/False por evento (True = el evento pasa)."""
    return {
        "pt": (d["pt_e_min"] > p["pt_e"]) & (d["pt_mu_min"] > p["pt_mu"]),
        "eta": (d["eta_e_max"] < p["eta_e"]) & (d["eta_mu_max"] < p["eta_mu"]),
        "ip": (d["dxy_max"] < p["dxy"]) & (d["dz_max"] < p["dz"]),
        "sip": d["sip_max"] < p["sip"],
        "iso": d["iso_max"] < p["iso"],
        "carga": d["q_sum"] == 0,
        "pt12": (d["pt1"] > p["pt1"]) & (d["pt2"] > p["pt2"]),
        "pares": d["mz1"].notna(),
        "mz1": (d["mz1"] > p["mz1_min"]) & (d["mz1"] < p["mz1_max"]),
        "mz2": (d["mz2"] > p["mz2_min"]) & (d["mz2"] < p["mz2_max"]),
        "mll": d["mll_min"] > p["mll"],
        "m4l": d["m4l"] > p["m4l_min"],
    }


def mascara_total(d, activos, p):
    """True/False por evento: pasa TODOS los cortes activos (aplicados de forma acumulada)."""
    m = mascaras(d, p)
    ok = np.ones(len(d), dtype=bool)
    for clave in ORDEN:
        if clave in activos:
            ok &= m[clave].to_numpy()
    return ok


def aplicar(d, activos, p):
    """Devuelve los eventos que sobreviven a los cortes activos."""
    return d[mascara_total(d, activos, p)]


def parecido(d, activos, p):
    """
    Qué tanto se parece la selección del estudiante a la oficial (0 a 1).
    Es un índice de Jaccard ponderado: eventos que pasan AMBAS selecciones / eventos que pasan AL MENOS UNA.
    Los datos pesan 1 y las simulaciones pesan su peso físico.
    """
    w = np.where(d["muestra"] == DATOS, 1.0, d["peso"].to_numpy())
    a = mascara_total(d, activos, p)
    b = mascara_total(d, ORDEN, OFICIAL)
    union = w[a | b].sum()
    return float(w[a & b].sum() / union) if union > 0 else 1.0


def etiqueta(clave, p):
    """Texto de cada fila de la tabla cutflow, con los valores elegidos."""
    return {
        "pt": f"pT: e > {p['pt_e']:g}, μ > {p['pt_mu']:g} GeV",
        "eta": f"|η|: e < {p['eta_e']:g}, μ < {p['eta_mu']:g}",
        "ip": f"|dxy| < {p['dxy']:g} cm y |dz| < {p['dz']:g} cm",
        "sip": f"SIP3D < {p['sip']:g}",
        "iso": f"Aislamiento < {p['iso']:g}",
        "carga": "Carga total = 0",
        "pt12": f"pT1 > {p['pt1']:g} y pT2 > {p['pt2']:g} GeV",
        "pares": "Existe emparejamiento Z+Z",
        "mz1": f"mZ1 entre {p['mz1_min']:g} y {p['mz1_max']:g} GeV",
        "mz2": f"mZ2 entre {p['mz2_min']:g} y {p['mz2_max']:g} GeV",
        "mll": f"mll mínima > {p['mll']:g} GeV",
        "m4l": f"m4l > {p['m4l_min']:g} GeV",
    }[clave]


# =====================================================================================
# 2. DATOS, TABLAS Y GRÁFICAS
# =====================================================================================
@st.cache_data
def cargar_muestra():
    """Lee la muestra reducida (una fila por evento). Devuelve None si el archivo no existe."""
    if not RUTA_MUESTRA.exists():
        return None
    return pd.read_csv(RUTA_MUESTRA)


def tabla_cutflow(d, activos, p):
    """Eventos (ponderados en las simulaciones) que sobreviven tras cada corte activo."""
    columnas = [HIGGS] + MUESTRAS_ZZ + [DATOS]

    def fila(sub):
        suma = sub.groupby("muestra")["peso"].sum()
        return [float(suma.get(c, 0.0)) for c in columnas]

    m = mascaras(d, p)
    ok = np.ones(len(d), dtype=bool)
    filas = {"Muestra de partida (m4l entre 70 y 180 GeV)": fila(d)}
    for clave in ORDEN:
        if clave in activos:
            ok &= m[clave].to_numpy()
            filas[etiqueta(clave, p)] = fila(d[ok])
    tabla = pd.DataFrame(filas, index=columnas).T
    tabla.index.name = "Corte"
    return tabla


def histogramas(d):
    """Histogramas de m4l: fondo ZZ (ponderado), Higgs (ponderado) y datos reales (conteo)."""
    zz = d[d["muestra"].isin(MUESTRAS_ZZ)]
    hg = d[d["muestra"] == HIGGS]
    dt = d[d["muestra"] == DATOS]
    h_zz, _ = np.histogram(zz["m4l"], bins=BINS, weights=zz["peso"])
    h_h, _ = np.histogram(hg["m4l"], bins=BINS, weights=hg["peso"])
    h_dat, _ = np.histogram(dt["m4l"], bins=BINS)
    return h_zz, h_h, h_dat


def figura(d, titulo):
    """Gráfica al estilo CMS Open Data: ZZ azul, ZZ+Higgs línea roja y datos como puntos con error de Poisson."""
    h_zz, h_h, h_dat = histogramas(d)
    centros = 0.5 * (BINS[1:] + BINS[:-1])
    fig = Figure(figsize=(6.5, 4.6))
    ax = fig.subplots()
    ax.stairs(h_zz + h_h, BINS, color="red", linewidth=2, label="ZZ + Higgs (m$_H$ = 125 GeV)")
    ax.stairs(h_zz, BINS, fill=True, facecolor="#9ecae1", edgecolor="black", linewidth=1.2,
              label="ZZ (simulación)")
    con = h_dat > 0
    ax.errorbar(centros[con], h_dat[con], yerr=np.sqrt(h_dat[con]), fmt="o", color="black", markersize=4,
                label=f"Datos CMS 2012 ({int(h_dat.sum())} eventos)")
    tope = max((h_zz + h_h).max(), (h_dat + np.sqrt(h_dat)).max(), 1e-9)
    ax.set_ylim(0, tope * 1.3)                      # cada gráfica ajusta su eje Y
    ax.set_xlim(BINS[0], BINS[-1])
    ax.set_xlabel("Mass 4 leptons / GeV")
    ax.set_ylabel("N_Events")
    ax.set_title(titulo, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def aviso_sin_muestra():
    st.error("No encuentro `web_data/muestra_eventos.csv.gz`. Genera el archivo con "
             "`python scripts/preparar_muestra.py` y súbelo al repositorio.")


# =====================================================================================
# 3. SECCIONES DE LA PÁGINA
# =====================================================================================
# ---------- datos de las simulaciones (los mismos de 02_cutflow.py) ----------
LUMI_PB = 11580.0
SIMULACIONES = {  # nombre en la página: (proceso, sección eficaz σ [pb], N de eventos generados, factor k)
    "Higgs": ("H → ZZ → 4ℓ", 0.0065, 299973, 1.0),
    "ZZ→4μ": ("ZZ → 4μ", 0.077, 1499064, 1.386),
    "ZZ→4e": ("ZZ → 4e", 0.077, 1499093, 1.386),
    "ZZ→2e2μ": ("ZZ → 2e2μ", 0.18, 1497445, 1.386),
}
IMG = RAIZ / "imagenes"


def imagen(nombre, pie=None, ancho=720):
    """Muestra una ilustración de la carpeta imagenes/. Si no la encuentra, avisa dónde la buscó."""
    ruta = IMG / nombre
    if ruta.exists():
        st.image(str(ruta), caption=pie, width=ancho)
    else:
        st.caption(f"⚠️ Falta la imagen `{nombre}`. La página la busca en: `{IMG}`")


def tabla_simulaciones():
    """Tabla con σ, k, N generados, eventos esperados y peso por evento de cada simulación."""
    filas = []
    for nombre, (proceso, xsec, n_gen, k) in SIMULACIONES.items():
        esperado = xsec * LUMI_PB * k
        filas.append({"Muestra": nombre, "Proceso": proceso, "σ (pb)": f"{xsec:g}", "k": f"{k:g}",
                      "N generados": f"{n_gen:,}", "Eventos esperados (σ·L·k)": f"{esperado:,.1f}",
                      "Peso por evento": f"{esperado / n_gen:.6f}"})
    return pd.DataFrame(filas).set_index("Muestra")


def demo_pi():
    """Mini demostración del método Monte Carlo: estimar π con puntos aleatorios."""
    n = st.slider("Número de puntos aleatorios", 100, 20000, 2000, step=100, key="pi_n")
    rng = np.random.default_rng(7)
    x, y = rng.random(n), rng.random(n)
    dentro = x**2 + y**2 <= 1
    fig = Figure(figsize=(4.2, 4.2))
    ax = fig.subplots()
    ax.scatter(x[~dentro], y[~dentro], s=4, color="#BBBBB5")
    ax.scatter(x[dentro], y[dentro], s=4, color="#3B6FB6")
    t = np.linspace(0, np.pi / 2, 100)
    ax.plot(np.cos(t), np.sin(t), color="#C8553D", lw=1.5)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    fig.tight_layout()
    c1, c2 = st.columns([1, 1.2])
    c1.pyplot(fig)
    c2.metric("π estimado", f"{4 * dentro.mean():.4f}", delta=f"{4 * dentro.mean() - np.pi:+.4f} vs π real", delta_color="off")
    c2.write("Se lanzan puntos al azar en un cuadrado de lado 1. La fracción que cae dentro del cuarto de "
             "círculo es π/4. Con más puntos, la estimación mejora: así funcionan las simulaciones Monte Carlo, "
             "con muchísimos números aleatorios.")


def ir_a(seccion):
    """Botón de acceso rápido: cambia la sección que muestra el menú lateral."""
    st.session_state["seccion"] = seccion


def seccion_intro():
    st.title("Buscando el bosón de Higgs con datos abiertos de CMS")
    imagen("portada.png", ancho=760)
    st.write(
        "Bienvenido. En esta página vas a hacer lo mismo que hace un científico del experimento CMS: "
        "tomar millones de colisiones de partículas y **filtrar** los datos hasta encontrar una señal "
        "muy pequeña, la del **bosón de Higgs**. No necesitas conocimientos previos de física, solo curiosidad."
    )
    st.info("Los datos son reales: colisiones protón-protón a 8 TeV registradas por CMS en 2012 y "
            "liberadas públicamente en el portal CERN Open Data.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("¿Qué es el CERN?")
        st.write(
            "El CERN es la Organización Europea para la Investigación Nuclear, cerca de Ginebra, fundada en "
            "1954. Ahí está el **LHC**, el acelerador de partículas más grande del mundo, donde se hacen chocar "
            "protones para estudiar de qué está hecha la materia."
        )
    with c2:
        st.subheader("¿Qué es CMS?")
        st.write(
            "CMS (*Compact Muon Solenoid*) es uno de los grandes detectores del LHC. Es operado por una "
            "colaboración de miles de científicos de decenas de países. En 2012, CMS y ATLAS anunciaron el "
            "descubrimiento del bosón de Higgs."
        )
    st.subheader("Al terminar esta página podrás…")
    st.markdown(
        "- Explicar qué son las partículas, el Modelo Estándar y el bosón de Higgs.\n"
        "- Describir cómo un detector como CMS identifica electrones y muones.\n"
        "- Explicar qué es una simulación Monte Carlo y para qué sirve.\n"
        "- Aplicar cortes a datos reales y ver cómo aparece (o no) una señal."
    )
    st.header("Ir directo a una sección")
    destinos = [
        ("🧩 Fundamentos", "Fundamentos de la física de partículas"),
        ("🔬 LHC y CMS", "El LHC y el detector CMS"),
        ("📐 Cómo se analiza un dato", "Cómo se analiza un dato"),
        ("🎲 Monte Carlo", "Simulaciones Monte Carlo"),
        ("🎯 Analiza los datos", "Analiza los datos"),
        ("✅ Conclusiones y quiz", "Conclusiones y autoevaluación"),
        ("⬇️ Descargas", "Descarga de datos y créditos"),
    ]
    for fila in (destinos[:4], destinos[4:]):
        cols = st.columns(4)
        for col, (texto, seccion) in zip(cols, fila):
            col.button(texto, key=f"ir_{seccion}", use_container_width=True, on_click=ir_a, args=(seccion,))
    st.header("Hoja de ruta")
    st.table(pd.DataFrame({
        "Sección": ["Fundamentos de la física de partículas", "El LHC y el detector CMS",
                    "Cómo se analiza un dato", "Simulaciones Monte Carlo", "Analiza los datos",
                    "Conclusiones y autoevaluación", "Descarga de datos y créditos"],
        "¿Qué vas a hacer?": [
            "Conocer las partículas, el Modelo Estándar, los bosones Z y W y el Higgs.",
            "Ver cómo se producen las colisiones y cómo se detectan electrones y muones.",
            "Entender la masa invariante, el canal H → ZZ → 4 leptones, la estadística y los filtros de selección.",
            "Descubrir qué son las simulaciones, la señal, el fondo y los pesos.",
            "Resolver un reto: elegir cortes paso a paso hasta que tu gráfica se parezca a la oficial.",
            "Repasar lo aprendido, el glosario y contestar un mini examen.",
            "Bajar los datos y el código para repetirlo en tu computadora.",
        ],
    }).set_index("Sección"))


def seccion_fundamentos():
    st.title("Fundamentos de la física de partículas")
    t1, t2, t3, t4, t5 = st.tabs(["¿Qué son las partículas?", "Leptones y cargas", "El Modelo Estándar",
                                  "Los bosones Z y W", "El bosón de Higgs"])
    with t1:
        st.header("¿Qué son las partículas?")
        imagen("atomo.png", "De la materia cotidiana a los quarks (tamaños aproximados).")
        st.write(
            "Todo lo que ves está hecho de átomos. Un átomo tiene un núcleo (protones y neutrones) y "
            "electrones a su alrededor. Pero los protones y neutrones tampoco son lo más pequeño: están "
            "hechos de **quarks**, unidos por **gluones**. A las piezas que, hasta donde sabemos, ya no se "
            "pueden dividir las llamamos **partículas elementales**."
        )
        st.write(
            "Un protón, por ejemplo, es una partícula **compuesta**: dos quarks *up* y uno *down*, además de "
            "gluones. El electrón, en cambio, es elemental."
        )
        st.subheader("Los aceleradores son microscopios")
        st.write(
            "Para ver estructuras más pequeñas se necesita más energía. Por eso se construyen aceleradores: "
            "al hacer chocar partículas con energías enormes, se puede *explorar* lo muy pequeño y *producir* "
            "partículas nuevas, que existen solo un instante antes de desintegrarse."
        )
        st.caption("GeV = gigaelectronvoltio, unidad de energía y de masa en física de partículas. "
                   "Un protón pesa casi 1 GeV (y 1 TeV = 1,000 GeV).")
    with t2:
        st.header("Leptones: electrones, muones y compañía")
        st.write(
            "Los **leptones** son partículas elementales que no sienten la fuerza fuerte. Hay tres 'sabores' "
            "con carga eléctrica (electrón, muón, tau) y tres neutrinos, sin carga."
        )
        st.table(pd.DataFrame({
            "Carga": ["−1", "−1", "−1", "0"],
            "Masa": ["0.511 MeV", "105.7 MeV", "1.777 GeV", "casi cero"],
            "¿Qué pasa con ella?": [
                "Es estable: se detecta fácilmente en el rastreador y el calorímetro.",
                "Dura unos 2.2 microsegundos, pero a velocidades cercanas a la de la luz le alcanza para "
                "atravesar todo el detector.",
                "Se desintegra casi al instante (≈ 3×10⁻¹³ s): no la vemos directamente.",
                "Casi no interactúa con la materia: escapa del detector sin dejar señal.",
            ],
        }, index=["Electrón (e)", "Muón (μ)", "Tau (τ)", "Neutrinos (ν)"]))
        st.write(
            "Cada partícula tiene una **antipartícula** con la carga contraria: el positrón (+1) para el "
            "electrón y el antimuón (+1) para el muón. En el detector distinguimos ambos por hacia dónde "
            "se curvan en el campo magnético."
        )
        st.info("La carga eléctrica se conserva. Un Higgs es neutro, así que si se desintegra en 4 leptones, "
                "la suma de sus cargas debe ser cero: dos positivos y dos negativos.")
    with t3:
        st.header("El Modelo Estándar")
        imagen("modelo_estandar.png", "Las 17 partículas elementales del Modelo Estándar, con su masa y su carga.")
        st.write("Es la teoría que ordena todas las partículas elementales conocidas en dos grupos:")
        st.markdown(
            "- **Fermiones** (los ladrillos de la materia): 6 quarks y 6 leptones, en tres 'generaciones' "
            "cada vez más pesadas. La materia ordinaria solo usa la primera (u, d, e).\n"
            "- **Bosones** (transmiten las fuerzas): el fotón (electromagnetismo), los bosones W y Z (fuerza "
            "débil) y los gluones (fuerza fuerte).\n"
            "- El **bosón de Higgs**, una partícula especial ligada a cómo las demás obtienen su masa."
        )
        with st.expander("Lo que el Modelo Estándar todavía no explica"):
            st.markdown(
                "- **La gravedad**: no está incluida en el modelo.\n"
                "- **La materia oscura y la energía oscura**: la materia ordinaria es solo ≈ 5 % del universo; "
                "≈ 27 % sería materia oscura y ≈ 68 % energía oscura.\n"
                "- **Las masas de los neutrinos** y por qué hay más materia que antimateria."
            )
    with t4:
        st.header("Los bosones Z y W")
        st.write(
            "Los bosones **W** (carga ±1, masa de unos 80 GeV) y **Z** (neutro, unos **91 GeV**) transmiten la "
            "fuerza débil, responsable, por ejemplo, de ciertas desintegraciones radiactivas. Se descubrieron "
            "en el CERN en 1983."
        )
        st.write(
            "Ambos viven un instante muy corto y se desintegran enseguida. El Z se puede desintegrar en un par "
            "de leptones de carga opuesta: alrededor de 3 de cada 100 veces en un electrón y un positrón, y otras "
            "3 de cada 100 en un muón y un antimuón. Esos pares son los que buscamos."
        )
        st.write("Esto es importante porque el Higgs que buscamos se desintegra en *dos* bosones Z.")
        with st.expander("¿Qué es una partícula 'virtual'?"):
            st.write(
                "Una partícula virtual es una que existe solo un instante y no tiene exactamente su masa "
                "'normal'. Como el Higgs (125 GeV) pesa menos que dos Z juntos (≈ 182 GeV), una de las dos Z "
                "tiene que ser virtual, con una masa menor. La llamamos Z*."
            )
    with t5:
        st.header("El bosón de Higgs")
        st.write(
            "Según el Modelo Estándar, las partículas elementales adquieren masa al interactuar con el "
            "**campo de Higgs**, que llena todo el espacio. El bosón de Higgs es una pequeña vibración de ese campo, "
            "igual que una ola es una vibración del mar."
        )
        imagen("campo_higgs.png", "Esquema: el campo de Higgs tiene un valor distinto de cero en todo el espacio.")
        st.subheader("Un poco de historia")
        st.markdown(
            "- **1964:** varios físicos, entre ellos Peter Higgs, François Englert y Robert Brout, proponen el mecanismo.\n"
            "- **2012 (4 de julio):** ATLAS y CMS anuncian el descubrimiento de una nueva partícula de unos **125 GeV**.\n"
            "- **2013:** Peter Higgs y François Englert reciben el Premio Nobel de Física."
        )
        st.subheader("¿Qué lo hace especial?")
        st.markdown(
            "- Es la única partícula elemental conocida **sin espín** (espín 0).\n"
            "- Da masa a los bosones W y Z y a los quarks y leptones cargados.\n"
            "- **No** explica toda la masa de la materia: la mayor parte de la masa de un protón viene de la "
            "energía de la fuerza fuerte entre sus quarks, no del Higgs."
        )
        st.subheader("¿Cómo se desintegra?")
        st.write(
            "El Higgs vive una fracción minúscula de segundo (≈ 10⁻²² s) y no se ve directamente: lo "
            "identificamos por las partículas en las que se desintegra."
        )
        imagen("higgs_desintegraciones.png", "Cómo se desintegra un Higgs de 125 GeV (porcentajes aproximados).")
        st.write(
            "El modo más frecuente (b b̄) está sepultado en ruido de fondo, mientras que el canal de 4 leptones "
            "es muy raro pero muy limpio. Por eso lo usamos."
        )


def seccion_lhc():
    st.title("El LHC y el detector CMS")
    t1, t2, t3, t4 = st.tabs(["Colisiones y luminosidad", "Detectando electrones y muones",
                              "Variables que mide CMS", "Del detector al CSV"])
    with t1:
        st.header("Colisiones protón-protón")
        imagen("lhc.png", "El LHC y sus cuatro grandes experimentos.")
        st.write(
            "El **LHC** (Gran Colisionador de Hadrones) es un anillo de unos 27 km de circunferencia, "
            "enterrado cerca de Ginebra. Acelera protones casi a la velocidad de la luz en dos haces que giran "
            "en sentidos opuestos. En 2012 cada haz tenía 4 TeV, así que cada choque tenía 8 TeV de energía. "
            "Cada choque se llama **evento**."
        )
        st.write(
            "Los protones viajan en paquetes que se cruzan decenas de millones de veces por segundo, y en "
            "cada cruce chocan, en promedio, alrededor de 20 pares de protones a la vez. Es imposible guardar "
            "todo: un sistema de selección llamado *trigger* decide en una fracción de segundo qué eventos "
            "valen la pena y guarda solo unos cientos por segundo."
        )
        st.subheader("Luminosidad y número de eventos")
        st.write(
            "La **luminosidad integrada** (L) mide cuántas colisiones se acumularon. Los datos que usamos "
            "corresponden a **11.58 fb⁻¹** (= 11,580 pb⁻¹; 1 fb⁻¹ = 1,000 pb⁻¹). Si un proceso tiene sección "
            "eficaz σ (qué tan probable es), el número esperado de eventos es:"
        )
        st.latex(r"N = \sigma \times L")
        st.info("Ejemplo: la simulación de Higgs → ZZ → 4ℓ que usamos tiene σ = 0.0065 pb. Entonces se esperan "
                "0.0065 × 11,580 ≈ 75 eventos de este tipo producidos en todo 2012, antes de contar cuántos "
                "alcanza a ver el detector.")
    with t2:
        st.header("¿Cómo detecta CMS a los leptones?")
        imagen("cms_detector.png", "Corte transversal esquemático de CMS: cada partícula deja una huella distinta.")
        st.write(
            "CMS mide unos 21 m de largo y 15 m de diámetro, y pesa unas 14,000 toneladas. Está formado por "
            "capas, como una cebolla, y cada una mide algo distinto:"
        )
        st.markdown(
            "- **Rastreador de silicio:** sigue la trayectoria de las partículas cargadas. De ahí salen "
            "variables como pT, η, φ, dxy y dz.\n"
            "- **Calorímetro electromagnético:** frena a los electrones y mide su energía.\n"
            "- **Calorímetro hadrónico:** frena a los hadrones (protones, piones…).\n"
            "- **Imán superconductor (3.8 teslas):** curva las trayectorias. El sentido de la curva dice la carga.\n"
            "- **Cámaras de muones:** los muones son los únicos que atraviesan casi todo el detector."
        )
        with st.expander("¿Cómo se mide el pT con la curvatura?"):
            st.write("Una partícula cargada en un campo magnético B describe un círculo. Mientras más energía "
                     "tiene, más abierto es el círculo:")
            st.latex(r"p_T\,[\text{GeV}] \approx 0.3 \times B\,[\text{T}] \times R\,[\text{m}]")
            st.write("Con B = 3.8 T, un leptón de 20 GeV describe un círculo de unos 17.5 m de radio.")
    with t3:
        st.header("Las variables que usaremos")
        st.write("Estas son las medidas que usan los cortes del tutorial. Cada una se explica con un esquema.")
        st.subheader("η y pT: ¿hacia dónde va y con cuánta fuerza?")
        imagen("eta_pt.png")
        st.write(
            "**η (eta)** indica el ángulo respecto al haz: η = 0 es perpendicular y valores grandes son casi "
            "paralelos al haz. El detector solo mide bien hasta |η| ≈ 2.5. **pT** es la parte del momento "
            "perpendicular al haz, la que más nos interesa."
        )
        st.subheader("dxy y dz: ¿nació en el punto de colisión?")
        imagen("impacto.png")
        st.write(
            "Un leptón que viene de un Higgs nace justo en el punto de colisión, así que su trayectoria pasa "
            "casi por ahí. **dxy** y **dz** miden qué tan lejos pasa, en el plano transversal y a lo largo del haz. "
            "**SIP3D** combina ambas distancias y las compara con su incertidumbre."
        )
        st.subheader("Aislamiento: ¿viene solo o acompañado?")
        imagen("aislamiento.png")
        st.write(
            "Los leptones de un Higgs suelen estar solos. Los falsos (por ejemplo, de la desintegración de "
            "quarks pesados) vienen rodeados de otras partículas. El aislamiento relativo mide esa actividad "
            "alrededor del leptón."
        )
    with t4:
        st.header("Del detector al archivo CSV")
        imagen("evento_csv.png", "Cada evento con 4 leptones se convierte en una fila de la tabla.")
        st.markdown(
            "1. El detector registra la colisión y el *trigger* decide si la guarda.\n"
            "2. Se **reconstruyen** las partículas a partir de las señales de cada capa.\n"
            "3. CERN Open Data publica los eventos en archivos con formato ROOT.\n"
            "4. Nosotros los convertimos a **CSV**: una fila por evento y una columna por cada medición "
            "(`pt1`, `eta1`, `Q1`…).\n"
            "5. Los CSV completos pesan varios gigabytes, por eso en esta página trabajamos con una versión "
            "reducida."
        )


OPCIONES_BOSON = {
    "jpsi": "Partícula J/ψ (≈ 3.1 GeV)",
    "z": "Bosón Z (≈ 91 GeV)",
    "higgs": "Bosón de Higgs (≈ 125 GeV)",
    "zz": "Un par de bosones Z (más de 182 GeV)",
}

EXPLICACION_RESPUESTA = {
    "jpsi": "Una masa cercana a 3.1 GeV corresponde a la partícula J/ψ, formada por un quark charm y su antiquark. "
            "Se desintegra en un par de leptones de carga opuesta.",
    "z": "Una masa cercana a 91 GeV corresponde al bosón Z, que se desintegra en un par de leptones de carga opuesta.",
    "higgs": "Cuatro leptones con masa invariante cercana a 125 GeV son la firma del Higgs: H → Z + Z* → 4 leptones "
             "(una de las dos Z es virtual).",
    "zz": "Cuatro leptones con masa mayor a 182 GeV (el doble de la masa de una Z) pueden venir de dos bosones Z "
          "reales, un proceso de fondo del análisis: ZZ → 4 leptones.",
}


def _boost(p, beta):
    """Aplica un 'boost' de Lorentz con velocidad beta (vector 3D, en unidades de c) a cuadrivectores p (n, 4)."""
    b2 = float(beta @ beta)
    if b2 == 0:
        return p
    gamma = 1.0 / np.sqrt(1.0 - b2)
    bp = p[:, 1:] @ beta
    e = gamma * (p[:, 0] + bp)
    pvec = p[:, 1:] + (((gamma - 1.0) / b2) * bp + gamma * p[:, 0])[:, None] * beta[None, :]
    return np.column_stack([e, pvec])


def _desintegrar(P, m1, m2, rng):
    """Desintegra una partícula con cuadrivector P en dos de masas m1 y m2 (dirección al azar en su reposo)."""
    M = np.sqrt(P[0] ** 2 - P[1:] @ P[1:])
    ps = np.sqrt((M**2 - (m1 + m2) ** 2) * (M**2 - (m1 - m2) ** 2)) / (2 * M)
    c, fi = rng.uniform(-1, 1), rng.uniform(0, 2 * np.pi)
    s = np.sqrt(1 - c**2)
    d = np.array([s * np.cos(fi), s * np.sin(fi), c])
    hijas = np.array([[np.sqrt(ps**2 + m1**2), *(ps * d)], [np.sqrt(ps**2 + m2**2), *(-ps * d)]])
    return _boost(hijas, P[1:] / P[0])


def _particula(M, rng, pt_escala):
    """Cuadrivector de una partícula de masa M con pT, rapidez y ángulo φ al azar."""
    pt = rng.exponential(pt_escala) + 3.0
    y, fi = rng.normal(0, 0.8), rng.uniform(-np.pi, np.pi)
    mt = np.sqrt(M**2 + pt**2)
    return np.array([mt * np.cosh(y), pt * np.cos(fi), pt * np.sin(fi), mt * np.sinh(y)])


def generar_ejercicio(nivel, semilla):
    """
    Crea un 'evento' con 2 o 4 leptones (sin masa) a partir de una partícula conocida.
    Devuelve un diccionario con la respuesta correcta y la tabla de leptones (pT, η, φ ya redondeados),
    de modo que la masa correcta se calcula con los mismos números que ve el estudiante.
    """
    rng = np.random.default_rng(semilla * 7919 + nivel)
    if nivel == 2:
        caso = str(rng.choice(["z", "jpsi"]))
        M = rng.normal(91.19, 2.0) if caso == "z" else rng.normal(3.097, 0.03)
        padre = _particula(M, rng, 18.0 if caso == "z" else 14.0)
        pares = [_desintegrar(padre, 0.0, 0.0, rng)]
        sabores = [str(rng.choice(["e", "μ"])) if caso == "z" else "μ"]
    else:
        caso = str(rng.choice(["higgs", "zz"]))
        if caso == "higgs":
            M = rng.normal(125.0, 1.0)
            m_z1 = rng.normal(91.19, 2.0)
            m_z2 = rng.uniform(12.0, min(30.0, M - m_z1 - 1.0))
        else:
            M = rng.uniform(190.0, 260.0)
            m_z1, m_z2 = rng.normal(91.19, 2.0), rng.normal(91.19, 2.0)
        padre = _particula(M, rng, 25.0)
        z1, z2 = _desintegrar(padre, m_z1, m_z2, rng)
        pares = [_desintegrar(z1, 0.0, 0.0, rng), _desintegrar(z2, 0.0, 0.0, rng)]
        tipo = str(rng.choice(["4μ", "4e", "2e2μ"]))
        sabores = {"4μ": ["μ", "μ"], "4e": ["e", "e"], "2e2μ": ["e", "μ"]}[tipo]

    leptones = []
    for par, sabor in zip(pares, sabores):
        carga_primero = int(rng.choice([-1, 1]))
        for k, v in enumerate(par):
            px, py, pz = v[1], v[2], v[3]
            pt = float(np.hypot(px, py))
            leptones.append({"tipo": sabor, "carga": carga_primero if k == 0 else -carga_primero,
                             "pt": round(pt, 2), "eta": round(float(np.arcsinh(pz / pt)), 3),
                             "phi": round(float(np.arctan2(py, px)), 3)})
    leptones.sort(key=lambda l: -l["pt"])
    return {"caso": caso, "leptones": leptones, "masa": masa_de_leptones(leptones)}


def componentes(leptones):
    """Componentes (px, py, pz, E) de cada leptón, tratándolos sin masa: E = |p|."""
    pt = np.array([l["pt"] for l in leptones]); eta = np.array([l["eta"] for l in leptones])
    phi = np.array([l["phi"] for l in leptones])
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return px, py, pz, np.sqrt(px**2 + py**2 + pz**2)


def masa_de_leptones(leptones):
    """Masa invariante de un grupo de leptones sin masa (suma de cuadrivectores)."""
    px, py, pz, e = componentes(leptones)
    return float(np.sqrt(max(e.sum() ** 2 - px.sum() ** 2 - py.sum() ** 2 - pz.sum() ** 2, 0.0)))


def nuevo_ejercicio():
    """Botón: pasa al siguiente ejercicio (cambia la semilla)."""
    st.session_state["ej_semilla"] = st.session_state.get("ej_semilla", 1) + 1


def ejercicio_masa_invariante():
    """Ejercicio interactivo: calcular la masa invariante de unos leptones y decir de qué partícula vienen."""
    st.subheader("Ejercicio: ¿de qué partícula vienen estos leptones?")
    st.write(
        "Un detector midió los leptones de la tabla (los tratamos sin masa, porque es muy pequeña comparada "
        "con su energía). **Calcula su masa invariante** y decide qué partícula los produjo."
    )
    st.session_state.setdefault("ej_semilla", 1)
    c1, c2 = st.columns([2, 1])
    nivel_txt = c1.radio("Nivel", ["Dos leptones (más fácil)", "Cuatro leptones (reto)"], horizontal=True,
                         key="ej_nivel")
    c2.button("🎲 Nuevo ejercicio", on_click=nuevo_ejercicio)
    nivel = 2 if nivel_txt.startswith("Dos") else 4
    semilla = st.session_state["ej_semilla"]
    ej = generar_ejercicio(nivel, semilla)
    leptones = ej["leptones"]
    id_ej = f"{nivel}_{semilla}"      # las respuestas se reinician con cada ejercicio nuevo

    st.table(pd.DataFrame({
        "Leptón": range(1, len(leptones) + 1),
        "Tipo": [l["tipo"] for l in leptones],
        "Carga": [f"{l['carga']:+d}".replace("-", "−") for l in leptones],
        "pT (GeV)": [f"{l['pt']:.2f}" for l in leptones],
        "η": [f"{l['eta']:.3f}" for l in leptones],
        "φ (rad)": [f"{l['phi']:.3f}" for l in leptones],
    }).set_index("Leptón"))

    with st.expander("📐 ¿Qué son pT, η y φ? (esquema)"):
        imagen("coordenadas.png", "Coordenadas del detector: el eje z es el haz de protones.", ancho=760)
        st.write(
            "**pT** es la parte del momento perpendicular al haz, **φ** es el ángulo alrededor del haz y "
            "**η** (pseudorapidez) indica qué tan cerca del haz va la partícula (η = −ln tan(θ/2), con θ el "
            "ángulo respecto al eje z)."
        )
    with st.expander("💡 Pista: cómo calcular la masa invariante"):
        st.write("Para cada leptón calcula sus componentes (como no tiene masa, su energía es igual a su momento):")
        st.latex(r"p_x = p_T\cos\varphi,\quad p_y = p_T\sin\varphi,\quad p_z = p_T\sinh\eta,\quad E=\sqrt{p_x^2+p_y^2+p_z^2}")
        st.write("Suma las componentes de todos los leptones y calcula:")
        st.latex(r"m=\sqrt{\left(\sum E\right)^2-\left(\sum p_x\right)^2-\left(\sum p_y\right)^2-\left(\sum p_z\right)^2}")
        st.write("**Fórmula con los ángulos** (equivalente, sin calcular componentes). Para un par de leptones sin masa:")
        st.latex(r"m^2 = 2\,p_{T1}\,p_{T2}\left[\cosh(\eta_1-\eta_2)-\cos(\varphi_1-\varphi_2)\right]")
        if nivel == 2:
            st.write("Solo debes calcular Δη = η₁ − η₂ y Δφ = φ₁ − φ₂, y sustituir.")
        else:
            st.write("Para cuatro leptones, suma esa expresión sobre **todos los pares** posibles (hay 6):")
            st.latex(r"m^2 = \sum_{i<j} 2\,p_{Ti}\,p_{Tj}\left[\cosh(\eta_i-\eta_j)-\cos(\varphi_i-\varphi_j)\right]")

            st.write("Con cuatro leptones conviene usar una hoja de cálculo o Python. Por ejemplo:")
            st.code(
                "import numpy as np\n"
                f"pt  = np.array({[l['pt'] for l in leptones]})\n"
                f"eta = np.array({[l['eta'] for l in leptones]})\n"
                f"phi = np.array({[l['phi'] for l in leptones]})\n"
                "px, py, pz = pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)\n"
                "E = np.sqrt(px**2 + py**2 + pz**2)\n"
                "m = np.sqrt(E.sum()**2 - px.sum()**2 - py.sum()**2 - pz.sum()**2)\n"
                "print(m)",
                language="python")

    masa = st.number_input("Tu masa invariante (GeV)", min_value=0.0, value=0.0, step=0.1, format="%.2f",
                           key=f"ej_masa_{id_ej}")
    opciones = list(OPCIONES_BOSON.values())
    resp = st.radio("¿A qué partícula corresponde?", opciones, index=None, key=f"ej_resp_{id_ej}")

    if st.button("Revisar mi respuesta", key=f"ej_rev_{id_ej}"):
        if masa <= 0 or resp is None:
            st.warning("Escribe tu masa invariante y elige una partícula.")
        else:
            if abs(masa - ej["masa"]) <= 0.03 * ej["masa"]:
                st.success(f"Tu cálculo es correcto: m ≈ {ej['masa']:.2f} GeV.")
            else:
                st.error("Tu masa no coincide con el resultado. Revisa las fórmulas de la pista y vuelve a intentarlo.")
            if resp == OPCIONES_BOSON[ej["caso"]]:
                st.success("¡Identificaste la partícula! " + EXPLICACION_RESPUESTA[ej["caso"]])
            else:
                st.error("Esa no es la partícula. Compara tu masa con las masas de la lista de opciones.")

    with st.expander("✅ Ver la solución paso a paso"):
        px, py, pz, e = componentes(leptones)
        st.dataframe(pd.DataFrame({"px": px, "py": py, "pz": pz, "E": e},
                                  index=[f"Leptón {i + 1}" for i in range(len(leptones))]).round(2))
        st.write(
            f"Sumas: ΣE = {e.sum():.2f}, Σpx = {px.sum():.2f}, Σpy = {py.sum():.2f}, Σpz = {pz.sum():.2f} (en GeV)."
        )
        st.write(f"**Masa invariante: m ≈ {ej['masa']:.2f} GeV**")
        st.write(f"**Respuesta: {OPCIONES_BOSON[ej['caso']]}.** " + EXPLICACION_RESPUESTA[ej["caso"]])


def seccion_analisis_teoria():
    st.title("Cómo se analiza un dato")
    t0, t1, t2, t3, t4 = st.tabs(["Visión general", "Masa invariante", "El canal H → ZZ → 4ℓ",
                                  "Estadística básica", "Selección de eventos"])
    with t0:
        st.header("La receta de un análisis")
        st.markdown(
            "1. **Elegir qué buscar:** un proceso concreto, aquí H → ZZ → 4 leptones.\n"
            "2. **Seleccionar eventos:** aplicar cortes para quedarnos con los que parecen la señal.\n"
            "3. **Predecir el fondo:** con simulaciones Monte Carlo, qué esperamos *sin* Higgs.\n"
            "4. **Comparar con los datos:** poner datos y predicción en la misma gráfica.\n"
            "5. **Medir la incertidumbre:** ¿el exceso es mayor que una fluctuación al azar?\n"
            "6. **Concluir:** ¿hay evidencia de algo nuevo?"
        )
    with t1:
        st.header("La masa invariante")
        imagen("masa_invariante.png")
        st.write(
            "Si una partícula pesada se desintegra, no podemos verla, pero sí a sus hijas. Con la energía "
            "y el momento de las hijas se reconstruye la masa de la madre:"
        )
        st.latex(r"m^2 = E^2 - p_x^2 - p_y^2 - p_z^2 \quad (E, \vec p \text{ sumados de todas las hijas})")
        st.write(
            "Ejemplo: si un electrón y un positrón vienen de un Z, su masa invariante sale cerca de 91 GeV. "
            "Si vienen de un Higgs, los 4 leptones dan un valor cercano a 125 GeV. Si vienen de otros procesos, "
            "la masa se reparte en muchos valores. Cada evento aporta *un* número, y al juntar muchos "
            "eventos buscamos un **pico**."
        )
        st.divider()
        ejercicio_masa_invariante()
    with t2:
        st.header("¿Por qué buscar el Higgs en 4 leptones?")
        imagen("higgs_decay.png", "Esquema de la desintegración H → ZZ* → 4 leptones.")
        st.write(
            "Una forma en que el Higgs se desintegra es **H → Z + Z\\* → 4 leptones** (electrones o muones). "
            "Es un canal muy raro, solo alrededor de 1 Higgs de cada 8,000, pero muy **limpio**: los leptones "
            "se miden con gran precisión y hay poco ruido de fondo. Junto con la desintegración en dos fotones, "
            "fue uno de los canales del descubrimiento de 2012."
        )
        st.write(
            "Como el Higgs (125 GeV) pesa menos que dos Z (2 × 91 GeV), una de las dos Z es *virtual* y "
            "tiene una masa menor. Por eso en los cortes la masa de Z2 se permite desde 12 GeV."
        )
        st.write("Hay tres combinaciones finales: **4 muones**, **4 electrones** o **2 electrones + 2 muones**.")
    with t3:
        st.header("Estadística básica")
        imagen("poisson.png")
        st.markdown(
            "- **Histograma:** cuenta cuántos eventos caen en cada intervalo de masa.\n"
            "- **Incertidumbre estadística:** si en un intervalo hay N eventos, la incertidumbre es √N. "
            "Por eso los puntos de los datos llevan barras de error.\n"
            "- **Fluctuaciones:** aunque el fondo esperado sea 10 eventos, a veces se observan 8 y a veces 13. "
            "Un exceso solo es convincente si es mucho mayor que esas fluctuaciones.\n"
            "- **5 sigmas (5σ):** el estándar para declarar un descubrimiento. Significa que, sin una partícula "
            "nueva, un exceso así ocurriría por azar aproximadamente 1 vez de cada 3.5 millones."
        )
        st.latex(r"\text{incertidumbre} \approx \sqrt{N}")
    with t4:
        st.header("Selección de eventos: los cortes")
        st.write(
            "Entre millones de eventos, casi todos contienen leptones *falsos* (por ejemplo, de otras "
            "partículas que se parecen a un leptón). Un **corte** (o filtro) descarta eventos que no "
            "se comportan como un Higgs real. Los que usamos:"
        )
        st.table(tabla_oficial_resumen())
        st.info("Los cortes se aplican en orden y cada uno incluye a los anteriores. Cuidado: si los aprietas "
                "demasiado también pierdes señal. Un buen análisis busca el equilibrio entre eliminar fondo y "
                "conservar señal.")


def tabla_oficial_resumen():
    """Corte -> explicación física (sin valores, esos están en el tutorial)."""
    return pd.DataFrame({
        "Corte": ["pT mínimo", "|η| máximo", "|dxy| y |dz|", "SIP3D", "Aislamiento", "Carga total = 0",
                  "pT1 y pT2", "Emparejamiento Z+Z", "Masas de Z1 y Z2", "mll mínima", "m4l mínima"],
        "Explicación física": [
            "Los leptones de baja energía suelen ser falsos o ruido.",
            "Más allá de cierto ángulo el detector no mide bien.",
            "Un leptón del Higgs nace en el punto de la colisión; si pasa lejos, viene de otro lugar.",
            "Mide qué tan compatible es el leptón con nacer en el punto de colisión.",
            "Los leptones buenos están 'solos'; los falsos vienen rodeados de otras partículas.",
            "El Higgs es neutro: 2 leptones positivos y 2 negativos suman cero.",
            "Los dos leptones más energéticos deben pasar un umbral (por el sistema de disparo).",
            "Deben poder formarse dos pares del mismo sabor y carga opuesta (dos Z).",
            "Z1 debe tener una masa cercana a la de la Z y Z2 puede ser virtual.",
            "Elimina resonancias de baja masa que no son del Higgs.",
            "Se descartan masas demasiado bajas para el Higgs.",
        ],
    }).set_index("Corte")


def seccion_mc():
    st.title("Simulaciones Monte Carlo (MC)")
    t1, t2, t3, t4, t5 = st.tabs(["¿Qué es una simulación?", "Cómo se simula una colisión",
                                  "Señal y fondo", "Pesos y luminosidad", "Limitaciones"])
    with t1:
        st.header("¿Qué es una simulación Monte Carlo?")
        st.write(
            "Es un programa que **inventa colisiones** siguiendo las leyes de la física, con números "
            "aleatorios (de ahí el nombre, por el casino de Montecarlo). También simula cómo el "
            "detector vería las partículas. Así sabemos *qué esperar* antes de mirar los datos reales."
        )
        imagen("montecarlo_flujo.png", "Las simulaciones y los datos reales terminan en los mismos tipos de archivo.")
        st.subheader("Pruébalo: estima π con números aleatorios")
        demo_pi()
    with t2:
        st.header("Cómo se simula una colisión")
        st.markdown(
            "1. **Generador de eventos** (por ejemplo, Pythia): sortea qué proceso ocurre y con qué energías, "
            "según las probabilidades que da la teoría.\n"
            "2. **Quarks y gluones → partículas:** los quarks no existen solos; se simula cómo se convierten "
            "en chorros de partículas que sí veremos.\n"
            "3. **Simulación del detector** (con un programa llamado GEANT4): se calcula cómo cada partícula "
            "atraviesa el material de CMS y qué señales deja.\n"
            "4. **Reconstrucción:** se aplica el *mismo* programa que a los datos reales."
        )
        st.info("Por eso los archivos de simulación tienen las mismas columnas que los de datos reales: "
                "podemos tratarlos exactamente igual.")
        st.write("Se simulan millones de eventos porque los procesos raros ocurren muy pocas veces, y para "
                 "tener una predicción con poca incertidumbre estadística hacen falta muchos.")
    with t3:
        st.header("Señal y fondo")
        imagen("senal_fondo.png")
        st.markdown(
            "- **Señal:** lo que buscamos. Aquí, la simulación de Higgs → ZZ → 4 leptones.\n"
            "- **Fondo:** otros procesos que también producen 4 leptones y se parecen a la señal. Aquí, "
            "**ZZ → 4μ, ZZ → 4e y ZZ → 2e2μ**.\n"
            "- Si en los datos hay *más* eventos que el fondo cerca de 125 GeV, hay algo extra: el Higgs."
        )
        st.subheader("Las simulaciones que usamos")
        st.table(tabla_simulaciones())
    with t4:
        st.header("Pesos: cómo comparar simulación con datos")
        imagen("pesos.png", "Ejemplo con la simulación ZZ → 4μ.")
        st.write(
            "Se simulan millones de eventos, pero no son tantos como los reales. Cada evento simulado recibe un "
            "**peso** para que el total corresponda a lo que se esperaría en los datos reales:"
        )
        st.latex(r"\text{peso} = \frac{L \times \sigma \times k}{N_{\text{generados}}}")
        st.markdown(
            "- **L**: luminosidad, 11,580 pb⁻¹.\n"
            "- **σ**: sección eficaz del proceso (qué tan probable es), en pb.\n"
            "- **k**: factor de corrección a órdenes superiores (en el fondo ZZ vale 1.386).\n"
            "- **N generados**: eventos que produjo el programa."
        )
        st.write("Cuando haces la gráfica, cada evento simulado cuenta con su peso y no como '1'. Los datos reales "
                 "siempre pesan 1.")
    with t5:
        st.header("Limitaciones")
        st.write(
            "Nuestras simulaciones solo incluyen el fondo **ZZ**. En el análisis completo de CMS hay otros "
            "fondos (por ejemplo, Z + jets), que aquí no están. Por eso no todo lo que ves en los datos "
            "reales se puede explicar con las simulaciones."
        )
        st.markdown(
            "- Solo usamos una parte de los datos y de los procesos.\n"
            "- No incluimos incertidumbres sistemáticas (por ejemplo, la eficiencia de identificar leptones).\n"
            "- Las simulaciones también tienen incertidumbre estadística."
        )
        st.info("Parte de hacer ciencia es saber qué incluye tu modelo y qué no.")


def seccion_analisis():
    st.title("Analiza los datos")
    d = cargar_muestra()
    t1, t2, t3, t4, t5 = st.tabs(["Los archivos CSV", "Tutorial de cortes", "Tabla cutflow",
                                  "Resultado final", "Preguntas para pensar"])

    with t1:
        st.header("¿Cómo se ven los archivos CSV?")
        st.write(
            "Cada fila es un **evento** con 4 leptones. Las columnas con número (1 a 4) se repiten para "
            "cada leptón, ordenados por pT."
        )
        st.table(pd.DataFrame({
            "Columna": ["run, luminosityBlock, event", "MET_pt, MET_phi", "PID1…PID4", "Q1…Q4", "pt1…pt4",
                        "eta1…eta4", "phi1…phi4", "mass1…mass4", "dxy1…dxy4 y dz1…dz4",
                        "dxyErr1…dzErr4", "pfRelIso03_all, pfRelIso04_all"],
            "Significado": [
                "Identifican el evento de forma única.",
                "Energía transversal 'perdida' (neutrinos, por ejemplo).",
                "Tipo de partícula: ±11 es electrón, ±13 es muón.",
                "Carga eléctrica (+1 o −1).",
                "Momento transversal (GeV).",
                "Ángulo respecto al haz.",
                "Ángulo alrededor del haz.",
                "Masa del leptón (GeV).",
                "Distancia de la trayectoria al punto de colisión (cm).",
                "Incertidumbre de esas distancias.",
                "Aislamiento relativo con cono 0.3 y 0.4.",
            ],
        }).set_index("Columna"))
        if d is not None:
            st.write("**Muestra reducida que usa esta página** (una fila por evento, ya resumida):")
            st.dataframe(d.sample(8, random_state=1).reset_index(drop=True))

    with t2:
        if d is None:
            aviso_sin_muestra()
        else:
            tutorial_cortes(d)

    with t3:
        st.header("Tabla cutflow")
        if d is None:
            aviso_sin_muestra()
        else:
            st.subheader("Con los cortes oficiales")
            st.write("Eventos que sobreviven después de aplicar cada filtro, en orden. Para las simulaciones "
                     "son eventos esperados a 11.58 fb⁻¹.")
            st.dataframe(tabla_cutflow(d, ORDEN, OFICIAL).style.format("{:.2f}").format("{:.0f}", subset=[DATOS]))
            st.caption("La tabla parte de una muestra que ya incluye una selección muy suelta (m4l entre 70 y "
                       "180 GeV), por eso las primeras filas pueden ser menores que en el análisis completo.")
            st.subheader("Con los cortes que elegiste en el tutorial")
            p, activos = leer_estado()
            st.dataframe(tabla_cutflow(d, activos, p).style.format("{:.2f}").format("{:.0f}", subset=[DATOS]))

    with t4:
        st.header("Resultado final")
        if d is None:
            aviso_sin_muestra()
        else:
            final = aplicar(d, ORDEN, OFICIAL)
            st.pyplot(figura(final, "Todos los cortes oficiales aplicados"))
            h_zz, h_h, h_dat = histogramas(final)
            c1, c2, c3 = st.columns(3)
            c1.metric("Eventos de datos en la gráfica", f"{int(h_dat.sum())}")
            c2.metric("Fondo ZZ esperado", f"{h_zz.sum():.1f}")
            c3.metric("Higgs esperado", f"{h_h.sum():.1f}")
            cerca = final[(final["m4l"] > 120) & (final["m4l"] < 130)]
            n_dat = int((cerca["muestra"] == DATOS).sum())
            esp_zz = cerca.loc[cerca["muestra"].isin(MUESTRAS_ZZ), "peso"].sum()
            esp_h = cerca.loc[cerca["muestra"] == HIGGS, "peso"].sum()
            st.write(
                f"Cerca de 125 GeV (entre 120 y 130 GeV) hay **{n_dat}** eventos de datos, y la "
                f"simulación espera **{esp_zz:.1f}** de fondo ZZ y **{esp_h:.1f}** del Higgs."
            )
            st.caption("¿Es suficiente para hablar de un descubrimiento? Recuerda la incertidumbre √N y que "
                       "solo usamos una parte de los datos y un fondo incompleto.")
            if RUTA_FINAL.exists():
                with st.expander("Gráfica generada con el script 03_grafica_final.py"):
                    st.image(str(RUTA_FINAL))

    with t5:
        st.header("Preguntas para pensar")
        st.markdown(
            "1. ¿Qué corte elimina más eventos de datos reales? ¿Y cuál afecta menos a la señal del Higgs?\n"
            "2. Si aprietas demasiado los cortes, ¿qué le pasa a la señal? ¿Por qué no conviene?\n"
            "3. ¿Qué corte te pareció más importante para que aparezca el pico? Justifica tu respuesta.\n"
            "4. ¿Por qué crees que se exige que la carga total sea cero?\n"
            "5. Con tus cortes, ¿los datos de CMS están de acuerdo con la simulación ZZ + Higgs?"
        )


# Valores iniciales de los controles: tan sueltos que NO eliminan nada
LOOSE = {
    "pt_e": 0.0, "pt_mu": 0.0, "eta_e": 3.0, "eta_mu": 3.0, "dxy": 2.0, "dz": 3.0, "sip": 10.0, "iso": 1.0,
    "pt1": 0.0, "pt2": 0.0, "mz1_min": 0.0, "mz1_max": 180.0, "mz2_min": 0.0, "mz2_max": 180.0,
    "mll": 0.0, "m4l_min": 70.0,
}

# Para cada corte: (título, explicación física, pista, controles). Cada control: (texto, mín, máx, paso, clave)
PASOS = {
    "pt": ("Energía mínima de los leptones",
           "Los leptones de poca energía suelen ser falsos o ruido de fondo.",
           "Los electrones suelen pedir un umbral un poco mayor que los muones. Prueba entre 3 y 10 GeV.",
           [("pT mínimo electrones (GeV)", 0.0, 20.0, 0.5, "pt_e"), ("pT mínimo muones (GeV)", 0.0, 20.0, 0.5, "pt_mu")]),
    "eta": ("Ángulo respecto al haz",
            "Muy cerca del haz el detector no mide bien. η = 0 es perpendicular al haz.",
            "Los muones se miden bien un poco menos lejos que los electrones. Prueba entre 2.0 y 2.6.",
            [("|η| máximo electrones", 1.0, 3.0, 0.1, "eta_e"), ("|η| máximo muones", 1.0, 3.0, 0.1, "eta_mu")]),
    "ip": ("Distancia al punto de colisión",
           "Un leptón del Higgs nace en el punto de la colisión. Si su trayectoria pasa lejos, viene de otro lugar.",
           "dxy (en el plano perpendicular al haz) suele ser más estricto que dz (a lo largo del haz). Prueba valores menores a 1 cm.",
           [("|dxy| máximo (cm)", 0.05, 2.0, 0.05, "dxy"), ("|dz| máximo (cm)", 0.1, 3.0, 0.1, "dz")]),
    "sip": ("SIP3D",
            "Mide qué tan compatible es el leptón con nacer en el punto de colisión, tomando en cuenta la incertidumbre.",
            "Es un número sin unidades; los leptones buenos tienen valores pequeños. Prueba entre 2 y 6.",
            [("SIP3D máximo", 1.0, 10.0, 0.5, "sip")]),
    "iso": ("Aislamiento",
            "Los leptones buenos están 'solos'; los falsos vienen rodeados de otras partículas.",
            "Es un cociente: valores pequeños significan más aislado. Prueba entre 0.2 y 0.6.",
            [("Aislamiento relativo máximo", 0.05, 1.0, 0.05, "iso")]),
    "carga": ("Carga total cero",
              "El Higgs es neutro, así que los 4 leptones deben sumar carga 0 (2 positivos y 2 negativos).",
              "Este corte no tiene valor que elegir: solo actívalo.", []),
    "pt12": ("Los dos leptones más duros",
             "Los dos leptones más energéticos deben superar un umbral, por la forma en que el detector decide guardar el evento.",
             "El primero siempre pide más que el segundo. Prueba 15 a 25 GeV para pT1 y 8 a 12 GeV para pT2.",
             [("pT1 mínimo (GeV)", 0.0, 40.0, 1.0, "pt1"), ("pT2 mínimo (GeV)", 0.0, 40.0, 1.0, "pt2")]),
    "pares": ("Emparejamiento en dos Z",
              "Debe poderse formar dos pares de leptones del mismo tipo y carga opuesta (cada par viene de una Z).",
              "Tampoco tiene valor: solo actívalo.", []),
    "mz1": ("Masa de la primera Z",
            "Z1 es la Z con masa más cercana a 91 GeV, así que debe estar en una ventana alrededor de ese valor.",
            "Mueve el mínimo hacia arriba (entre 30 y 50 GeV) y deja el máximo por arriba de 91.",
            [("Masa mínima de Z1 (GeV)", 0.0, 180.0, 1.0, "mz1_min"), ("Masa máxima de Z1 (GeV)", 0.0, 180.0, 1.0, "mz1_max")]),
    "mz2": ("Masa de la segunda Z",
            "Como el Higgs pesa menos que dos Z, la segunda Z es virtual y puede tener una masa menor.",
            "El mínimo es mucho más bajo que el de Z1. Prueba entre 8 y 15 GeV.",
            [("Masa mínima de Z2 (GeV)", 0.0, 180.0, 1.0, "mz2_min"), ("Masa máxima de Z2 (GeV)", 0.0, 180.0, 1.0, "mz2_max")]),
    "mll": ("Masa mínima de pares",
            "Elimina eventos con pares de leptones de masa muy baja, que vienen de otras resonancias.",
            "Es un valor pequeño. Prueba entre 2 y 8 GeV.",
            [("mll mínima (GeV)", 0.0, 20.0, 0.5, "mll")]),
    "m4l": ("Masa mínima de los 4 leptones",
            "Se descartan masas demasiado bajas para ser un Higgs.",
            "Está en el borde izquierdo de la gráfica.",
            [("m4l mínima (GeV)", 70.0, 120.0, 1.0, "m4l_min")]),
}


IMAGEN_PASO = {"eta": "eta_pt.png", "ip": "impacto.png", "iso": "aislamiento.png"}


def leer_estado():
    """Devuelve (valores de los cortes, lista de cortes activos) desde los controles de la página."""
    p = {k: st.session_state.get(k, LOOSE[k]) for k in LOOSE}
    activos = [c for c in ORDEN if st.session_state.get(f"on_{c}", False)]
    return p, activos


def mostrar_respuesta():
    """Botón: pone todos los cortes en sus valores oficiales."""
    for k, v in OFICIAL.items():
        st.session_state[k] = v
    for c in ORDEN:
        st.session_state[f"on_{c}"] = True


def empezar_de_nuevo():
    """Botón: quita todos los cortes."""
    for k, v in LOOSE.items():
        st.session_state[k] = v
    for c in ORDEN:
        st.session_state[f"on_{c}"] = False


def tabla_oficial():
    """Tabla con los cortes oficiales, el valor y su razón física."""
    o = OFICIAL
    valores = {
        "pt": f"Electrones > {o['pt_e']:g} GeV; muones > {o['pt_mu']:g} GeV",
        "eta": f"Electrones < {o['eta_e']:g}; muones < {o['eta_mu']:g}",
        "ip": f"|dxy| < {o['dxy']:g} cm y |dz| < {o['dz']:g} cm",
        "sip": f"SIP3D < {o['sip']:g}",
        "iso": f"Aislamiento < {o['iso']:g}",
        "carga": "Suma de cargas = 0",
        "pt12": f"pT1 > {o['pt1']:g} GeV y pT2 > {o['pt2']:g} GeV",
        "pares": "Debe existir el emparejamiento Z + Z",
        "mz1": f"{o['mz1_min']:g} < mZ1 < {o['mz1_max']:g} GeV",
        "mz2": f"{o['mz2_min']:g} < mZ2 < {o['mz2_max']:g} GeV",
        "mll": f"mll mínima > {o['mll']:g} GeV",
        "m4l": f"m4l > {o['m4l_min']:g} GeV",
    }
    return pd.DataFrame({
        "Paso": range(1, len(ORDEN) + 1),
        "Corte": [PASOS[c][0] for c in ORDEN],
        "Valor oficial": [valores[c] for c in ORDEN],
        "¿Por qué?": [PASOS[c][1] for c in ORDEN],
    }).set_index("Paso")


def tutorial_cortes(d):
    """Reto guiado: partir sin cortes y elegir valores hasta parecerse a la selección oficial."""
    st.header("Tutorial: limpia los datos paso a paso")
    st.write(
        "Empiezas **sin ningún corte**, con todo el ruido. Tu misión es activar los cortes y elegir sus "
        "valores para que la gráfica **\"Tus cortes\"** se parezca a la gráfica **\"Meta\"**, que usa la "
        "selección oficial. Avanza por los pasos, uno a la vez, y fíjate cómo cambia la gráfica."
    )
    with st.expander("📋 Tabla de cortes oficiales", expanded=True):
        st.table(tabla_oficial())

    for k, v in LOOSE.items():
        st.session_state.setdefault(k, v)
    for c in ORDEN:
        st.session_state.setdefault(f"on_{c}", False)

    c_ctrl, c_graf = st.columns([1, 2.2])
    with c_ctrl:
        b1, b2 = st.columns(2)
        b1.button("Empezar de nuevo", on_click=empezar_de_nuevo)
        b2.button("🔑 Mostrar la respuesta correcta", on_click=mostrar_respuesta)
        for n, clave in enumerate(ORDEN, start=1):
            titulo, explicacion, pista, controles = PASOS[clave]
            activo = st.session_state.get(f"on_{clave}", False)
            with st.expander(f"Paso {n}: {titulo}" + (" ✅" if activo else ""), expanded=(n == 1)):
                st.caption(explicacion)
                if clave in IMAGEN_PASO:
                    imagen(IMAGEN_PASO[clave], ancho=460)
                st.checkbox("Aplicar este corte", key=f"on_{clave}")
                for texto, mn, mx, paso, k in controles:
                    st.slider(texto, mn, mx, step=paso, key=k)
                st.caption("💡 Pista: " + pista)

    p, activos = leer_estado()
    sel = aplicar(d, activos, p)
    meta = aplicar(d, ORDEN, OFICIAL)
    with c_graf:
        g1, g2 = st.columns(2)
        with g1:
            st.pyplot(figura(sel, "Tus cortes"))
        with g2:
            st.pyplot(figura(meta, "Meta: cortes oficiales"))

        j = parecido(d, activos, p)
        st.progress(min(max(j, 0.0), 1.0), text=f"Parecido con la selección oficial: {j * 100:.0f} %")
        n_dat = int((sel["muestra"] == DATOS).sum())
        n_meta = int((meta["muestra"] == DATOS).sum())
        h_zz, h_h, _ = histogramas(sel)
        m1, m2, m3 = st.columns(3)
        m1.metric("Datos que sobreviven", f"{n_dat}", delta=f"{n_dat - n_meta:+d} vs oficial", delta_color="off")
        m2.metric("Fondo ZZ esperado", f"{h_zz.sum():.1f}")
        m3.metric("Higgs esperado", f"{h_h.sum():.1f}")

        if not activos:
            st.info("Todavía no has aplicado ningún corte. Empieza con el Paso 1: ábrelo, activa el corte y "
                    "mueve los controles.")
        elif j >= 0.95:
            st.success("¡Lo lograste! Tu selección es prácticamente la oficial. Compara tus valores con la "
                       "tabla de cortes oficiales.")
        elif n_dat > n_meta * 1.1:
            st.info(f"Todavía quedan {n_dat - n_meta} eventos de datos más que en la gráfica oficial. Activa "
                    "más cortes o hazlos más estrictos.")
        elif n_dat < n_meta * 0.9:
            st.warning("Estás eliminando más eventos que la selección oficial. Afloja algún corte.")
        else:
            st.info("Vas muy cerca: el número de eventos es parecido, pero no son exactamente los mismos. "
                    "Revisa tus valores contra la tabla.")
        st.caption(f"Has activado {len(activos)} de {len(ORDEN)} cortes.")


PREGUNTAS = [
    ("¿Cuál de estas partículas es un leptón?",
     ["Protón", "Electrón", "Quark up", "Fotón"], "Electrón",
     "El electrón es un leptón; el protón está hecho de quarks."),
    ("¿Por qué un muón puede atravesar todo el detector CMS?",
     ["Porque no tiene carga", "Porque interactúa poco con la materia y vive el tiempo suficiente",
      "Porque es más pesado que el protón", "Porque lo frena el calorímetro"],
     "Porque interactúa poco con la materia y vive el tiempo suficiente",
     "Los electrones se frenan en el calorímetro; los muones llegan hasta las cámaras externas."),
    ("¿Por qué se busca el Higgs en 4 leptones?",
     ["Porque es muy frecuente", "Porque es raro pero muy limpio y fácil de medir",
      "Porque no hay fondo", "Porque los leptones son neutros"],
     "Porque es raro pero muy limpio y fácil de medir",
     "Es un canal raro, pero los leptones se miden con precisión."),
    ("La masa invariante de 4 leptones que vienen de un Higgs debe estar cerca de…",
     ["1 GeV", "91 GeV", "125 GeV", "1,000 GeV"], "125 GeV",
     "125 GeV es la masa del Higgs."),
    ("¿Para qué sirven las simulaciones Monte Carlo?",
     ["Para reemplazar los datos", "Para saber qué esperar de la señal y del fondo",
      "Para detectar leptones", "Para acelerar protones"],
     "Para saber qué esperar de la señal y del fondo",
     "Se comparan con los datos reales."),
    ("¿Qué hace el 'peso' de un evento simulado?",
     ["Indica su masa", "Hace que el total simulado corresponda a lo esperado en los datos reales",
      "Mide su carga", "Lo elimina de la gráfica"],
     "Hace que el total simulado corresponda a lo esperado en los datos reales",
     "peso = L × σ × k / N generados."),
    ("Si en un intervalo hay N eventos, su incertidumbre estadística es aproximadamente…",
     ["N", "√N", "N²", "1/N"], "√N",
     "En conteos, la incertidumbre es la raíz del número de eventos."),
    ("¿Por qué se exige que la carga total de los 4 leptones sea cero?",
     ["Porque el Higgs es neutro", "Porque los muones no tienen carga",
      "Para que el detector funcione", "No es necesario"],
     "Porque el Higgs es neutro",
     "Dos Z neutras dan 2 leptones positivos y 2 negativos."),
]

GLOSARIO = {
    "Evento": "Una colisión registrada por el detector.",
    "Leptón": "Partícula elemental como el electrón o el muón.",
    "Fermión / bosón": "Fermiones forman la materia; bosones transmiten las fuerzas.",
    "Campo de Higgs": "Campo que llena el espacio y da masa a las partículas elementales.",
    "Partícula virtual": "Partícula que existe solo un instante y no tiene su masa 'normal' (como la Z*).",
    "GeV": "Unidad de energía y masa (1 GeV ≈ masa de un protón).",
    "pT": "Momento transversal: perpendicular al haz de protones.",
    "η (eta)": "Ángulo respecto al haz; η = 0 es perpendicular.",
    "dxy, dz": "Distancia de la trayectoria al punto de colisión (cm).",
    "SIP3D": "Qué tan compatible es el leptón con nacer en el punto de colisión.",
    "Aislamiento": "Cuánta otra actividad hay alrededor del leptón.",
    "Trigger": "Sistema que decide, en una fracción de segundo, qué eventos se guardan.",
    "Masa invariante": "Masa de una partícula reconstruida con sus hijas.",
    "Monte Carlo": "Simulación por computadora de colisiones.",
    "Señal / fondo": "Lo que buscamos / lo que se le parece.",
    "Sección eficaz (σ)": "Probabilidad de que ocurra un proceso.",
    "Luminosidad (L)": "Cantidad de colisiones acumuladas.",
    "fb⁻¹, pb⁻¹": "Unidades de luminosidad integrada (1 fb⁻¹ = 1,000 pb⁻¹).",
    "Peso": "Factor que hace que una simulación corresponda a la cantidad esperada en datos reales.",
    "Cutflow": "Tabla de cuántos eventos sobreviven a cada corte.",
    "5 sigmas (5σ)": "Estándar para declarar un descubrimiento en física de partículas.",
}


def seccion_conclusiones():
    st.title("Conclusiones, glosario y autoevaluación")
    t1, t2, t3 = st.tabs(["Conclusiones", "Glosario", "Autoevaluación"])
    with t1:
        st.header("Lo que aprendiste")
        st.markdown(
            "- Los datos de colisiones contienen muchísimo ruido; los **cortes** sirven para limpiarlos.\n"
            "- Las **simulaciones Monte Carlo** nos dicen qué esperar de la señal y del fondo, y los **pesos** "
            "permiten compararlas con los datos.\n"
            "- El Higgs se busca como un **pico** cerca de 125 GeV en la masa de 4 leptones.\n"
            "- Para afirmar un descubrimiento hace falta compararlo con la incertidumbre estadística."
        )
        st.header("¿Y ahora qué?")
        st.markdown(
            "- Prueba con otros cortes en el tutorial y observa qué pasa con la señal.\n"
            "- Explora otros canales del Higgs, como la desintegración en dos fotones.\n"
            "- Los experimentos del LHC siguen buscando física nueva: materia oscura, partículas todavía "
            "desconocidas y medidas más precisas del Higgs."
        )
    with t2:
        st.table(pd.DataFrame({"Significado": GLOSARIO.values()}, index=list(GLOSARIO.keys())))
    with t3:
        respuestas = []
        for i, (pregunta, opciones, _, _) in enumerate(PREGUNTAS):
            respuestas.append(st.radio(f"{i + 1}. {pregunta}", opciones, index=None, key=f"q{i}"))
        if st.button("Revisar mis respuestas"):
            buenas = 0
            for i, ((_, _, correcta, explica), r) in enumerate(zip(PREGUNTAS, respuestas)):
                if r == correcta:
                    buenas += 1
                    st.success(f"{i + 1}. Correcta. {explica}")
                else:
                    st.error(f"{i + 1}. Respuesta correcta: {correcta}. {explica}")
            st.write(f"**Obtuviste {buenas} de {len(PREGUNTAS)}.**")


def seccion_descargas():
    st.title("Descarga de datos y créditos")
    st.markdown(
        f"- 💻 **Código y resultados (GitHub):** [{REPO_URL}]({REPO_URL})\n"
        f"- 📁 **CSV completos (Google Drive):** [{DRIVE_URL}]({DRIVE_URL})"
    )
    st.info("Los CSV completos pesan varios GB. Descárgalos solo si vas a trabajar en tu computadora.")
    st.subheader("Para ir más lejos")
    st.markdown(
        "- [CERN Open Data](https://opendata.cern.ch): datos públicos de CMS y otros experimentos.\n"
        "- [ATLAS Open Data](https://opendata.atlas.cern): recursos educativos del experimento ATLAS.\n"
        "- [CERN](https://home.cern) y [CMS](https://cms.cern): noticias y material divulgativo."
    )
    st.subheader("Créditos y referencias")
    st.markdown(
        "- Datos: colaboración CMS, portal [CERN Open Data](https://opendata.cern.ch/record/12360), "
        "licencia CC0.\n"
        "- Análisis basado en: CMS Collaboration, *Observation of a new boson at a mass of 125 GeV*, "
        "arXiv:1207.7235.\n"
        "- Las ilustraciones son esquemas hechos para esta página; no son imágenes reales de CMS ni están a escala.\n"
        "- Esta página está inspirada en la aplicación *ATLAS Open Data en el aula*."
    )


# =====================================================================================
# 4. NAVEGACIÓN (menú lateral)
# =====================================================================================
SECCIONES = {
    "Introducción": seccion_intro,
    "Fundamentos de la física de partículas": seccion_fundamentos,
    "El LHC y el detector CMS": seccion_lhc,
    "Cómo se analiza un dato": seccion_analisis_teoria,
    "Simulaciones Monte Carlo": seccion_mc,
    "Analiza los datos": seccion_analisis,
    "Conclusiones y autoevaluación": seccion_conclusiones,
    "Descarga de datos y créditos": seccion_descargas,
}

st.sidebar.title("El Higgs en CMS Open Data")
st.markdown(
    """<style>
    html, body, [class*="css"], .stMarkdown p, .stMarkdown li, label, .stCaption, td, th { font-size: 1.08rem; }
    .stMarkdown p, .stMarkdown li { line-height: 1.65; }
    [data-testid="stSidebar"] * { font-size: 1.05rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1.05rem; }
    </style>""",
    unsafe_allow_html=True,
)
st.session_state.setdefault("seccion", list(SECCIONES.keys())[0])
st.sidebar.radio("Secciones", list(SECCIONES.keys()), key="seccion", label_visibility="collapsed")
SECCIONES[st.session_state["seccion"]]()
