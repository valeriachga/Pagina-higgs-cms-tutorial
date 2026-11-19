"""
preparar_muestra.py
===================

QUÉ HACE ESTE SCRIPT
--------------------
Reduce los 8 CSV de 4 leptones (varios GB) a UN archivo pequeño,
`web_data/muestra_eventos.csv.gz`, que es el que lee la página web.

La página no puede cargar los CSV completos (la descarga el navegador de cada
estudiante y pesarían varios GB). Así que, en lugar de guardar los leptones, este script guarda UNA FILA
POR EVENTO con las variables ya resumidas que la página necesita para aplicar
los cortes de forma interactiva.

Las definiciones (SIP3D, aislamiento, emparejamiento en dos Z, mll mínima,
pesos, eliminación de eventos duplicados) son IGUALES a las de 02_cutflow.py,
para que la página reproduzca exactamente tus resultados.

VARIABLES POR EVENTO QUE SE GUARDAN
-----------------------------------
  m4l                         masa invariante de los 4 leptones (GeV)
  pt_e_min, pt_mu_min         pT mínimo entre los electrones / entre los muones
  eta_e_max, eta_mu_max       |eta| máximo entre los electrones / entre los muones
  dxy_max, dz_max             |dxy| y |dz| máximos entre los 4 leptones
  sip_max                     SIP3D máximo entre los 4 leptones
  iso_max                     aislamiento relativo máximo entre los 4 leptones
  q_sum                       suma de las 4 cargas
  pt1, pt2                    pT de los dos leptones más duros
  mz1, mz2                    masas de Z1 y Z2 (vacío si no hay emparejamiento Z+Z)
  mll_min                     masa mínima de cualquier par de carga opuesta
  n_e, n_mu                   número de electrones y de muones
  muestra, peso               nombre de la muestra y peso del evento

PRESELECCIÓN RELAJADA
---------------------
Para que el archivo sea pequeño solo se guardan eventos con m4l en 70-180 GeV y
con cortes MUY sueltos (diccionario PRE). Cualquier valor de los controles de la
página que esté dentro de esos límites da resultados exactos. Por eso en la
página los controles nunca deben ser más sueltos que PRE.

CÓMO EJECUTARLO (desde la carpeta principal del repositorio)
------------------------------------------------------------
    python scripts/preparar_muestra.py
"""

import os

import numpy as np
import pandas as pd

# =============================================================================
# 1. CONFIGURACIÓN (la misma estructura que 02_cutflow.py)
# =============================================================================
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARPETA_DATOS = os.path.join(RAIZ, "data")                    # CSV de entrada
SALIDA = os.path.join(RAIZ, "web_data", "muestra_eventos.csv.gz")

MZ = 91.1876
CHUNKSIZE = 200_000
VENTANA = (70.0, 180.0)           # rango de m4l que se guarda

DATOS_REALES = [
    os.path.join(CARPETA_DATOS, "Run2012B_DoubleElectron_Data_Aplanada.csv"),
    os.path.join(CARPETA_DATOS, "Run2012B_DoubleMuParked_Data_Aplanada.csv"),
    os.path.join(CARPETA_DATOS, "Run2012C_DoubleElectron_Data_Aplanada.csv"),
    os.path.join(CARPETA_DATOS, "Run2012C_DoubleMuParked_Data_Aplanada.csv"),
]

# nombre interno -> (archivo, nombre que verá el estudiante, xsec [pb], nevt, k)
LUMI = 11580.0
SIMULACIONES = {
    "SMHiggsToZZTo4L": (os.path.join(CARPETA_DATOS, "SMHiggs_4lep.csv"),   "Higgs",   0.0065, 299973.0,  1.0),
    "ZZTo4mu":         (os.path.join(CARPETA_DATOS, "ZZTo4mu_4lep.csv"),   "ZZ→4μ",   0.077,  1499064.0, 1.386),
    "ZZTo4e":          (os.path.join(CARPETA_DATOS, "ZZTo4e_4lep.csv"),    "ZZ→4e",   0.077,  1499093.0, 1.386),
    "ZZTo2e2mu":       (os.path.join(CARPETA_DATOS, "ZZTo2e2mu_4lep.csv"), "ZZ→2e2μ", 0.18,   1497445.0, 1.386),
}

# Preselección RELAJADA (ver explicación arriba). Valores tipo "límite de los controles".
PRE = {"eta_e_max": 3.0, "eta_mu_max": 3.0, "dxy_max": 2.0, "dz_max": 3.0,
       "sip_max": 10.0, "iso_max": 1.0}

# Tope de eventos por simulación (para que el archivo pese poco). Al recortar se
# corrige el peso, así el total esperado no cambia. Los datos reales NO se recortan.
MAX_EVENTOS_SIMULACION = 50_000

COLS_LEP = ["PID", "Q", "pt", "eta", "phi", "mass", "dxy", "dxyErr", "dz", "dzErr",
            "pfRelIso03_all", "pfRelIso04_all"]
NECESARIAS = {f"{c}{i}" for c in COLS_LEP for i in range(1, 5)} | {"run", "luminosityBlock", "event"}
COLS_VALIDAS = [f"{v}{i}" for v in ["PID", "Q", "pt", "eta", "phi", "mass"] for i in range(1, 5)]
SIN_VALOR = 9999.0   # valor que se guarda cuando una variable "falla siempre" (p. ej. dato faltante)


# =============================================================================
# 2. FUNCIONES DE FÍSICA (copiadas de 02_cutflow.py)
# =============================================================================
def p4(df, i):
    """Cuadrivector (E, px, py, pz) del leptón i (1 a 4)."""
    pt = df[f"pt{i}"].to_numpy()
    eta = df[f"eta{i}"].to_numpy()
    phi = df[f"phi{i}"].to_numpy()
    m = df[f"mass{i}"].to_numpy()
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    return np.stack([E, px, py, pz], axis=1)


def inv_mass(*vecs):
    """Masa invariante de un sistema de partículas (suma de cuadrivectores)."""
    s = sum(vecs)
    m2 = s[:, 0]**2 - s[:, 1]**2 - s[:, 2]**2 - s[:, 3]**2
    return np.sqrt(np.clip(m2, 0, None))


def sip3d(dxy, dxyErr, dz, dzErr):
    """SIP3D = IP3D / sigma_IP3D, con propagación de errores sin correlaciones (igual que 02_cutflow.py)."""
    dxy, dxyErr = np.asarray(dxy, float), np.asarray(dxyErr, float)
    dz, dzErr = np.asarray(dz, float), np.asarray(dzErr, float)
    ip = np.sqrt(dxy**2 + dz**2)
    sig = np.sqrt((dxy**2 * dxyErr**2 + dz**2 * dzErr**2) / np.clip(ip**2, 1e-12, None))
    return ip / np.clip(sig, 1e-12, None)


# =============================================================================
# 3. VARIABLES A NIVEL DE EVENTO
# =============================================================================
def variables_evento(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte un bloque de eventos (4 leptones por fila) en las variables por evento
    descritas arriba. NO aplica ningún corte; solo resume cada evento.
    """
    # Clave única del evento (para eliminar duplicados entre archivos de datos)
    claves = pd.util.hash_pandas_object(df[["run", "luminosityBlock", "event"]], index=False).to_numpy()

    # -999 significa "este leptón no existe": se pasa a NaN y se descartan esos eventos.
    df = df.replace(-999, np.nan)
    valido = df[COLS_VALIDAS].notna().all(axis=1).to_numpy()
    df, claves = df[valido], claves[valido]
    if len(df) == 0:
        return pd.DataFrame()

    n = len(df)
    v = [p4(df, i) for i in range(1, 5)]
    m4l = inv_mass(*v)

    arr = lambda nombre: np.stack([df[f"{nombre}{i}"].to_numpy(float) for i in range(1, 5)], axis=1)  # noqa: E731
    pid, q, pt, eta = np.abs(arr("PID")), arr("Q"), arr("pt"), np.abs(arr("eta"))
    dxy, dz, dxy_err, dz_err = arr("dxy"), arr("dz"), arr("dxyErr"), arr("dzErr")
    es_e, es_mu = pid == 11, pid == 13

    # Mínimo/máximo SOLO entre los leptones de cada sabor (si no hay de ese sabor, no restringe)
    pt_e_min = np.where(es_e, pt, np.inf).min(1)
    pt_mu_min = np.where(es_mu, pt, np.inf).min(1)
    eta_e_max = np.where(es_e, eta, 0.0).max(1)
    eta_mu_max = np.where(es_mu, eta, 0.0).max(1)

    def max_o_falla(x):
        """Máximo por evento; si falta algún valor, el evento 'falla siempre' (como en 02_cutflow.py)."""
        m = np.max(np.nan_to_num(x, nan=0.0), axis=1)
        return np.where(np.isnan(x).any(axis=1), SIN_VALOR, m)

    dxy_max = max_o_falla(np.abs(dxy))
    dz_max = max_o_falla(np.abs(dz))

    # SIP3D: si no se puede calcular no se rechaza el leptón (igual que 02_cutflow.py)
    sip = sip3d(dxy, dxy_err, dz, dz_err)
    sip_max = np.where(np.isnan(sip), 0.0, sip).max(1)

    # Aislamiento: cono 0.4; si no existe (electrones) se usa el de cono 0.3
    iso = np.stack([df[f"pfRelIso04_all{i}"].fillna(df[f"pfRelIso03_all{i}"]).to_numpy(float)
                    for i in range(1, 5)], axis=1)
    iso_max = max_o_falla(iso)

    pts_ordenados = np.sort(pt, axis=1)[:, ::-1]

    # mll mínima sobre todos los pares de carga opuesta
    mll_min = np.full(n, np.inf)
    for a in range(4):
        for b in range(a + 1, 4):
            os_ = (q[:, a] + q[:, b]) == 0
            mll_min = np.where(os_, np.minimum(mll_min, inv_mass(v[a], v[b])), mll_min)

    # Emparejamiento en dos Z (mismo sabor y carga opuesta); Z1 = la más cercana a MZ
    mejor_score = np.full(n, np.inf)
    mz1 = np.full(n, np.nan)
    mz2 = np.full(n, np.nan)
    for (a, b), (c, d) in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]:
        ok = ((pid[:, a] == pid[:, b]) & (q[:, a] + q[:, b] == 0)
              & (pid[:, c] == pid[:, d]) & (q[:, c] + q[:, d] == 0))
        mab, mcd = inv_mass(v[a], v[b]), inv_mass(v[c], v[d])
        ab_es_z1 = np.abs(mab - MZ) < np.abs(mcd - MZ)
        z1, z2 = np.where(ab_es_z1, mab, mcd), np.where(ab_es_z1, mcd, mab)
        mejor = ok & (np.abs(z1 - MZ) < mejor_score)
        mejor_score = np.where(mejor, np.abs(z1 - MZ), mejor_score)
        mz1, mz2 = np.where(mejor, z1, mz1), np.where(mejor, z2, mz2)

    return pd.DataFrame({
        "m4l": m4l,
        "pt_e_min": np.where(np.isinf(pt_e_min), SIN_VALOR, pt_e_min),
        "pt_mu_min": np.where(np.isinf(pt_mu_min), SIN_VALOR, pt_mu_min),
        "eta_e_max": eta_e_max, "eta_mu_max": eta_mu_max,
        "dxy_max": dxy_max, "dz_max": dz_max, "sip_max": sip_max, "iso_max": iso_max,
        "q_sum": q.sum(1),
        "pt1": pts_ordenados[:, 0], "pt2": pts_ordenados[:, 1],
        "mz1": mz1, "mz2": mz2,
        "mll_min": np.where(np.isinf(mll_min), SIN_VALOR, mll_min),
        "n_e": es_e.sum(1), "n_mu": es_mu.sum(1),
        "clave": claves,
    })


def preseleccion(ev: pd.DataFrame) -> pd.DataFrame:
    """Ventana de m4l + cortes muy sueltos (solo para reducir el tamaño del archivo)."""
    ok = (ev["m4l"] >= VENTANA[0]) & (ev["m4l"] <= VENTANA[1])
    for col, tope in PRE.items():
        ok &= ev[col] <= tope
    return ev[ok]


def procesar(ruta):
    """Lee un CSV por bloques y devuelve los eventos que pasan la preselección."""
    partes = []
    leidas = 0
    for bloque in pd.read_csv(ruta, chunksize=CHUNKSIZE, usecols=lambda c: c in NECESARIAS):
        ev = variables_evento(bloque)
        if len(ev):
            partes.append(preseleccion(ev))
        leidas += len(bloque)
        print(f"   {os.path.basename(ruta)}: {leidas} filas leídas", end="\r")
    print()
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


# =============================================================================
# 4. PROGRAMA PRINCIPAL
# =============================================================================
def main():
    resultados = []

    for interno, (ruta, nombre, xsec, nevt, k) in SIMULACIONES.items():
        print(f"Simulación {nombre}")
        ev = procesar(ruta)
        peso = LUMI * xsec * k / nevt           # igual que peso_por_evento() de 02_cutflow.py
        if len(ev) > MAX_EVENTOS_SIMULACION:
            peso *= len(ev) / MAX_EVENTOS_SIMULACION   # el peso sube para conservar el total
            ev = ev.sample(MAX_EVENTOS_SIMULACION, random_state=1)
        ev["muestra"], ev["peso"] = nombre, peso
        resultados.append(ev.drop(columns="clave"))
        print(f"   -> {len(ev)} eventos guardados, peso por evento = {peso:.6g}")

    print("Datos reales (4 datasets unidos)")
    partes = [procesar(r) for r in DATOS_REALES]
    datos = pd.concat(partes, ignore_index=True)
    antes = len(datos)
    datos = datos.drop_duplicates(subset="clave")      # mismo evento en dos datasets -> una sola vez
    print(f"   -> {len(datos)} eventos guardados ({antes - len(datos)} duplicados eliminados)")
    datos["muestra"], datos["peso"] = "Datos", 1.0
    resultados.append(datos.drop(columns="clave"))

    final = pd.concat(resultados, ignore_index=True)
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    # 6 cifras significativas: archivo pequeño sin mover eventos que estén justo en el borde de un corte
    final.to_csv(SALIDA, index=False, compression="gzip", float_format="%.6g")
    print(f"\nListo: {SALIDA}  ({len(final)} filas, {os.path.getsize(SALIDA) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
