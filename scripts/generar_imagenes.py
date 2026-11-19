"""
generar_imagenes.py
===================
Genera las ilustraciones de la página web (esquemas, NO fotografías ni imágenes
reales de CMS) y las guarda como PNG en la carpeta `imagenes/` del repositorio.

Todas usan un fondo claro propio, para que se vean bien tanto si el navegador
está en modo claro como en modo oscuro.

Cómo se usa (desde la carpeta principal del repositorio):
    python scripts/generar_imagenes.py

Si quieres cambiar colores o textos, edita este archivo y vuelve a ejecutarlo.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle, Ellipse, FancyBboxPatch, Rectangle, Wedge

SALIDA = Path(__file__).resolve().parent.parent / "imagenes"
SALIDA.mkdir(exist_ok=True)

FONDO, TINTA, GRIS, GRIS_CLARO = "#FAFAF8", "#222222", "#8A8A85", "#DEDDD7"
AZUL, ROJO, VERDE, AMARILLO, MORADO = "#3B6FB6", "#C8553D", "#4F9D69", "#E0A93B", "#7A5C9E"
AZUL_CLARO, VERDE_CLARO, AMARILLO_CLARO, ROJO_CLARO = "#DCE7F5", "#DCEEDF", "#F7EBCB", "#F4DBD4"


# ---------------------------------------------------------------- utilidades
def lienzo(w, h):
    """Figura de w x h pulgadas con un eje que ocupa todo; 1 unidad = 1 pulgada."""
    fig = Figure(figsize=(w, h), facecolor=FONDO)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def guardar(fig, nombre):
    fig.savefig(SALIDA / nombre, dpi=200, facecolor=FONDO)
    print("listo:", nombre)


def texto(ax, x, y, s, fs=10, color=TINTA, ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, **kw)


def caja(ax, x, y, w, h, fc, ec=None, s="", fs=10, color=TINTA, lw=1.2, radio=0.08, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radio}",
                                fc=fc, ec=ec or fc, lw=lw))
    if s:
        texto(ax, x + w / 2, y + h / 2, s, fs=fs, color=color, **kw)


def flecha(ax, p0, p1, color=GRIS, lw=1.6, ms=13, estilo="-|>"):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle=estilo, color=color, lw=lw,
                                                       mutation_scale=ms, shrinkA=0, shrinkB=0))


def onda(ax, p0, p1, color=ROJO, amp=0.07, ciclos=6, lw=1.8):
    """Línea ondulada (para bosones) entre dos puntos."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    largo = np.hypot(*d)
    u, n = d / largo, np.array([-d[1], d[0]]) / largo
    t = np.linspace(0, 1, 300)
    pts = p0[None, :] + np.outer(t * largo, u) + np.outer(amp * np.sin(2 * np.pi * ciclos * t), n)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, solid_capstyle="round")


def traza(ax, c, ang, k, R, color, lw=2.0, ls="-", r0=0.0):
    """Trayectoria curva (partícula cargada en campo magnético) que sale de c con ángulo `ang`."""
    r = np.linspace(r0, R, 200)
    fi = ang + k * r
    ax.plot(c[0] + r * np.cos(fi), c[1] + r * np.sin(fi), color=color, lw=lw, ls=ls, solid_capstyle="round")
    return c[0] + R * np.cos(ang + k * R), c[1] + R * np.sin(ang + k * R)


# ---------------------------------------------------------------- 1. portada
def portada():
    fig, ax = lienzo(8, 3.6)
    c = (4.0, 1.8)
    for i, (r, col) in enumerate(zip([1.75, 1.4, 1.05, 0.7], [GRIS_CLARO, "#E8E7E1", GRIS_CLARO, "#E8E7E1"])):
        ax.add_patch(Circle(c, r, fc=col, ec="none"))
    rng = np.random.default_rng(4)
    for _ in range(18):
        traza(ax, c, rng.uniform(0, 2 * np.pi), rng.uniform(-0.5, 0.5), rng.uniform(0.7, 1.4), GRIS, lw=0.7)
    for ang, k, col, R in [(0.6, 0.30, ROJO, 1.7), (2.2, -0.22, ROJO, 1.7), (3.7, 0.18, AZUL, 1.75), (5.3, -0.28, AZUL, 1.75)]:
        traza(ax, c, ang, k, R, col, lw=2.2)
    ax.add_patch(Circle(c, 0.05, fc=TINTA, ec="none"))
    guardar(fig, "portada.png")


# ---------------------------------------------------------------- 2. de la materia al quark
def atomo():
    fig, ax = lienzo(8, 2.7)
    xs = [0.8, 2.4, 4.0, 5.6, 7.2]
    y = 1.5
    nombres = [("Materia", "metros"), ("Átomo", "≈10⁻¹⁰ m"), ("Núcleo", "≈10⁻¹⁴ m"),
               ("Protón", "≈10⁻¹⁵ m"), ("Quark", "< 10⁻¹⁸ m")]
    for x, (n, e) in zip(xs, nombres):
        texto(ax, x, 0.62, n, fs=11, weight="bold")
        texto(ax, x, 0.32, e, fs=9.5, color=GRIS)
    for a, b in zip(xs[:-1], xs[1:]):
        flecha(ax, (a + 0.55, y), (b - 0.55, y))
    # materia: red de puntos
    for i in range(5):
        for j in range(5):
            ax.add_patch(Circle((xs[0] - 0.36 + i * 0.18, y - 0.36 + j * 0.18), 0.045, fc=GRIS, ec="none"))
    # átomo
    for ang in (0, 60, 120):
        ax.add_patch(Ellipse((xs[1], y), 0.95, 0.34, angle=ang, fc="none", ec=GRIS, lw=1))
    ax.add_patch(Circle((xs[1], y), 0.07, fc=ROJO, ec="none"))
    for ang, px, py in [(0, 0.47, 0), (60, -0.23, 0.4), (120, -0.24, -0.4)]:
        ax.add_patch(Circle((xs[1] + px, y + py), 0.05, fc=AZUL, ec="none"))
    # núcleo
    rng = np.random.default_rng(1)
    pos = [(0, 0), (0.17, 0.05), (-0.17, 0.06), (0.07, 0.17), (-0.07, -0.17), (0.12, -0.14), (-0.1, 0.15)]
    for i, (dx, dy) in enumerate(pos):
        ax.add_patch(Circle((xs[2] + dx, y + dy), 0.12, fc=ROJO if i % 2 else AZUL_CLARO, ec=TINTA, lw=0.6))
    # protón
    ax.add_patch(Circle((xs[3], y), 0.5, fc="none", ec=GRIS, lw=1.2))
    q = [(0, 0.2, ROJO), (-0.18, -0.12, ROJO), (0.18, -0.12, AZUL)]
    for (a, b, col), (a2, b2, _) in zip(q, q[1:] + q[:1]):
        ax.plot([xs[3] + a, xs[3] + a2], [y + b, y + b2], color=GRIS, lw=1, ls=":")
    for dx, dy, col in q:
        ax.add_patch(Circle((xs[3] + dx, y + dy), 0.09, fc=col, ec="none"))
    # quark
    ax.add_patch(Circle((xs[4], y), 0.13, fc=ROJO, ec="none"))
    guardar(fig, "atomo.png")


# ---------------------------------------------------------------- 3. Modelo Estándar
def modelo_estandar():
    fig, ax = lienzo(8, 5.9)
    lado, sep = 1.0, 0.1
    x0, ytop = 0.45, 4.35
    fermiones = [
        [("u", "up", "2.2 MeV", "+⅔"), ("c", "charm", "1.27 GeV", "+⅔"), ("t", "top", "173 GeV", "+⅔")],
        [("d", "down", "4.7 MeV", "−⅓"), ("s", "strange", "95 MeV", "−⅓"), ("b", "bottom", "4.18 GeV", "−⅓")],
        [("e", "electrón", "0.511 MeV", "−1"), ("μ", "muón", "105.7 MeV", "−1"), ("τ", "tau", "1.777 GeV", "−1")],
        [(r"$\nu_e$", "neutrino e", "≈ 0", "0"), (r"$\nu_\mu$", "neutrino μ", "≈ 0", "0"), (r"$\nu_\tau$", "neutrino τ", "≈ 0", "0")],
    ]
    colores = [AZUL_CLARO, AZUL_CLARO, VERDE_CLARO, VERDE_CLARO]
    bordes = [AZUL, AZUL, VERDE, VERDE]

    def celda(x, y, sim, nombre, masa, carga, fc, ec):
        caja(ax, x, y, lado, lado, fc, ec=ec, lw=1.3)
        texto(ax, x + lado / 2, y + 0.55, sim, fs=19, weight="bold")
        texto(ax, x + lado / 2, y + 0.24, nombre, fs=8)
        texto(ax, x + lado / 2, y + 0.09, masa, fs=7.5, color=GRIS)
        texto(ax, x + lado - 0.07, y + lado - 0.12, carga, fs=7.5, color=GRIS, ha="right")

    for j, g in enumerate(["I", "II", "III"]):
        texto(ax, x0 + j * (lado + sep) + lado / 2, ytop + 1.0 + 0.25, f"Generación {g}", fs=8.5, color=GRIS)
    for i, fila in enumerate(fermiones):
        y = ytop - i * (lado + sep)
        for j, (sim, nom, masa, q) in enumerate(fila):
            celda(x0 + j * (lado + sep), y, sim, nom, masa, q, colores[i], bordes[i])
    # bosones de fuerza
    xb = x0 + 3 * (lado + sep) + 0.3
    bos = [("g", "gluón", "0", "0"), ("γ", "fotón", "0", "0"), ("Z", "bosón Z", "91.2 GeV", "0"), ("W", "bosón W", "80.4 GeV", "±1")]
    for i, (sim, nom, masa, q) in enumerate(bos):
        celda(xb, ytop - i * (lado + sep), sim, nom, masa, q, ROJO_CLARO, ROJO)
    # Higgs
    xh = xb + lado + 0.3
    celda(xh, ytop - 1.5 * (lado + sep), "H", "Higgs", "125 GeV", "0", AMARILLO_CLARO, AMARILLO)
    # etiquetas de grupo
    texto(ax, x0 + 1.5 * (lado + sep) - 0.05, 0.28, "Fermiones: materia", fs=9.5, color=GRIS)
    texto(ax, xb + lado / 2, 0.28, "Bosones: fuerzas", fs=9.5, color=GRIS)
    texto(ax, xh + lado / 2, ytop - 1.5 * (lado + sep) - 0.25, "Campo de Higgs", fs=9.5, color=GRIS)
    texto(ax, 4.1, 0.0, "", fs=1)
    guardar(fig, "modelo_estandar.png")


# ---------------------------------------------------------------- 4. campo de Higgs
def campo_higgs():
    fig = Figure(figsize=(8, 3.5), facecolor=FONDO)
    phi = np.linspace(-1.6, 1.6, 400)
    pot = -2 * phi**2 + phi**4
    for k, (rect, tit) in enumerate([([0.07, 0.15, 0.4, 0.7], "Un campo de Higgs que vale cero"),
                                      ([0.55, 0.15, 0.4, 0.7], "El campo de Higgs real")]):
        ax = fig.add_axes(rect, facecolor=FONDO)
        ax.set_title(tit, fontsize=10.5, color=TINTA)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("valor del campo", fontsize=9, color=GRIS)
        ax.set_ylabel("energía", fontsize=9, color=GRIS)
        if k == 0:
            ax.plot(phi, phi**2, color=GRIS, lw=2)
            ax.plot([0], [0], "o", color=AZUL, ms=11)
            ax.text(0, 1.7, "partículas sin masa", ha="center", fontsize=9, color=GRIS)
        else:
            ax.plot(phi, pot, color=GRIS, lw=2)
            ax.plot([0], [0], "o", color=GRIS_CLARO, ms=11, mec=GRIS)
            ax.plot([1], [-1], "o", color=AZUL, ms=11)
            ax.set_ylim(-1.9, 2.1)
            ax.text(0, 1.25, "el campo 'prefiere' valer algo\ndistinto de cero (≈ 246 GeV)\nen todo el espacio", ha="center",
                    fontsize=8.5, color=TINTA)
            ax.annotate("", xy=(1.3, -0.95), xytext=(0.7, -0.95),
                        arrowprops=dict(arrowstyle="<->", color=ROJO, lw=1.8))
            ax.text(1.0, -1.55, "una vibración = bosón de Higgs", ha="center", fontsize=8.5, color=ROJO)
    guardar(fig, "campo_higgs.png")


# ---------------------------------------------------------------- 5. desintegración H -> ZZ* -> 4l
def higgs_decay():
    fig, ax = lienzo(8, 3.9)
    v0, v1 = (2.2, 1.95), (4.5, 1.95)
    ax.plot([0.5, v0[0]], [1.95, 1.95], color=AMARILLO, lw=2.6, ls=(0, (4, 2)))
    texto(ax, 0.45, 2.25, "H", fs=15, color=AMARILLO, weight="bold")
    texto(ax, 0.55, 1.62, "125 GeV", fs=8.5, color=GRIS, ha="center")
    zu, zd = (4.4, 3.0), (4.4, 0.9)
    onda(ax, v0, zu, color=ROJO)
    onda(ax, v0, zd, color=ROJO, amp=0.05, ciclos=7)
    texto(ax, 3.0, 2.95, "Z  (≈ 91 GeV)", fs=10, color=ROJO)
    texto(ax, 3.0, 1.0, "Z*  (virtual, masa menor)", fs=10, color=ROJO)
    for vtx, sup in [(zu, True), (zd, False)]:
        dy = 0.75
        for sgn, col, lab in [(1, AZUL, "ℓ⁻"), (-1, VERDE, "ℓ⁺")]:
            fin = (7.3, vtx[1] + sgn * dy)
            flecha(ax, vtx, fin, color=col, lw=2.2, ms=14)
            texto(ax, fin[0] + 0.3, fin[1], lab, fs=13, color=col)
    texto(ax, 4.0, 0.2, "ℓ = electrón o muón  →  4μ,  4e  o  2e2μ", fs=10, color=GRIS)
    guardar(fig, "higgs_decay.png")


# ---------------------------------------------------------------- 6. LHC
def lhc():
    fig, ax = lienzo(8, 4.9)
    c, R = (2.75, 2.45), 1.7
    ax.add_patch(Circle(c, R, fc="none", ec=TINTA, lw=2.2))
    ax.add_patch(Circle(c, R - 0.09, fc="none", ec=GRIS, lw=1))
    # puntos 1..8 en sentido horario empezando arriba
    exp = {1: ("ATLAS", AZUL), 2: ("ALICE", VERDE), 5: ("CMS", ROJO), 8: ("LHCb", MORADO)}
    for n in range(8):
        ang = np.pi / 2 - n * np.pi / 4
        px, py = c[0] + R * np.cos(ang), c[1] + R * np.sin(ang)
        punto = n + 1
        if punto in exp:
            nom, col = exp[punto]
            ax.add_patch(Circle((px, py), 0.14, fc=col, ec="none"))
            tx, ty = c[0] + (R + 0.62) * np.cos(ang), c[1] + (R + 0.42) * np.sin(ang)
            texto(ax, tx, ty, nom, fs=12 if nom == "CMS" else 10.5, color=col, weight="bold")
        else:
            ax.add_patch(Circle((px, py), 0.06, fc=GRIS, ec="none"))
    texto(ax, c[0], c[1] + 0.12, "LHC", fs=18, weight="bold")
    texto(ax, c[0], c[1] - 0.22, "≈ 27 km de circunferencia", fs=9.5, color=GRIS)
    flecha(ax, (c[0] - 0.9, c[1] - 0.75), (c[0] + 0.9, c[1] - 0.75), color=AZUL)
    flecha(ax, (c[0] + 0.9, c[1] - 0.95), (c[0] - 0.9, c[1] - 0.95), color=ROJO)
    texto(ax, 5.3, 3.5, "Dos haces de protones", fs=10.5, ha="left", weight="bold")
    texto(ax, 5.3, 3.15, "giran en sentidos opuestos", fs=9.5, ha="left", color=GRIS)
    texto(ax, 5.3, 2.7, "y chocan en cuatro puntos,", fs=9.5, ha="left", color=GRIS)
    texto(ax, 5.3, 2.4, "donde están los grandes", fs=9.5, ha="left", color=GRIS)
    texto(ax, 5.3, 2.1, "detectores.", fs=9.5, ha="left", color=GRIS)
    texto(ax, 5.3, 0.4, "Esquema, no a escala", fs=8.5, ha="left", color=GRIS)
    guardar(fig, "lhc.png")


# ---------------------------------------------------------------- 7. detector CMS (corte transversal)
def cms_detector():
    fig, ax = lienzo(8, 5.4)
    c, esc = (2.75, 2.78), 0.72
    capas = [  # (radio externo, ancho, color)
        (3.35, 1.0, "#F1D8D2"), (2.35, 0.25, "#CFCFCA"), (2.1, 0.85, AMARILLO_CLARO),
        (1.25, 0.55, VERDE_CLARO), (0.7, 0.55, AZUL_CLARO),
    ]
    for r, ancho, col in capas:
        ax.add_patch(Wedge(c, r * esc, 0, 360, width=ancho * esc, fc=col, ec="white", lw=1.2))
    for r in (2.5, 2.9, 3.25):  # cámaras de muones
        ax.add_patch(Circle(c, r * esc, fc="none", ec=ROJO, lw=1.0, alpha=0.6))
    ax.add_patch(Circle(c, 0.06, fc=TINTA, ec="none"))
    # partículas
    k = 0.09
    fin_mu = traza(ax, c, 0.55, k, 3.3 * esc / 1.0 * 1.0, AZUL, lw=2.4, r0=0.0)
    fin_e = traza(ax, c, 2.35, -0.55, 1.2 * esc, VERDE, lw=2.4)
    ax.add_patch(Circle(fin_e, 0.1, fc=VERDE, ec="none", alpha=0.45))
    fin_h = traza(ax, c, 3.65, 0.45, 2.05 * esc, AMARILLO, lw=2.4)
    ax.add_patch(Circle(fin_h, 0.12, fc=AMARILLO, ec="none", alpha=0.45))
    ax.plot([c[0], c[0] + 3.35 * esc * np.cos(1.35)], [c[1], c[1] + 3.35 * esc * np.sin(1.35)],
            color=GRIS, lw=1.6, ls=(0, (1, 2)))
    # leyenda a la derecha
    x0 = 5.55
    texto(ax, x0, 5.0, "Capas del detector", fs=11, ha="left", weight="bold")
    capas_txt = [(AZUL_CLARO, AZUL, "Rastreador de silicio"), (VERDE_CLARO, VERDE, "Calorímetro electromagnético"),
                 (AMARILLO_CLARO, AMARILLO, "Calorímetro hadrónico"), ("#CFCFCA", GRIS, "Imán superconductor (3.8 T)"),
                 ("#F1D8D2", ROJO, "Cámaras de muones")]
    for i, (fc, ec, t) in enumerate(capas_txt):
        y = 4.55 - i * 0.34
        caja(ax, x0, y - 0.1, 0.22, 0.2, fc, ec=ec, lw=1)
        texto(ax, x0 + 0.33, y, t, fs=8.8, ha="left")
    texto(ax, x0, 2.62, "Qué deja cada partícula", fs=11, ha="left", weight="bold")
    parts = [(AZUL, "Muón: atraviesa todo el detector"), (VERDE, "Electrón: se frena en el\ncalorímetro electromagnético"),
             (AMARILLO, "Hadrón: se frena en el\ncalorímetro hadrónico"), (GRIS, "Neutrino: no deja señal")]
    y = 2.25
    for col, t in parts:
        ax.plot([x0, x0 + 0.25], [y, y], color=col, lw=2.6)
        texto(ax, x0 + 0.35, y, t, fs=8.8, ha="left")
        y -= 0.42 if "\n" not in t else 0.62
    texto(ax, 2.75, 0.2, "Corte transversal esquemático de CMS (no a escala)", fs=8.5, color=GRIS)
    guardar(fig, "cms_detector.png")


# ---------------------------------------------------------------- 8. eta y pT
def eta_pt():
    fig = Figure(figsize=(8, 3.6), facecolor=FONDO)
    # panel izquierdo: eta
    ax = fig.add_axes([0.02, 0.05, 0.54, 0.85], facecolor=FONDO)
    ax.set_xlim(-3, 3); ax.set_ylim(-0.5, 3.2); ax.axis("off"); ax.set_aspect("equal")
    ax.set_title("Ángulo respecto al haz: η", fontsize=10.5)
    ax.annotate("", xy=(2.9, 0), xytext=(-2.9, 0), arrowprops=dict(arrowstyle="-", color=TINTA, lw=2))
    ax.text(2.9, -0.28, "haz", fontsize=9, color=GRIS, ha="right")
    ax.add_patch(Circle((0, 0), 0.08, fc=ROJO, ec="none"))
    for eta, col in [(0, AZUL), (1, VERDE), (2, AMARILLO), (3, ROJO)]:
        th = 2 * np.arctan(np.exp(-eta))
        L = 2.9
        ax.plot([0, L * np.cos(th)], [0, L * np.sin(th)], color=col, lw=2)
        ax.text(L * np.cos(th) + 0.08, L * np.sin(th) + 0.02, f"η = {eta}", fontsize=8.5, color=col, ha="left")
    ax.text(0, -0.33, "punto de colisión", fontsize=8.5, color=GRIS, ha="center")
    # panel derecho: pT
    ax2 = fig.add_axes([0.6, 0.05, 0.38, 0.85], facecolor=FONDO)
    ax2.set_xlim(-0.3, 3.2); ax2.set_ylim(-0.5, 3.2); ax2.axis("off"); ax2.set_aspect("equal")
    ax2.set_title("Momento transversal: pT", fontsize=10.5)
    ax2.annotate("", xy=(3.1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-", color=TINTA, lw=2))
    ax2.text(3.1, -0.28, "haz", fontsize=9, color=GRIS, ha="right")
    ax2.annotate("", xy=(2.2, 2.4), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=AZUL, lw=2.4))
    ax2.plot([2.2, 2.2], [0, 2.4], color=ROJO, lw=2, ls="--")
    ax2.plot([0, 2.2], [0, 0], color=GRIS, lw=2, ls="--")
    ax2.text(1.0, 1.45, "p", fontsize=12, color=AZUL, ha="right")
    ax2.text(2.32, 1.2, "pT\n(perpendicular\nal haz)", fontsize=8.5, color=ROJO, ha="left", va="center")
    ax2.text(1.1, -0.28, "pz", fontsize=9, color=GRIS, ha="center")
    guardar(fig, "eta_pt.png")


# ---------------------------------------------------------------- 9. dxy y dz
def impacto():
    fig = Figure(figsize=(8, 3.6), facecolor=FONDO)
    ax = fig.add_axes([0.02, 0.05, 0.47, 0.85], facecolor=FONDO)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2, 2); ax.axis("off"); ax.set_aspect("equal")
    ax.set_title("Vista frontal (plano x–y): dxy", fontsize=10.5)
    ax.add_patch(Circle((0, 0), 0.1, fc=TINTA, ec="none"))
    ax.text(-0.15, -0.3, "punto de\ncolisión", fontsize=8, color=GRIS, ha="right", va="center")
    ax.annotate("", xy=(1.9, 1.3), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=VERDE, lw=2.4))
    ax.text(1.55, 1.55, "leptón del Higgs", fontsize=8.5, color=VERDE, ha="center")
    # trayectoria desplazada: línea paralela a (0.9, -1) a distancia dxy
    d = np.array([1.0, 0.45]); d /= np.hypot(*d)
    n = np.array([-d[1], d[0]])
    p0 = -1.7 * d + 0.75 * n
    ax.annotate("", xy=p0 + 3.5 * d, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=ROJO, lw=2.4))
    pie = np.dot(np.array([0, 0]) - p0, d) * d + p0   # punto de la trayectoria más cercano al vértice
    ax.plot([0, pie[0]], [0, pie[1]], color=TINTA, lw=1.4, ls=":")
    ax.text(pie[0] * 0.5 + 0.12, pie[1] * 0.5 + 0.05, "dxy", fontsize=10, color=TINTA, ha="left")
    ax.text(-1.9, 1.35, "leptón falso", fontsize=8.5, color=ROJO, ha="left")
    ax2 = fig.add_axes([0.54, 0.05, 0.45, 0.85], facecolor=FONDO)
    ax2.set_xlim(-2.4, 2.4); ax2.set_ylim(-1.5, 1.7); ax2.axis("off"); ax2.set_aspect("equal")
    ax2.set_title("Vista lateral (eje z): dz", fontsize=10.5)
    ax2.annotate("", xy=(2.3, 0), xytext=(-2.3, 0), arrowprops=dict(arrowstyle="-", color=TINTA, lw=2))
    ax2.text(2.3, -0.25, "haz (z)", fontsize=8.5, color=GRIS, ha="right")
    ax2.add_patch(Circle((0, 0), 0.1, fc=TINTA, ec="none"))
    ax2.annotate("", xy=(-1.2, 1.4), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="-|>", color=VERDE, lw=2.4))
    ax2.annotate("", xy=(0.15, 1.45), xytext=(1.3, -0.0), arrowprops=dict(arrowstyle="-|>", color=ROJO, lw=2.4))
    ax2.annotate("", xy=(1.3, -0.45), xytext=(0.0, -0.45), arrowprops=dict(arrowstyle="<->", color=TINTA, lw=1.4))
    ax2.text(0.65, -0.78, "dz", fontsize=10, ha="center")
    ax2.text(-1.3, 1.5, "leptón del Higgs", fontsize=8.5, color=VERDE, ha="center")
    ax2.text(0.55, 1.55, "leptón falso", fontsize=8.5, color=ROJO, ha="left")
    guardar(fig, "impacto.png")


# ---------------------------------------------------------------- 10. aislamiento
def aislamiento():
    fig = Figure(figsize=(8, 3.5), facecolor=FONDO)
    rng = np.random.default_rng(7)
    for k, (x, tit, col) in enumerate([(0.03, "Leptón aislado (bueno)", VERDE), (0.52, "Leptón no aislado (falso)", ROJO)]):
        ax = fig.add_axes([x, 0.1, 0.45, 0.78], facecolor=FONDO)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRIS_CLARO)
        ax.set_title(tit, fontsize=10.5, color=col)
        ax.add_patch(Circle((0, 0), 0.6, fc="none", ec=GRIS, lw=1.4, ls="--"))
        ax.text(0.0, 0.67, "cono ΔR = 0.4", fontsize=8.5, color=GRIS, ha="center")
        ax.plot([0], [0], "o", color=col, ms=11)
        ax.text(0.0, -0.2, "leptón", fontsize=8.5, color=col, ha="center")
        fuera = rng.uniform(-0.95, 0.95, (9, 2))
        fuera = fuera[np.hypot(fuera[:, 0], fuera[:, 1]) > 0.72]
        ax.plot(fuera[:, 0], fuera[:, 1], "o", color=GRIS, ms=5, alpha=0.7)
        if k == 1:
            dentro = np.array([[0.25, 0.3], [-0.35, 0.2], [0.3, -0.35], [-0.2, -0.42], [0.45, 0.05]])
            ax.plot(dentro[:, 0], dentro[:, 1], "o", color=GRIS, ms=7)
    fig.text(0.5, 0.01, "Aislamiento relativo = (suma del pT de las partículas dentro del cono) / (pT del leptón)",
             ha="center", fontsize=9, color=TINTA)
    guardar(fig, "aislamiento.png")


# ---------------------------------------------------------------- 11. masa invariante
def masa_invariante():
    fig = Figure(figsize=(8, 3.4), facecolor=FONDO)
    m = np.linspace(70, 180, 40)
    fondo = 60 * np.exp(-(m - 70) / 70) + 4
    pico = 22 * np.exp(-0.5 * ((m - 125) / 3.5) ** 2)
    rng = np.random.default_rng(3)
    for k, (x, tit) in enumerate([(0.07, "Sin Higgs"), (0.55, "Con Higgs")]):
        ax = fig.add_axes([x, 0.2, 0.41, 0.62], facecolor=FONDO)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([70, 100, 125, 150, 180]); ax.set_yticks([])
        ax.tick_params(labelsize=8.5)
        ax.set_xlabel("masa de 4 leptones (GeV)", fontsize=9)
        ax.set_ylabel("eventos", fontsize=9)
        ax.set_title(tit, fontsize=10.5)
        y = fondo + (pico if k else 0)
        ax.bar(m, y, width=2.6, color=AZUL_CLARO if k == 0 else AZUL_CLARO, edgecolor=AZUL, lw=0.8)
        if k:
            ax.bar(m, pico, bottom=fondo, width=2.6, color=ROJO_CLARO, edgecolor=ROJO, lw=0.8)
            ax.annotate("pico del Higgs\n(≈ 125 GeV)", xy=(125, 33), xytext=(143, 48), fontsize=8.5, color=ROJO,
                        arrowprops=dict(arrowstyle="->", color=ROJO))
        ax.set_ylim(0, 70)
    fig.text(0.5, 0.01, "Esquema ilustrativo (no son datos reales)", ha="center", fontsize=8.5, color=GRIS)
    guardar(fig, "masa_invariante.png")


# ---------------------------------------------------------------- 12. flujo Monte Carlo
def montecarlo_flujo():
    fig, ax = lienzo(8, 4.2)
    texto(ax, 0.1, 3.95, "Simulación (Monte Carlo)", fs=10.5, ha="left", color=AZUL, weight="bold")
    texto(ax, 0.1, 1.72, "Datos reales", fs=10.5, ha="left", color=ROJO, weight="bold")
    sim = ["Generador:\nleyes de la física +\nnúmeros aleatorios", "Los quarks se\nconvierten en\npartículas", "Simulación\ndel detector",
           "Reconstrucción"]
    for i, s_ in enumerate(sim):
        x = 0.1 + i * 1.45
        caja(ax, x, 2.45, 1.25, 1.2, AZUL_CLARO, ec=AZUL, s=s_, fs=8.3)
        if i < 3:
            flecha(ax, (x + 1.27, 3.05), (x + 1.43, 3.05), color=AZUL)
    real = ["Colisión real\nen el LHC", "Detector CMS", "Reconstrucción"]
    for i, s_ in enumerate(real):
        x = 0.1 + i * 1.45
        caja(ax, x, 0.55, 1.25, 1.0, ROJO_CLARO, ec=ROJO, s=s_, fs=8.3)
        if i < 2:
            flecha(ax, (x + 1.27, 1.05), (x + 1.43, 1.05), color=ROJO)
    caja(ax, 6.2, 1.25, 1.7, 1.65, AMARILLO_CLARO, ec=AMARILLO, s="Mismas variables:\nse comparan en\nun histograma", fs=9.3)
    flecha(ax, (5.72, 3.05), (6.18, 2.6), color=AZUL)
    flecha(ax, (4.28, 1.05), (6.18, 1.55), color=ROJO)
    guardar(fig, "montecarlo_flujo.png")


# ---------------------------------------------------------------- 13. señal y fondo
def senal_fondo():
    fig = Figure(figsize=(8, 3.8), facecolor=FONDO)
    ax = fig.add_axes([0.09, 0.17, 0.88, 0.72], facecolor=FONDO)
    ax.spines[["top", "right"]].set_visible(False)
    bordes = np.linspace(70, 180, 37)
    c = 0.5 * (bordes[1:] + bordes[:-1])
    zz = 14 * np.exp(-(c - 70) / 45) + 2 + 5 * np.exp(-0.5 * ((c - 91) / 4) ** 2)
    h = 7 * np.exp(-0.5 * ((c - 125) / 3.2) ** 2)
    ax.stairs(zz + h, bordes, color=ROJO, lw=2, label="Señal + fondo (simulación)")
    ax.stairs(zz, bordes, fill=True, fc=AZUL_CLARO, ec=AZUL, lw=1.2, label="Fondo ZZ (simulación)")
    rng = np.random.default_rng(11)
    datos = rng.poisson(zz + h)
    ax.errorbar(c, datos, yerr=np.sqrt(np.maximum(datos, 1)), fmt="o", color=TINTA, ms=3.5, label="Datos")
    ax.set_xlabel("masa de 4 leptones (GeV)", fontsize=9.5)
    ax.set_ylabel("eventos", fontsize=9.5)
    ax.set_ylim(0, 25)
    ax.tick_params(labelsize=9)
    ax.legend(frameon=False, fontsize=9)
    fig.text(0.5, 0.01, "Esquema ilustrativo (no son datos reales)", ha="center", fontsize=8.5, color=GRIS)
    guardar(fig, "senal_fondo.png")


# ---------------------------------------------------------------- 14. pesos
def pesos():
    fig, ax = lienzo(8, 2.9)
    caja(ax, 0.1, 0.6, 2.2, 1.9, AZUL_CLARO, ec=AZUL)
    texto(ax, 1.2, 1.95, "1,499,064", fs=15, weight="bold")
    texto(ax, 1.2, 1.5, "eventos simulados", fs=9.5)
    texto(ax, 1.2, 1.1, "(ZZ → 4μ)", fs=9, color=GRIS)
    caja(ax, 2.9, 0.6, 2.2, 1.9, AMARILLO_CLARO, ec=AMARILLO)
    texto(ax, 4.0, 1.95, "× 0.00082", fs=15, weight="bold")
    texto(ax, 4.0, 1.5, "peso de cada evento", fs=9.5)
    texto(ax, 4.0, 1.05, "= σ · L · k / N", fs=10, color=GRIS)
    caja(ax, 5.7, 0.6, 2.2, 1.9, ROJO_CLARO, ec=ROJO)
    texto(ax, 6.8, 1.95, "≈ 1,236", fs=15, weight="bold")
    texto(ax, 6.8, 1.5, "eventos esperados", fs=9.5)
    texto(ax, 6.8, 1.1, "en los datos reales", fs=9, color=GRIS)
    flecha(ax, (2.35, 1.55), (2.85, 1.55))
    flecha(ax, (5.15, 1.55), (5.65, 1.55))
    texto(ax, 4.0, 0.25, "σ = 0.077 pb,  L = 11,580 pb⁻¹,  k = 1.386", fs=9, color=GRIS)
    guardar(fig, "pesos.png")


# ---------------------------------------------------------------- 15. Poisson
def poisson():
    fig = Figure(figsize=(8, 3.3), facecolor=FONDO)
    ax = fig.add_axes([0.09, 0.2, 0.88, 0.62], facecolor=FONDO)
    ax.spines[["top", "right"]].set_visible(False)
    Ns = [4, 25, 100, 400]
    for i, n in enumerate(Ns):
        ax.errorbar([i], [1.0], yerr=[1 / np.sqrt(n)], fmt="o", color=AZUL, ms=8, capsize=7, lw=2)
        ax.text(i, 1.0 + 1 / np.sqrt(n) + 0.07, f"N = {n}", ha="center", fontsize=9.5, color=TINTA, weight="bold")
        ax.text(i, 1.0 - 1 / np.sqrt(n) - 0.15, f"± {100 / np.sqrt(n):.0f} %", ha="center", fontsize=9.5, color=ROJO)
    ax.axhline(1.0, color=GRIS_CLARO, lw=1, zorder=0)
    ax.set_xticks([]); ax.set_ylim(0.2, 1.9); ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel("valor medido (relativo)", fontsize=9.5)
    ax.tick_params(labelsize=9)
    fig.text(0.5, 0.03, "Con más eventos, la incertidumbre relativa (1/√N) se hace más pequeña",
             ha="center", fontsize=9.5, color=TINTA)
    guardar(fig, "poisson.png")


# ---------------------------------------------------------------- 16. del evento a la fila del CSV
def evento_csv():
    fig, ax = lienzo(8, 3.6)
    c = (1.65, 2.0)
    ax.add_patch(Circle(c, 1.2, fc="#EFEEE9", ec=GRIS_CLARO))
    ax.add_patch(Circle(c, 0.8, fc="#F6F5F1", ec=GRIS_CLARO))
    for ang, k, col, lab in [(0.5, 0.26, AZUL, "1"), (2.0, -0.2, AZUL, "2"), (3.4, 0.3, VERDE, "3"), (5.0, -0.34, VERDE, "4")]:
        fin = traza(ax, c, ang, k, 1.15, col, lw=2.2)
        texto(ax, c[0] + 1.38 * np.cos(ang + k * 1.15), c[1] + 1.38 * np.sin(ang + k * 1.15), lab, fs=10, color=col, weight="bold")
    ax.add_patch(Circle(c, 0.05, fc=TINTA, ec="none"))
    texto(ax, c[0], 0.2, "Un evento con 4 leptones", fs=10)
    flecha(ax, (3.2, 1.95), (3.85, 1.95), lw=2.2)
    texto(ax, 3.52, 2.3, "se convierte\nen una fila", fs=8.5, color=GRIS)
    cols = ["leptón", "tipo", "Q", "pt (GeV)", "eta"]
    filas = [("1", "μ", "−1", "38.2", "0.41"), ("2", "μ", "+1", "24.7", "−0.88"),
             ("3", "e", "−1", "14.3", "1.12"), ("4", "e", "+1", "8.9", "−1.67")]
    xs = [4.15, 4.9, 5.55, 6.25, 7.25]
    ax.add_patch(Rectangle((3.95, 2.75), 3.9, 0.36, fc=GRIS_CLARO, ec="none"))
    for x, cn in zip(xs, cols):
        texto(ax, x, 2.93, cn, fs=9, weight="bold")
    for i, f in enumerate(filas):
        y = 2.4 - i * 0.4
        col = AZUL if f[1] == "μ" else VERDE
        for x, v in zip(xs, f):
            texto(ax, x, y, v, fs=9.5, color=col if v == f[1] else TINTA)
        ax.plot([3.95, 7.85], [y - 0.2, y - 0.2], color=GRIS_CLARO, lw=0.8)
    texto(ax, 5.9, 0.62, "En el CSV real hay una fila por evento y las columnas se repiten\n"
                         "para cada leptón: pt1, eta1, Q1 … pt4, eta4, Q4", fs=8.5, color=GRIS)
    texto(ax, 5.9, 0.12, "Valores de ejemplo ilustrativos, no son un evento real", fs=8, color=GRIS)
    guardar(fig, "evento_csv.png")


# ---------------------------------------------------------------- 17. desintegraciones del Higgs
def higgs_desintegraciones():
    fig = Figure(figsize=(8, 3.9), facecolor=FONDO)
    ax = fig.add_axes([0.2, 0.17, 0.75, 0.74], facecolor=FONDO)
    ax.spines[["top", "right"]].set_visible(False)
    modos = [("b b̄", 58.2), ("W W*", 21.4), ("gluones", 8.2), ("τ⁺ τ⁻", 6.27), ("c c̄", 2.89),
             ("Z Z*", 2.62), ("γ γ", 0.227), ("Z Z* → 4ℓ (e, μ)", 0.0125)]
    nombres = [m[0] for m in modos][::-1]
    vals = [m[1] for m in modos][::-1]
    cols = [ROJO if "4ℓ" in n else AZUL for n in nombres]
    ax.barh(nombres, vals, color=cols, height=0.62)
    ax.set_xscale("log")
    ax.set_xlim(0.005, 120)
    for y, v in enumerate(vals):
        ax.text(v * 1.18, y, f"{v:g} %", va="center", fontsize=8.5, color=TINTA)
    ax.set_xlabel("porcentaje de los Higgs que se desintegran así (escala logarítmica)", fontsize=9)
    ax.tick_params(labelsize=9)
    fig.text(0.5, 0.01, "Valores aproximados para m_H = 125 GeV", ha="center", fontsize=8.5, color=GRIS)
    guardar(fig, "higgs_desintegraciones.png")


# ---------------------------------------------------------------- 18. coordenadas: pT, eta, phi, theta
def coordenadas():
    """Esquema 3D (proyección oblicua) del sistema de coordenadas del detector y de las variables pT, η, φ y θ."""
    fig, ax = lienzo(8.6, 4.8)
    O = np.array([2.75, 2.3])
    ex, ey, ez = np.array([-0.55, -0.42]), np.array([0.0, 1.0]), np.array([1.0, 0.13])

    def P(x, y, z):
        return O + x * ex + y * ey + z * ez

    R, L = 1.3, 2.0
    t = np.linspace(0, 2 * np.pi, 200)
    circ = lambda z, r=R: np.array([P(r * np.cos(a), r * np.sin(a), z) for a in t])
    # contorno del cilindro (la parte "barril" del detector)
    perp = lambda v: -ez[1] * v[0] + ez[0] * v[1]
    pts0 = circ(0)
    ia, ib = int(np.argmax([perp(v - O) for v in pts0])), int(np.argmin([perp(v - O) for v in pts0]))
    for z in (-L, L):
        c = circ(z)
        ax.fill(c[:, 0], c[:, 1], color=ROJO_CLARO, alpha=0.55, ec=ROJO, lw=1.3)
    for i in (ia, ib):
        a = t[i]
        p0, p1 = P(R * np.cos(a), R * np.sin(a), -L), P(R * np.cos(a), R * np.sin(a), L)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=ROJO, lw=1.3)
    ax.plot(*circ(0).T, color=ROJO, lw=0.8, alpha=0.7)

    # ejes
    for v, nom, dx in [(P(0, 0, -L - 0.2), None, None)]:
        pass
    flecha(ax, P(0, 0, -L - 0.25), P(0, 0, L + 0.55), color=TINTA, lw=1.8)
    flecha(ax, O, P(0, 1.85, 0), color=TINTA, lw=1.8)
    flecha(ax, O, P(1.55, 0, 0), color=TINTA, lw=1.8)
    zt, yt, xt = P(0, 0, L + 0.62), P(0, 1.95, 0), P(1.7, 0, 0)
    texto(ax, zt[0] - 0.1, zt[1] + 0.22, "z (haz)", fs=10.5)
    texto(ax, yt[0], yt[1] + 0.08, "y", fs=11)
    texto(ax, xt[0] - 0.05, xt[1] - 0.05, "x", fs=11)
    ax.add_patch(Circle(O, 0.05, fc=ROJO, ec="none", zorder=5))
    texto(ax, O[0] + 0.22, O[1] - 0.2, "punto de colisión", fs=8.5, color=ROJO, ha="left")

    # partícula: pT = 1, phi = 55°, eta = 0.9
    phi, eta = np.radians(55), 0.9
    th = 2 * np.arctan(np.exp(-eta))
    pT = 1.15
    px, py, pz = pT * np.cos(phi), pT * np.sin(phi), pT * np.sinh(eta)
    pp, ppt = P(px, py, pz), P(px, py, 0)
    flecha(ax, O, pp, color=AZUL, lw=2.4, ms=14)
    flecha(ax, O, ppt, color=ROJO, lw=2.2, ms=13)
    ax.plot([ppt[0], pp[0]], [ppt[1], pp[1]], color=GRIS, lw=1.2, ls="--")
    ax.plot([P(px, 0, 0)[0], ppt[0]], [P(px, 0, 0)[1], ppt[1]], color=GRIS, lw=0.9, ls=":")
    ax.plot([P(0, py, 0)[0], ppt[0]], [P(0, py, 0)[1], ppt[1]], color=GRIS, lw=0.9, ls=":")
    texto(ax, pp[0] - 0.12, pp[1] + 0.12, r"$\vec{p}$", fs=15, color=AZUL)
    texto(ax, ppt[0] - 0.42, ppt[1] + 0.12, r"$\vec{p}_T$", fs=14, color=ROJO)
    # arco de phi (en el plano x-y)
    a = np.linspace(0, phi, 40)
    arc = np.array([P(0.55 * np.cos(u), 0.55 * np.sin(u), 0) for u in a])
    ax.plot(arc[:, 0], arc[:, 1], color=TINTA, lw=1.4)
    m = P(0.85 * np.cos(phi / 2), 0.85 * np.sin(phi / 2), 0)
    texto(ax, m[0] - 0.28, m[1] - 0.3, r"$\phi$", fs=13)
    # arco de theta (entre el eje z y p)
    u = np.array([np.cos(phi), np.sin(phi), 0])
    arc = np.array([P(*(1.0 * (np.sin(al) * u + np.cos(al) * np.array([0, 0, 1])))) for al in np.linspace(0, th, 40)])
    ax.plot(arc[:, 0], arc[:, 1], color=TINTA, lw=1.4)
    m = P(*(1.3 * (np.sin(th / 2) * u + np.cos(th / 2) * np.array([0, 0, 1]))))
    texto(ax, m[0] + 0.05, m[1] + 0.12, r"$\theta$", fs=13)

    # etiquetas de pseudorapidez
    e0 = P(0, -(R - 0.2), -0.15)
    texto(ax, e0[0] + 0.15, e0[1], r"$\eta = 0$", fs=10, color=ROJO, ha="left")
    ez_p = P(0, 0, L + 0.1)
    texto(ax, ez_p[0] + 0.1, ez_p[1] - 0.3, r"$\eta \to +\infty$", fs=9.5, color=ROJO)
    ez_m = P(0, 0, -L - 0.1)
    texto(ax, ez_m[0] + 0.05, ez_m[1] - 0.3, r"$\eta \to -\infty$", fs=9.5, color=ROJO)

    # recuadro de relaciones
    x0 = 6.55
    caja(ax, x0, 0.75, 1.95, 3.55, "white", ec=GRIS_CLARO, lw=1.2)
    texto(ax, x0 + 0.97, 4.05, "Relaciones", fs=11, weight="bold")
    filas = [r"$p_T = p\,\sin\theta$", r"$p_x = p_T\cos\phi$", r"$p_y = p_T\sin\phi$",
             r"$p_z = p_T\sinh\eta$", r"$\eta = -\ln\tan(\theta/2)$"]
    for i, f in enumerate(filas):
        texto(ax, x0 + 0.15, 3.6 - i * 0.42, f, fs=10.5, ha="left")
    texto(ax, x0 + 0.2, 1.38, "η grande: casi paralelo\nal haz. η = 0:\nperpendicular al haz.", fs=8, color=GRIS, ha="left")
    texto(ax, 4.3, 0.18, "Esquema de las coordenadas del detector (no a escala)", fs=8.5, color=GRIS)
    guardar(fig, "coordenadas.png")


if __name__ == "__main__":
    for f in (portada, atomo, modelo_estandar, campo_higgs, higgs_decay, lhc, cms_detector, eta_pt, impacto,
              aislamiento, masa_invariante, montecarlo_flujo, senal_fondo, pesos, poisson, evento_csv,
              higgs_desintegraciones, coordenadas):
        f()
