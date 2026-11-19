# El bosón de Higgs con CMS Open Data: tutorial interactivo

**Autor:** Valeria Rocha Garcia. Esta página es el producto de sus prácticas profesionales,
con la asesoría de Isabel Pedraza.

Página web educativa (nivel preparatoria) para buscar el bosón de Higgs en el canal H → ZZ → 4 leptones
con datos abiertos de CMS de 2012.

## Contenido del repositorio

| Ruta | Para qué sirve |
|---|---|
| `index.html` | La página web completa (HTML + JavaScript). Todo el cálculo lo hace el navegador. |
| `web_data/muestra_eventos.csv.gz` | Muestra reducida (una fila por evento) que lee la página. **Es lo único de datos que necesita.** |
| `imagenes/` | Ilustraciones de la página (PNG). |
| `scripts/` | Código del análisis: `02_cutflow.py`, `03_grafica_final.py`, `preparar_muestra.py` (genera la muestra) y `generar_imagenes.py` (genera las ilustraciones). |
| `resultados/` | Gráficas y tablas del análisis (opcional). Si existe `resultados/grafica_final.png`, la página la muestra en "Resultado final". |
| `requirements.txt` | Librerías de Python para los scripts (la página no las necesita). |

Los CSV completos (varios GB) **no están en este repositorio**: se descargan desde Google Drive
(el enlace aparece en la sección "Descarga de datos" de la página).

## Cómo se publica en un servidor

La página es estática: no necesita Python ni nada instalado en el servidor. Basta con copiar
a una carpeta que publique el servidor web (Apache, nginx, etc.), con la misma estructura:

```
index.html
imagenes/
web_data/muestra_eventos.csv.gz
resultados/grafica_final.png   (opcional)
```

## Cómo verla en tu computadora

Abrir `index.html` con doble clic **no funciona**: el navegador no deja leer la muestra de datos
desde archivos locales. Hay que abrir un servidor local en la carpeta del repositorio:

```bash
python -m http.server 8000
```

y entrar a http://localhost:8000

## Regenerar los archivos (opcional)

Solo si se tienen los CSV completos en una carpeta `data/` junto a `scripts/`:

```bash
pip install -r requirements.txt
python scripts/preparar_muestra.py     # crea web_data/muestra_eventos.csv.gz
python scripts/generar_imagenes.py     # crea las imágenes en imagenes/
```

Los valores de los cortes oficiales están al inicio del código JavaScript de `index.html`
(sección "1. LOS CORTES") y son los mismos que en `scripts/02_cutflow.py`.

## Créditos

- Autor: Valeria Rocha Garcia. Producto de sus prácticas profesionales, con la asesoría de Isabel Pedraza.
- Datos: colaboración CMS, [CERN Open Data](https://opendata.cern.ch/record/12360), licencia CC0.
- Análisis basado en: CMS Collaboration, *Observation of a new boson at a mass of 125 GeV*, arXiv:1207.7235.
- Las ilustraciones son esquemas hechos para esta página; no son imágenes reales de CMS ni están a escala.
