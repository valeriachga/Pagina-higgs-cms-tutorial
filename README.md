# El bosón de Higgs con CMS Open Data: tutorial interactivo

## Contenido del repositorio

| Ruta | Para qué sirve |
|---|---|
| `app.py` | La página web (archivo principal). |
| `requirements.txt` | Librerías de Python necesarias. |
| `web_data/muestra_eventos.csv.gz` | Muestra reducida (una fila por evento) que lee la página. **Es lo único de datos que necesita.** |
| `imagenes/` | Ilustraciones de la página (PNG). |
| `scripts/` | Código del análisis: `02_cutflow.py`, `03_grafica_final.py`, `preparar_muestra.py` (genera la muestra) y `generar_imagenes.py` (genera las ilustraciones). |
| `resultados/` | Gráficas y tablas del análisis (opcional). |

Los CSV completos (varios GB) **no están en este repositorio**: se descargan desde Google Drive
(el enlace aparece en la sección "Descarga de datos" de la página).

## Cómo ejecutarla (servidor o computadora)

Requiere **Python 3.10 o superior**.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Por defecto abre en el puerto 8501. Para un servidor:

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
```

Notas para el servidor:
- Necesita un servidor que **ejecute Python** (no sirve un hosting estático de solo HTML).
- Si se publica detrás de un proxy (nginx, Apache), hay que habilitar **WebSockets**, porque Streamlit los usa.
- Para dejarla corriendo de forma permanente se puede usar `systemd`, `supervisor` o Docker.
- Memoria: con la muestra reducida basta con unos 500 MB de RAM.

## Regenerar los archivos (opcional)

Solo si se tienen los CSV completos en una carpeta `data/` junto a `scripts/`:

```bash
python scripts/preparar_muestra.py     # crea web_data/muestra_eventos.csv.gz
python scripts/generar_imagenes.py     # crea las imágenes en imagenes/
```

## Créditos

- Datos: colaboración CMS, [CERN Open Data](https://opendata.cern.ch/record/12360), licencia CC0.
- Análisis basado en: CMS Collaboration, *Observation of a new boson at a mass of 125 GeV*, arXiv:1207.7235.
- Las ilustraciones son esquemas hechos para esta página; no son imágenes reales de CMS ni están a escala.
