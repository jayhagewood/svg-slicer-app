
# Cricut-style Color Slicer (Streamlit)

- Color mode: K-Means in Lab space -> per-color masks -> contour -> SVG per color (+ combined SVG). Controls for tolerance, smoothing, min region, kerf offset, scale.
- Grayscale mode: Threshold N layers -> SVGs.

## Run
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
