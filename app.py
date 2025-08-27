
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io, zipfile, os, base64
import svgwrite

try:
    import cairosvg
except Exception:
    cairosvg = None

try:
    from skimage import measure, morphology, color as skcolor, util
except Exception:
    measure = None

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

st.set_page_config(page_title="Cricut-style Color Slicer", layout="wide")
st.title("Cricut-style Color Slicer (Streamlit)")
st.caption("Slice an image into per-color SVG layers (like Cricut). Also includes grayscale layering.")

with st.expander("How it works / Notes"):
    st.markdown("""
**Color mode**:
- Convert your image to Lab color space and run K-Means to find K dominant colors.
- You can toggle which colors to export, adjust tolerance to merge close swatches, and apply kerf offset.
- Each selected color becomes its own SVG layer (trace via contours).

**Grayscale mode**:
- Same as before: threshold into N layers, trace to SVGs.

> Accuracy depends on vectorization (contour tracing). Inspect paths before production.
""")

# Sidebar
st.sidebar.header("Upload")
file = st.sidebar.file_uploader("Upload PNG/JPG/SVG", type=["png","jpg","jpeg","svg"])
scale = st.sidebar.slider("Import scale (x)", 0.2, 4.0, 1.0, 0.1)

# --------------------- Utilities ---------------------
def rasterize_svg_to_png(svg_bytes, px=1024):
    if cairosvg is None:
        raise RuntimeError("cairosvg is not installed. Add to requirements.txt to enable SVG import.")
    png_bytes = cairosvg.svg2png(bytestring=svg_bytes, output_width=px, output_height=px)
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    return img

def to_rgba(img):
    return img.convert("RGBA")

def alpha_composite_on_white(img):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (255,255,255,255))
    return Image.alpha_composite(bg, img).convert("RGB")

def clean_mask(mask, smooth=1, min_area=50):
    if measure is None:
        return mask
    m = mask.copy().astype(bool)
    if smooth >= 1:
        m = morphology.binary_opening(m, morphology.disk(smooth))
        m = morphology.binary_closing(m, morphology.disk(smooth))
    if min_area > 0:
        m = morphology.remove_small_objects(m, min_size=int(min_area))
    return m.astype(np.uint8)

def mask_to_paths(mask):
    if measure is None:
        return []
    contours = measure.find_contours(mask, level=0.5)
    paths = []
    for c in contours:
        if len(c) < 3:
            continue
        d = "M " + " L ".join([f"{p[1]:.2f},{p[0]:.2f}" for p in c]) + " Z"
        paths.append(d)
    return paths

def svg_from_paths(paths, w, h, scale=1.0, fill="#000000"):
    svg = svgwrite.Drawing(size=(f"{w*scale}px", f"{h*scale}px"))
    for d in paths:
        svg.add(svg.path(d=d, fill=fill, stroke="none", transform=f"scale({scale})"))
    return svg.tostring()

def offset_mask(mask, radius_px=0):
    if measure is None or radius_px == 0:
        return mask
    selem = morphology.disk(abs(int(radius_px)))
    if radius_px > 0:
        m = morphology.binary_dilation(mask.astype(bool), selem)
    else:
        m = morphology.binary_erosion(mask.astype(bool), selem)
    return m.astype(np.uint8)

# --------------------- UI Tabs ---------------------
tab_color, tab_gray = st.tabs(["Color slicing", "Grayscale layering"])

# Load image
img = None
if file is not None:
    ext = os.path.splitext(file.name.lower())[-1]
    if ext == ".svg":
        try:
            img = rasterize_svg_to_png(file.read(), px=int(1024*scale))
        except Exception as e:
            st.error(f"SVG rasterization failed: {e}")
    else:
        img = Image.open(file).convert("RGBA")
        if scale != 1.0:
            w,h = img.size
            img = img.resize((int(w*scale), int(h*scale)))
    st.image(img, caption="Input", use_container_width=True)

# --------------------- Color Mode ---------------------
with tab_color:
    st.subheader("Color slicing (Cricut-style)")

    k = st.slider("Number of colors (K-Means)", 2, 12, 5, 1)
    tolerance = st.slider("Color tolerance (Delta-E merge)", 0.0, 40.0, 8.0, 0.5)
    smooth = st.slider("Smooth/denoise (px)", 0, 6, 1, 1)
    min_region = st.slider("Minimum region area (px^2)", 0, 800, 60, 10)
    kerf = st.slider("Kerf/offset (px)", -6, 6, 0, 1, help="Positive = expand; Negative = shrink")
    export_scale = st.slider("Export scale (x)", 0.5, 4.0, 1.0, 0.1)

    if img is None:
        st.info("Upload an image to begin.")
    else:
        if KMeans is None:
            st.warning("scikit-learn not installed. Add it in requirements.txt to enable color clustering.")
        else:
            # Prepare data in Lab space for perceptual clustering
            rgb = alpha_composite_on_white(img)
            rgb_np = np.array(rgb)
            lab = skcolor.rgb2lab(rgb_np/255.0)
            H,W,_ = lab.shape
            X = lab.reshape(-1, 3)

            km = KMeans(n_clusters=int(k), n_init=4, random_state=0)
            labels = km.fit_predict(X)
            centers = km.cluster_centers_
            label_img = labels.reshape(H, W)

            # Palette preview
            centers_rgb = (skcolor.lab2rgb(centers.reshape(-1,1,1,3)).reshape(-1,3)*255).astype(np.uint8)
            st.write("Detected palette:")
            cols = st.columns(min(k,6))
            selected = []
            for i in range(k):
                chip = Image.new("RGB", (80, 40), tuple(centers_rgb[i].tolist()))
                with cols[i % len(cols)]:
                    st.image(chip, caption=f"Color {i+1}", width=100)
                    sel = st.checkbox(f"Export Color {i+1}", value=True, key=f"sel_{i}")
                    selected.append(sel)

            if st.button("Slice & Export ZIP"):
                mem = io.BytesIO()
                with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
                    combined = svgwrite.Drawing(size=(f"{W*export_scale}px", f"{H*export_scale}px"))
                    # optional: compute deltaE for merging
                    try:
                        from skimage.color import deltaE_cie76
                        have_delta = True
                    except Exception:
                        have_delta = False

                    for i in range(k):
                        if not selected[i]:
                            continue
                        mask = (label_img == i).astype(np.uint8)

                        if tolerance > 0 and have_delta:
                            dE = np.linalg.norm(centers - centers[i], axis=1)
                            similar_idxs = np.where(dE <= tolerance)[0]
                            for j in similar_idxs:
                                mask = np.maximum(mask, (label_img == j).astype(np.uint8))

                        mask = clean_mask(mask, smooth=smooth, min_area=min_region)
                        if kerf != 0:
                            mask = offset_mask(mask, kerf)

                        paths = mask_to_paths(mask)
                        fill_hex = "#{:02x}{:02x}{:02x}".format(*centers_rgb[i])
                        svg_txt = svg_from_paths(paths, W, H, scale=export_scale, fill=fill_hex)
                        z.writestr(f"color_{i+1:02d}.svg", svg_txt)

                        for d in paths:
                            combined.add(combined.path(d=d, fill=fill_hex, stroke="none", transform=f"scale({export_scale})"))

                    z.writestr("combined_colors.svg", combined.tostring())
                st.download_button("Download color layers (ZIP)", data=mem.getvalue(), file_name="color_layers.zip", mime="application/zip")

# --------------------- Grayscale Mode ---------------------
with tab_gray:
    st.subheader("Grayscale layering")
    n_layers = st.slider("Number of layers", 2, 12, 6, 1, key="n_gray")
    invert = st.checkbox("Invert (light<->dark)", value=False, key="inv_gray")
    smooth_g = st.slider("Smooth/denoise (px)", 0, 6, 1, 1, key="sm_gray")
    min_region_g = st.slider("Minimum region area (px^2)", 0, 800, 60, 10, key="mr_gray")
    kerf_g = st.slider("Kerf/offset (px)", -6, 6, 0, 1, key="kf_gray")
    export_scale_g = st.slider("Export scale (x)", 0.5, 4.0, 1.0, 0.1, key="sc_gray")

    if img is None:
        st.info("Upload an image to begin.")
    else:
        gray = ImageOps.grayscale(img)
        g = np.array(gray).astype(np.float32) / 255.0
        if invert:
            g = 1.0 - g
        thresholds = np.linspace(0,1,n_layers+1)[1:]
        masks = [(g >= t).astype(np.uint8) for t in thresholds]

        H,W = gray.size[1], gray.size[0]
        if st.button("Export grayscale layers ZIP"):
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
                combined = svgwrite.Drawing(size=(f"{W*export_scale_g}px", f"{H*export_scale_g}px"))
                for i, m in enumerate(masks, start=1):
                    m = clean_mask(m, smooth=smooth_g, min_area=min_region_g)
                    if kerf_g != 0:
                        m = offset_mask(m, kerf_g)
                    paths = mask_to_paths(m)
                    svg_txt = svg_from_paths(paths, W, H, scale=export_scale_g, fill="#000000")
                    z.writestr(f"layer_{i:02d}.svg", svg_txt)
                    for d in paths:
                        combined.add(combined.path(d=d, fill="#666666", stroke='none', transform=f"scale({export_scale_g})"))
                z.writestr("combined_layers.svg", combined.tostring())
            st.download_button("Download grayscale layers (ZIP)", data=mem.getvalue(), file_name="grayscale_layers.zip", mime="application/zip")
