# app.py
# Fake News Detector (text + optional image OCR)
# - Uses EasyOCR on cloud (if installed) or pytesseract locally as fallback.
# - Put news_model.pkl and vectorizer.pkl in same folder to enable predictions.
# - Requirements (add to requirements.txt): streamlit, pillow, scikit-learn, easyocr, opencv-python-headless
#   (torch will be installed as a dependency of easyocr).

import os
import re
import string
import pickle
import difflib
from io import BytesIO
from typing import Tuple

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

import streamlit as st
from sklearn.utils.validation import check_is_fitted

# -------------------------
# Optional packages (safe imports)
# -------------------------
# cv2 (OpenCV) - used for advanced preprocessing. If missing fall back to PIL-only tweaks.
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

# Try EasyOCR first (best for cloud). If not installed, we'll try pytesseract (local).
EASYOCR_AVAILABLE = False
PYTESSERACT_AVAILABLE = False
try:
    import easyocr
    # create reader lazily later (so startup is faster in environments where not used)
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    import shutil
    # enable pytesseract only if tesseract binary is found (local)
    TESS_BIN = shutil.which("tesseract")
    if TESS_BIN:
        pytesseract.pytesseract.tesseract_cmd = TESS_BIN
        PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

# Decide final OCR availability: prefer EasyOCR if available, otherwise pytesseract (local)
OCR_BACKEND = None
if EASYOCR_AVAILABLE:
    OCR_BACKEND = "easyocr"
elif PYTESSERACT_AVAILABLE:
    OCR_BACKEND = "pytesseract"
else:
    OCR_BACKEND = None

# -------------------------
# Files & config
# -------------------------
MODEL_FILE = "news_model.pkl"
VECT_FILE = "vectorizer.pkl"
CSV_FILE = "News_dataset.csv"  # optional preview

# -------------------------
# small helpers
# -------------------------
def clean_text_simple(text: str) -> str:
    text = str(text).lower()
    text = text.replace("can't", "can not").replace("won't", "will not")
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_artifacts():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECT_FILE):
        try:
            model = pickle.load(open(MODEL_FILE, "rb"))
            vectorizer = pickle.load(open(VECT_FILE, "rb"))
            try:
                check_is_fitted(vectorizer)
            except Exception:
                pass
            return model, vectorizer
        except Exception as e:
            st.error(f"Error loading artifacts: {e}")
            return None, None
    return None, None

model, vectorizer = load_artifacts()

# -------------------------
# prediction wrapper with basic domain checks
# -------------------------
def predict_with_checks(raw_text: str) -> dict:
    cleaned = clean_text_simple(raw_text)
    tokens = [t for t in cleaned.split() if any(ch.isalpha() for ch in t)]
    token_count = len(tokens)

    min_tokens = 3
    min_vocab_fraction = 0.30

    if token_count < min_tokens:
        return {"ok": False, "reason": f"Input too short (found {token_count} words).", "cleaned": cleaned}

    if model is None or vectorizer is None:
        return {"ok": False, "reason": "Model not available (place news_model.pkl & vectorizer.pkl in app folder).", "cleaned": cleaned}

    try:
        vocab_list = list(vectorizer.get_feature_names_out())
    except Exception:
        vocab_list = list(getattr(vectorizer, "vocabulary_", {}).keys())
    vocab_set = set(vocab_list)

    if not vocab_list:
        vocab_fraction = 1.0
        in_vocab = tokens
    else:
        in_vocab = [t for t in tokens if t in vocab_set]
        vocab_fraction = len(in_vocab) / float(token_count) if token_count > 0 else 0.0

    if vocab_fraction < min_vocab_fraction:
        return {
            "ok": False,
            "reason": (
                "Input looks unfamiliar. "
                f"Only {len(in_vocab)}/{token_count} words ({vocab_fraction:.2f}) are in model vocabulary; prediction would be unreliable."
            ),
            "cleaned": cleaned
        }

    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(max(model.predict_proba(vec)[0]))

    low_conf = (prob is not None and prob < 0.6)

    return {"ok": True, "cleaned": cleaned, "prediction": "REAL" if int(pred) == 1 else "FAKE", "prob": prob, "low_confidence": low_conf}

# -------------------------
# OCR helpers
# -------------------------
def normalize_ocr_text(s: str) -> str:
    s = s or ""
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r'[^A-Za-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

def pil_enhance_basic(pil_img: Image.Image) -> Image.Image:
    # contrast, sharpen, median filter
    img = pil_img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img

# EasyOCR extraction helper
def easyocr_extract_text(reader, pil_img: Image.Image) -> str:
    # convert to RGB numpy
    arr = np.array(pil_img.convert("RGB"))
    # easyocr returns list of (bbox, text, confidence)
    try:
        results = reader.readtext(arr, detail=1)
        texts = [r[1] for r in results if len(r) >= 2 and r[1].strip() != ""]
        joined = " ".join(texts)
        return normalize_ocr_text(joined)
    except Exception:
        return ""

# pytesseract extraction helper
def pytess_extract_text(pil_img: Image.Image, config: str = "--oem 3 --psm 6") -> str:
    try:
        txt = pytesseract.image_to_string(pil_img, config=config)
        return normalize_ocr_text(txt)
    except Exception:
        return ""

# Try a few preprocessing variants and attempt OCR
def try_ocr_variants(pil_img: Image.Image) -> Tuple[str, str]:
    """Return (extracted_text, method_name). empty text if none found."""
    # resize if small
    w, h = pil_img.size
    if w < 1000:
        big = pil_img.resize((1000, int(h * (1000 / w))), Image.LANCZOS)
    else:
        big = pil_img.copy()

    variants = [("orig_resized", big), ("enhanced", pil_enhance_basic(big))]

    # center crop variant
    try:
        cw, ch = big.size[0] // 4, big.size[1] // 6
        crop = big.crop((cw, ch, big.size[0] - cw, big.size[1] - ch))
        variants.append(("center_crop", crop))
        variants.append(("center_crop_enh", pil_enhance_basic(crop)))
    except Exception:
        pass

    # OpenCV deskew / adapt if available (optional)
    if CV2_AVAILABLE:
        try:
            import cv2
            import numpy as np
            bgr = cv2.cvtColor(np.array(big), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            coords = cv2.findNonZero(cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)[1])
            if coords is not None and len(coords) > 10:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                (hcv, wcv) = bgr.shape[:2]
                M = cv2.getRotationMatrix2D((wcv // 2, hcv // 2), angle, 1.0)
                rotated = cv2.warpAffine(bgr, M, (wcv, hcv), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                variants.append(("cv_deskew", Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))))
        except Exception:
            pass

    # If EasyOCR available, use it first across variants
    if EASYOCR_AVAILABLE:
        # create reader once and reuse
        try:
            # cache reader in module-level variable so we don't recreate often
            global _easyocr_reader
            if "_easyocr_reader" not in globals():
                # create english-only CPU reader to keep resources small
                _easyocr_reader = easyocr.Reader(["en"], gpu=False)
            reader = _easyocr_reader
        except Exception:
            reader = None

        if reader is not None:
            for name, vimg in variants:
                try:
                    txt = easyocr_extract_text(reader, vimg)
                    if txt and len(txt) > 2:
                        return txt, f"easyocr_{name}"
                except Exception:
                    continue

    # Fallback to pytesseract if available
    if PYTESSERACT_AVAILABLE:
        for name, vimg in variants:
            try:
                txt = pytess_extract_text(vimg)
                if txt and len(txt) > 2:
                    return txt, f"pytess_{name}"
            except Exception:
                continue

    return "", "none_found"

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.markdown("""
    <style>
    .title {font-size:36px; font-weight:800; text-align:center; margin-bottom:6px; color: #0b6e4f;}
    .subtitle {text-align:center; color:#666; margin-top:-6px;}
    .panel {padding:16px; background:#fff; border-radius:8px; box-shadow: 0 6px 18px rgba(0,0,0,0.04);}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>📰 Fake News Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Text prediction (TF-IDF → RandomForest). Optional image OCR when EasyOCR or Tesseract is available.</div>", unsafe_allow_html=True)
st.write("")

# artifact notice
if model is None or vectorizer is None:
    st.warning("Model or vectorizer missing — place `news_model.pkl` and `vectorizer.pkl` in the app folder to enable predictions.")

# show which OCR backend (if any) is enabled
if OCR_BACKEND == "easyocr":
    st.success("OCR backend: EasyOCR (enabled).")
elif OCR_BACKEND == "pytesseract":
    st.info("OCR backend: pytesseract (local).")
else:
    st.info("OCR not available on this system. To enable: add EasyOCR to requirements (for cloud) or install Tesseract locally (for local).")

# session state keys
if 'user_text' not in st.session_state:
    st.session_state['user_text'] = ""
if 'last_message' not in st.session_state:
    st.session_state['last_message'] = None

# Layout
tab_text, tab_image = st.tabs(["📝 Text", "🖼 Image (OCR)"])


with tab_text:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Enter News Text")
    user_text = st.text_area(
        "Paste news content here:",
        value=st.session_state.get('user_text', ""),
        key="user_text_area",
        height=160
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Predict Text"):
            txt = st.session_state.get('user_text_area', "").strip()
            if txt == "":
                st.warning("Please enter some text first.")
            else:
                out = predict_with_checks(txt)
                if not out.get("ok"):
                    st.warning(out.get("reason"))
                    st.write("Cleaned text (info):")
                    st.write(out.get("cleaned", ""))
                else:
                    label = out["prediction"]
                    prob = out.get("prob")
                    color = "#1b8b4a" if label == "REAL" else "#b02a2a"
                    low = out.get("low_confidence", False)
                    if low:
                        st.markdown(
                            f"### Prediction: <span style='color:{color}'>{label}</span> ⚠️ (low confidence)",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"### Prediction: <span style='color:{color}'>{label}</span>",
                            unsafe_allow_html=True
                        )
                    if prob is not None:
                        st.write(f"Confidence: {prob:.2f}")
                    st.write("Cleaned Text:")
                    st.write(out["cleaned"])
    with col2:
        if st.button("Clear Text"):
            st.session_state['user_text'] = ""
            st.session_state['last_message'] = "Text cleared."
            st.rerun()

    if st.session_state.get('last_message'):
        st.success(st.session_state.pop('last_message'))

    st.markdown("</div>", unsafe_allow_html=True)

with tab_image:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Upload news image (OCR)")
    if OCR_BACKEND is None:
        st.info("OCR disabled: EasyOCR or Tesseract not available. To enable on cloud add EasyOCR to requirements.txt; to enable locally install Tesseract.")
    uploaded = st.file_uploader("Upload JPG/PNG (crop to headline for best results)", type=['jpg', 'jpeg', 'png'])
    if uploaded:
        try:
            pil_img = Image.open(BytesIO(uploaded.read())).convert("RGB")
            st.image(pil_img, caption="Uploaded image (preview)", use_column_width=True)

            if OCR_BACKEND is not None:
                if st.button("Extract & Predict from image"):
                    with st.spinner("Running OCR (may take a moment)..."):
                        text_found, method = try_ocr_variants(pil_img)
                        if not text_found:
                            st.warning("No text was extracted after multiple preprocessing attempts. Try cropping headline and re-uploading.")
                        else:
                            st.success(f"Text found using: {method}")
                            st.write(text_found[:2000])
                            # attempt fuzzy repair to model vocab if vectorizer exists
                            repaired_text = text_found
                            if vectorizer is not None:
                                try:
                                    vocab_list = list(vectorizer.get_feature_names_out())
                                except Exception:
                                    vocab_list = list(getattr(vectorizer, "vocabulary_", {}).keys())

                                def fuzzy_repair(s):
                                    toks = s.split()
                                    out_toks = []
                                    for t in toks:
                                        if t in vocab_list:
                                            out_toks.append(t)
                                            continue
                                        matches = difflib.get_close_matches(t, vocab_list, n=1, cutoff=0.78)
                                        out_toks.append(matches[0] if matches else t)
                                    return " ".join(out_toks)

                                repaired_text = fuzzy_repair(text_found)
                                st.info("Attempted fuzzy repair to model vocabulary (may improve prediction).")
                            # predict
                            out = predict_with_checks(repaired_text)
                            if not out.get("ok"):
                                st.warning(out.get("reason"))
                                if st.button("Force predict anyway (use caution)"):
                                    if vectorizer is None or model is None:
                                        st.error("Model artifacts missing; cannot force predict.")
                                    else:
                                        vec = vectorizer.transform([repaired_text])
                                        pred = model.predict(vec)[0]
                                        prob = max(model.predict_proba(vec)[0]) if hasattr(model, "predict_proba") else None
                                        lbl = "REAL" if int(pred) == 1 else "FAKE"
                                        st.markdown(f"### Forced prediction: **{lbl}**")
                                        if prob is not None:
                                            st.write(f"Confidence: {prob:.2f}")
                            else:
                                lbl = out["prediction"]
                                prob = out.get("prob")
                                color = "#1b8b4a" if lbl == "REAL" else "#b02a2a"
                                if out.get("low_confidence"):
                                    st.markdown(
                                        f"### Prediction: <span style='color:{color}'>{lbl}</span> ⚠️ (Low confidence)",
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f"### Prediction: <span style='color:{color}'>{lbl}</span>",
                                        unsafe_allow_html=True
                                    )
                                if prob is not None:
                                    st.write(f"Confidence: {prob:.2f}")
            else:
                st.info("OCR backend not available. Add EasyOCR to requirements.txt (for cloud) or install Tesseract locally (for local OCR).")
        except Exception as e:
            st.error(f"Could not open/process image: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# footer
st.markdown("---")
st.write("Tip: Crop the headline area before upload for best OCR results. If model gives low confidence, prediction may be unreliable.")
