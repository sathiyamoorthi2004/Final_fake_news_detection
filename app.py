# app.py
# Robust Fake News Detector (text + optional image OCR)
# - Place news_model.pkl and vectorizer.pkl in same folder to enable predictions.
# - Optional packages: pytesseract, opencv-python (cv2), transformers/torch (image caption)
# - If tesseract is not installed OCR is disabled to avoid crashes on cloud.

import os
import re
import string
import pickle
import difflib
from io import BytesIO
from typing import Tuple, List

from PIL import Image, ImageEnhance, ImageFilter

import streamlit as st
from sklearn.utils.validation import check_is_fitted

# -------------------------
# optional packages (safe imports)
# -------------------------
# cv2 (OpenCV) - used for advanced preprocessing. If missing fall back to PIL-only tweaks.
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

# pytesseract - only enable OCR if tesseract binary exists on PATH
try:
    import pytesseract
    import shutil
    # if system has tesseract executable, enable OCR; otherwise keep disabled to avoid cloud errors
    TESS_BIN = shutil.which("tesseract")
    if TESS_BIN:
        pytesseract.pytesseract.tesseract_cmd = TESS_BIN
        OCR_AVAILABLE = True
    else:
        OCR_AVAILABLE = False
except Exception:
    OCR_AVAILABLE = False

# transformers/torch (optional heavy features). We'll not fail app if missing.
TRY_TRANSFORMERS = False
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    TRY_TRANSFORMERS = True
except Exception:
    TRY_TRANSFORMERS = False

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
# OCR helpers (PIL-only fallback + optional OpenCV variants)
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

# if OpenCV is available, provide a richer set of variants; otherwise use a few PIL variants
def try_ocr_variants(pil_img: Image.Image, tesseract_config: str = "--oem 3 --psm 6") -> Tuple[str, str]:
    """
    Try multiple preprocessing variants. Returns (extracted_text, method_name).
    If nothing found returns ("", "none_found").
    """
    variants = []
    # base (resized if small)
    w, h = pil_img.size
    if w < 1000:
        pil_big = pil_img.resize((1000, int(h * (1000 / w))), Image.LANCZOS)
    else:
        pil_big = pil_img.copy()
    variants.append(("orig_resized", pil_big))

    # PIL enhanced (contrast + sharpen)
    variants.append(("pil_enhanced", pil_enhance_basic(pil_big)))

    # small crop center (sometimes headline sits center)
    try:
        cw, ch = pil_big.size[0]//3, pil_big.size[1]//6
        crop = pil_big.crop((cw, ch, pil_big.size[0]-cw, pil_big.size[1]-ch))
        variants.append(("center_crop", crop))
        variants.append(("center_crop_enhanced", pil_enhance_basic(crop)))
    except Exception:
        pass

    # If OpenCV available, add deskew and adaptive threshold variants
    if CV2_AVAILABLE:
        import numpy as np
        def cv_from_pil(p):
            return cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)
        def pil_from_cv(c):
            return Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))

        base_cv = cv_from_pil(pil_big)
        # deskew
        gray = cv2.cvtColor(base_cv, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)[1])
        if coords is not None and len(coords) > 10:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (hcv, wcv) = base_cv.shape[:2]
            M = cv2.getRotationMatrix2D((wcv // 2, hcv // 2), angle, 1.0)
            rotated = cv2.warpAffine(base_cv, M, (wcv, hcv), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            variants.append(("cv_deskew", pil_from_cv(rotated)))
            # adaptive threshold variant
            adap = cv2.adaptiveThreshold(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 25, 9)
            variants.append(("cv_deskew_adapt", pil_from_cv(cv2.cvtColor(adap, cv2.COLOR_GRAY2BGR))))
        # morphological close/denoise
        den = cv2.medianBlur(gray, 3)
        variants.append(("cv_median", pil_from_cv(cv2.cvtColor(den, cv2.COLOR_GRAY2BGR))))

    # run OCR on variants in order
    for name, pimg in variants:
        try:
            if OCR_AVAILABLE:
                text = pytesseract.image_to_string(pimg, config=tesseract_config)
                text_norm = normalize_ocr_text(text)
                if text_norm and len(text_norm) > 2:
                    return text_norm, name
            else:
                # OCR not available - skip
                pass
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
    .result-box {border-radius:8px; padding:14px; border:2px solid #e6e6e6; background:#fbfffb;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>📰 Fake News Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Text prediction (TF-IDF → RandomForest). Optional image OCR when Tesseract is installed locally.</div>", unsafe_allow_html=True)
st.write("")

# artifact notice
if model is None or vectorizer is None:
    st.warning("Model or vectorizer missing — place `news_model.pkl` and `vectorizer.pkl` in the app folder to enable predictions.")

# session state keys
if 'user_text' not in st.session_state:
    st.session_state['user_text'] = ""
if 'last_message' not in st.session_state:
    st.session_state['last_message'] = None

# clear callback (works via experimental_rerun to avoid inline set-after-initialization errors)
def clear_text_callback():
    st.session_state['user_text'] = ""
    st.session_state['last_message'] = "Text cleared."
    st.experimental_rerun()

# Layout
tab_text, tab_image = st.tabs(["📝 Text", "🖼 Image (OCR)"])

with tab_text:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Enter News Text")
    # place text area inside a placeholder so rerun clears it visually
    user_text = st.text_area("Paste news content here:", value=st.session_state.get('user_text', ""), key="user_text_area", height=160)

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
                        st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span> ⚠️ (low confidence)", unsafe_allow_html=True)
                    else:
                        st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
                    if prob is not None:
                        st.write(f"Confidence: {prob:.2f}")
                    st.write("Cleaned Text:")
                    st.write(out["cleaned"])
    with col2:
        if st.button("Clear Text"):
            # set state and rerun to clear widget properly
            st.session_state['user_text'] = ""
            st.session_state['last_message'] = "Text cleared."
            st.experimental_rerun()

    if st.session_state.get('last_message'):
        st.success(st.session_state.pop('last_message'))

    st.markdown("</div>", unsafe_allow_html=True)

with tab_image:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Upload news image (OCR)")
    if not OCR_AVAILABLE:
        st.info("OCR disabled: Tesseract not found in PATH. Install Tesseract locally to enable image OCR.")
    uploaded = st.file_uploader("Upload JPG/PNG (crop to headline for best results)", type=['jpg','jpeg','png'])
    if uploaded:
        try:
            pil_img = Image.open(BytesIO(uploaded.read())).convert("RGB")
            st.image(pil_img, caption="Uploaded image (preview)", use_column_width=True)

            if OCR_AVAILABLE:
                if st.button("Extract & Predict from image"):
                    with st.spinner("Running OCR variants (may take a moment)..."):
                        text_found, method = try_ocr_variants(pil_img)
                        if not text_found:
                            st.warning("No text was extracted after multiple preprocessing attempts. Try cropping headline and re-uploading.")
                        else:
                            st.success(f"Text found using variant: {method}")
                            st.write(text_found[:2000])
                            # attempt prediction (repair tokens fuzzily if vectorizer available)
                            repaired_text = text_found
                            if vectorizer is not None:
                                # fuzzy repair: map tokens to closest vocab words if helpful
                                try:
                                    vocab_list = list(vectorizer.get_feature_names_out())
                                except Exception:
                                    vocab_list = list(getattr(vectorizer, "vocabulary_", {}).keys())
                                # simple fuzzy mapping for short OCR tokens
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
                                # offer force predict (if user understands risk)
                                if st.button("Force predict anyway (use with caution)"):
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
                                    st.markdown(f"### Prediction: <span style='color:{color}'>{lbl}</span> ⚠️ (Low confidence)", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"### Prediction: <span style='color:{color}'>{lbl}</span>", unsafe_allow_html=True)
                                if prob is not None:
                                    st.write(f"Confidence: {prob:.2f}")
            else:
                st.info("OCR is not available on this system. Install Tesseract and ensure `tesseract` is on PATH to enable image extraction.")
        except Exception as e:
            st.error(f"Could not open/process image: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# footer
st.markdown("---")
st.write("Tip: Crop the headline area before upload for best OCR results. If model gives low confidence, prediction may be unreliable.")
