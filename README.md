# Next-Word Predictor — by Ansh Raj

Demo Streamlit app that predicts the next word(s) using a small LSTM model.
This **fast demo version** is optimized for quick startup and is ideal for resumes or live demos.

## 🚀 Features
- Simple Streamlit UI with seed input and next-word prediction.
- Small, fast model trained at startup on a tiny sample corpus (demo-ready).
- Ready to deploy on Streamlit Community Cloud (free, provides HTTPS).

## 📁 Files
- `next_word_predictor_app.py` — Streamlit app (fast demo)
- `requirements.txt` — dependencies
- `.streamlit/config.toml` — Streamlit config
- `Procfile` — optional
- `README.md`, `LICENSE`, `.gitignore`

## 📦 Quick Start (local)
1. Create virtual env and install deps:
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS / Linux
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```
2. Run app locally:
   ```bash
   streamlit run next_word_predictor_app.py
   ```

## ☁️ Deploy on Streamlit Community Cloud (recommended)
1. Create a GitHub repo and push these files.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, select your repo, branch `main`, and file `next_word_predictor_app.py`.
4. Click **Deploy**. Your app will be available at `https://<your-subdomain>.streamlit.app`.

## ⚠️ Notes
- This demo trains a small model at startup to keep the example self-contained. For production:
  - Pretrain a model on a large corpus.
  - Save model weights (TensorFlow SavedModel or .h5) and **load** them at startup to avoid training on each cold start.
- Streamlit Community Cloud provides HTTPS automatically.

## 👤 Author
**Next-Word Predictor — by Ansh Raj**
