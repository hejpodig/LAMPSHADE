# FullControl Lampshade (Streamlit)

## Run locally (Windows / PowerShell)

From the repo root (`c:\fullcontrol_playground`).

If you have multiple Python versions installed, prefer Python 3.11+ (Streamlit support can lag behind the newest Python versions).

```powershell
# Use the Python launcher if available (recommended)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Share with friends (no install)

Use **Streamlit Community Cloud**:

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io
3. Create a new app:
   - **Repository**: your GitHub repo
   - **Branch**: `main` (or your branch)
   - **Main file path**: `streamlit_app/app.py`
4. Deploy.

Your friends can open the generated link in a browser.

### Notes

- Dependencies are defined in `requirements.txt` (root).
- Python version for hosting is set in `runtime.txt`.
