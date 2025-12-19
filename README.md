
# VizSprints Analytics Dashboard

Interactive Streamlit dashboard with:
- Login authentication
- Funnel (signup → onboarding → activation)
- Retention cohorts (heatmap & tables)
- A/B testing with z-test, p-value, CI
- PDF export (in-memory, no local file saved)

---

## 🔗 Links
- **Live Demo**: Click here to open the app
- **Source Code**: GitHub Repository

---

## 🚀 How to Run Locally
```bash
git clone https://github.com/p27074126-ctrl/vizsprints-analytics.git
cd vizsprints-analytics
pip install -r requirements.txt
python generate_data.py
streamlit run dashboard.py
