# Plot Viewer

Streamlit app for browsing evaluation plots by `t`, `num_samples`, and `pass@`.

## Folder layout

Put plot files under `plots/` (nested folders are fine). The app infers the
**evaluation name** from the first folder under `plots/` and reads metadata
from filenames or parent folders.

Examples:

- `plots/humaneval/t0_n1_pass_at_1.png`
- `plots/humaneval/t0p6_n200_pass_at_100.png`
- `plots/mmlu/ablation/t=0.6,num_samples=200,pass@{1,10,100}.png`

## Run locally

```bash
streamlit run streamlit_app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. In Streamlit Cloud, set the app file to `streamlit_app.py`.

The app uses `requirements.txt` for dependencies.
