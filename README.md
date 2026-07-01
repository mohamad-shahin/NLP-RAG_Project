# 👁️ RetinaScan AI

A simple, distinctive web interface for a retinal disease classification model.
Upload a fundus (retina) photo and get an instant multi-label screening for:

- **Overall Disease Risk**
- **Diabetic Retinopathy (DR)**
- **Optic Disc Cupping (ODC)**
- **Tessellation (TSLN)**

The model is an **EfficientNetB4**-based CNN, fine-tuned on the
[RFMiD retinal disease classification dataset](https://www.kaggle.com/datasets/andrewmvd/retinal-disease-classification).

---

## 1. Project structure

```
retinal-disease-classifier/
├── app.py                     # Streamlit interface
├── requirements.txt
├── .streamlit/config.toml     # theme colors
├── model/
│   └── retinal_model.h5       # ⬅ your trained model goes here (not included)
└── training/
    └── train_retinal_model.ipynb   # training notebook (from Kaggle), with a
                                     # save step added so it exports a .h5 file
```

## 2. Get a trained model file

The interface needs a trained model file to make predictions. This repo does
not include one — you need to run the training notebook once to produce it:

1. Open `training/train_retinal_model.ipynb` on **Kaggle** (with the
   `retinal-disease-classification` dataset attached, and a GPU accelerator
   turned on).
2. Run all cells. Training EfficientNetB4 on the full dataset will take a
   while (expect well over an hour, faster with a good GPU).
3. A cell has already been added right after training that runs:
   ```python
   model.save("retinal_model.h5")
   ```
   This produces `retinal_model.h5` in the Kaggle output panel.
4. Download that file.

You now have two options for using it with the app:

### Option A — Commit it to the repo (simplest, works if file < 100 MB)
Place the file at `model/retinal_model.h5`. Note it's excluded by
`.gitignore` by default — remove that line (`model/*.h5`) if you want to
commit it, or use [Git LFS](https://git-lfs.com/) if it's close to GitHub's
size limit.

### Option B — Host it externally (recommended for larger files)
Upload `retinal_model.h5` somewhere with a direct download link — e.g. a
[Hugging Face model repo](https://huggingface.co/new), a GitHub Release
asset, or cloud storage. Then set an environment variable when you deploy:

```
MODEL_URL=https://your-direct-download-link/retinal_model.h5
```

The app will download it automatically on first run and cache it.

## 3. Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## 4. Put it on GitHub and get a public link

1. Create a new repository on GitHub and push this folder to it:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: RetinaScan AI"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** (Streamlit
   Community Cloud) and sign in with your GitHub account.
3. Click **"New app"**, pick your repository, branch `main`, and main file
   `app.py`.
4. If you're using **Option B** above (external model URL), add `MODEL_URL`
   as a secret/environment variable in the app's "Advanced settings" before
   deploying.
5. Click **Deploy**. You'll get a public link like:
   ```
   https://your-app-name.streamlit.app
   ```
   Share that link with anyone — the app runs directly from your GitHub repo.

## 5. Notes

- The app applies the same preprocessing used in training
  (`preprocess_input` from `tensorflow.keras.applications.efficientnet`,
  resized to 380×380).
- The 0.7 decision threshold matches the one used in the original notebook;
  adjust `THRESHOLD` in `app.py` if you'd like a different sensitivity.
- This tool is for educational/research purposes only and is **not** a
  medical diagnostic device.
