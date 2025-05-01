Perfect! Since your GitHub repo is [https://github.com/mihika123945/Cinescope](https://github.com/mihika123945/Cinescope), here's the updated and finalized version of your `README.md` tailored specifically for that URL:

---

```markdown
# 🎬 CineScope: Personalized Movie Recommendations and Trend Analysis

**CineScope** is an interactive web dashboard that combines visual storytelling with intelligent movie recommendations using IMDb data. Built using Dash, Plotly, and PyTorch, it allows users to explore media consumption trends and receive personalized suggestions based on genre, rating, and release year.

---

## 📌 Features

- 📊 **Visual Analytics**:
  - Trend analysis of movies vs TV shows
  - Success ratio of high-rated titles across years
  - Genre popularity and rating distribution

- 🎯 **Recommendation Engine**:
  - Personalized movie suggestions based on cosine similarity
  - Adjustable filters for rating and release window
  - Real-time interactive dashboard built with Dash

---

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `app_dash.py` | Main Python script to run the Dash app |
| `movie_features.h5` | Precomputed content feature vectors |
| `movies_data.h5` | Cleaned movie metadata |
| `preprocessed_data.h5` | Aggregated data for visualizations |

---

## ⚙️ How to Run

1. **Clone the Repository**
```bash
git clone https://github.com/mihika123945/Cinescope.git
cd Cinescope
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Dashboard**
```bash
python app_dash.py
```

4. **Visit in Browser**
```
http://127.0.0.1:8050
```

---

## 🧠 Tech Stack

- Python (Dash, Plotly, Pandas, NumPy)
- PyTorch (Cosine similarity-based recommendations)
- HDF5 (Data storage with `h5py`)
- Scikit-learn (Feature preprocessing)

---

