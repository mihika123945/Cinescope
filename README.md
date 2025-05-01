
---

# 🎬 CineScope: Personalized Movie Recommendations & Trend Analysis

**CineScope** is an interactive dashboard that analyzes IMDb data to provide personalized movie recommendations and visualize trends in content consumption. Built with Dash and Plotly, it offers users insights into genre distributions, rating patterns, and more.

## 📊 Key Features

- **Interactive Visualizations**: Explore trends in movies and TV shows over time, genre distributions, and rating patterns.
- **Content-Based Recommendation System**: Receive personalized movie suggestions based on selected titles, genres, ratings, and release years.
- **User-Friendly Interface**: Intuitive design for seamless navigation and exploration.

## 🛠️ Technologies Used

- **Python**: Data processing and application logic.
- **Dash & Plotly**: Building interactive web applications and visualizations.
- **Pandas & NumPy**: Data manipulation and analysis.
- **Scikit-learn**: Implementing cosine similarity for recommendations.
- **HDF5 (h5py)**: Efficient storage and retrieval of large datasets.

## 📁 Project Structure

```
CineScope/
├── app_dash.py             # Main application script
├── movie_features.h5       # Precomputed feature vectors for movies
├── movies_data.h5          # Cleaned and processed movie metadata
├── preprocessed_data.h5    # Data used for visualizations
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
├── data.docx               # Data description
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mihika123945/Cinescope.git
   cd Cinescope
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**

   ```bash
   python app_dash.py
   ```

---
