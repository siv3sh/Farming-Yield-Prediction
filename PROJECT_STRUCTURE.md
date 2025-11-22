# Project Structure

```
PA HACKATHON/
├── README.md                          # Main project documentation
├── DASHBOARD_GUIDE.md                 # Streamlit dashboard user guide
├── Presentation_Slides.md             # Presentation slides content
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── Data & Models/
│   ├── Synthetic_Farming_Dataset_With_Seasonality_And_Challenge.csv
│   ├── trained_xgboost_model.pkl     # Trained XGBoost model
│   ├── feature_scaler.pkl            # Feature scaler
│   ├── label_encoder_crop.pkl        # Crop label encoder
│   ├── feature_columns.pkl           # Feature column names
│   └── optimized_input_recommendations.csv  # Optimization results
│
├── Notebooks & Analysis/
│   ├── Farming_Yield_Prediction_Complete_Solution.ipynb  # Main analysis notebook
│   └── Farming_Yield_Prediction_Complete_Solution.py     # Exported Python script
│
├── Dashboard/
│   └── streamlit_dashboard.py        # Interactive Streamlit dashboard
│
├── Scripts/
│   ├── train_and_save_model.py      # Standalone model training script
│   ├── run_notebook.py              # Notebook execution script
│   ├── run_notebook_robust.py       # Robust notebook execution
│   ├── run_dashboard.sh             # Dashboard startup script (Linux/Mac)
│   └── run_dashboard.bat            # Dashboard startup script (Windows)
│
└── Visualizations/
    ├── correlation_heatmap.png
    ├── eda_overview.png
    ├── feature_importance.png
    ├── seasonal_yield_trends.png
    ├── crop_input_efficiency.png
    ├── optimization_comparison.png
    ├── partial_dependence_plots.png
    ├── residual_plots.png
    └── cost_yield_tradeoff.png
```

## Key Files

### Core Files
- **Farming_Yield_Prediction_Complete_Solution.ipynb**: Complete analysis with EDA, modeling, SHAP, and optimization
- **streamlit_dashboard.py**: Interactive web dashboard with real-time analysis
- **Synthetic_Farming_Dataset_With_Seasonality_And_Challenge.csv**: Dataset

### Model Files
- **trained_xgboost_model.pkl**: Trained XGBoost model (724KB)
- **feature_scaler.pkl**: StandardScaler for feature preprocessing
- **label_encoder_crop.pkl**: Label encoder for crop types
- **feature_columns.pkl**: List of feature column names

### Scripts
- **train_and_save_model.py**: Train and save model files (use if model files are missing)
- **run_notebook_robust.py**: Execute notebook with error handling
- **run_dashboard.sh/bat**: Quick start scripts for dashboard

### Documentation
- **README.md**: Project overview and setup instructions
- **DASHBOARD_GUIDE.md**: Detailed dashboard usage guide
- **Presentation_Slides.md**: Presentation content
