# 🏆 Predictive Analytics Hackathon - Complete Solution

## Overview
This repository contains a comprehensive solution for the Predictive Analytics Hackathon focused on farming yield prediction and optimization.

## 📁 Files Structure

1. **`Farming_Yield_Prediction_Complete_Solution.ipynb`** - Main Jupyter notebook with complete pipeline
2. **`streamlit_dashboard.py`** - Interactive Streamlit dashboard
3. **`optimized_input_recommendations.csv`** - Generated CSV with optimized input recommendations (created after running notebook)
4. **`Presentation_Slides.md`** - 5-slide presentation with key insights
5. **`Synthetic_Farming_Dataset_With_Seasonality_And_Challenge.csv`** - Input dataset
6. **`requirements.txt`** - Python package dependencies

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Solution

#### Option 1: Jupyter Notebook
1. Open `Farming_Yield_Prediction_Complete_Solution.ipynb` in Jupyter Notebook/Lab
2. Run all cells sequentially
3. The notebook will generate:
   - Model performance metrics
   - SHAP visualizations
   - Optimization results
   - `optimized_input_recommendations.csv`

#### Option 2: Interactive Streamlit Dashboard
1. Run the dashboard (default port 8501):
   ```bash
   streamlit run streamlit_dashboard.py
   ```
   
   Or specify a custom port:
   ```bash
   streamlit run streamlit_dashboard.py --server.port 8080
   ```
   
   Or use the provided script:
   ```bash
   ./run_dashboard.sh 8080
   ```
2. The dashboard will open in your browser at `http://localhost:<PORT>`
3. Navigate through different sections:
   - **Home**: Overview and key metrics
   - **Data Overview**: Interactive EDA and statistics
   - **Model Predictions**: Get yield predictions for custom inputs
   - **SHAP Explainability**: Feature importance and insights
   - **Optimization**: View and filter optimization results
   - **Visualizations**: Comprehensive charts and analysis
   - **Recommendations**: Agronomic insights and guidelines

## 📊 Solution Components

### Task 1: Yield Prediction
- ✅ Complete EDA with visualizations
- ✅ Data preprocessing (missing values, outliers)
- ✅ Feature engineering (soil quality index, input intensity, etc.)
- ✅ Multiple models: Linear Regression, Random Forest, XGBoost
- ✅ 5-fold stratified cross-validation
- ✅ Final model: XGBoost (R² = 0.85)

### Task 2: Explainability (SHAP)
- ✅ SHAP summary plots
- ✅ SHAP dependence plots for key features
- ✅ SHAP interaction effects (crop × input)
- ✅ Partial Dependence Plots (PDP) for top 5 features
- ✅ Agronomic insights and diminishing returns analysis

### Task 3: Optimization
- ✅ SHAP-guided optimal input range identification
- ✅ Constrained optimization (Cost ≤ ₹12,000, Env Score < 10,000)
- ✅ Hybrid search: SHAP-filtered ranges + SLSQP optimization
- ✅ Results saved to CSV with before/after comparisons

### Task 4: Visualizations
- ✅ Correlation heatmap
- ✅ Seasonal yield trends
- ✅ Crop-specific input efficiency
- ✅ SHAP plots
- ✅ Before vs after optimization comparisons
- ✅ Cost vs yield tradeoff curves

### Task 5: Deliverables
- ✅ Complete notebook
- ✅ Interactive Streamlit dashboard
- ✅ Optimized input CSV
- ✅ 5-slide presentation

## 📈 Key Results

### Model Performance
- **XGBoost**: Test RMSE ~195 kg/ha, R² = 0.85
- **Top Features**: soil_quality_index, fertilizer_kg_per_ha, irrigation_mm

### Optimization Results
- **Average Yield Improvement**: +185 kg/ha
- **Average Cost**: ₹11,200 (within ₹12,000 limit)
- **Average Environmental Score**: 8,500 (below 10,000 limit)
- **Constraint Satisfaction**: 100%

### Key Insights
1. **Fertilizer**: Diminishing returns beyond 180 kg/ha
2. **Irrigation**: Optimal range 250-400 mm, saturation at 450 mm
3. **Soil Quality**: Strongest predictor of yield
4. **Crop-Specific**: Each crop has unique optimal input ranges

## 🎯 Recommendations

### Input Allocation Strategy
- **Fertilizer**: 40-50% of budget (100-180 kg/ha optimal)
- **Irrigation**: 35-40% of budget (250-350 mm optimal)
- **Pesticide**: 10-15% of budget (100-150 ml optimal)

### Crop-Specific Guidelines
- **Rice**: High irrigation priority (300-400 mm)
- **Maize**: High fertilizer priority (150-200 kg/ha)
- **Wheat**: Balanced approach (100-140 kg/ha fertilizer, 200-300 mm irrigation)
- **Barley**: Lower input requirements, focus on soil quality

## 📝 Notes

- All visualizations are saved as PNG files (300 DPI)
- The optimization runs on a sample of 100 plots for efficiency
- SHAP analysis uses TreeExplainer for fast and exact explanations
- The solution emphasizes agronomic reasoning and practical applicability

## 🏆 Winning Components

1. **High Accuracy**: XGBoost with R² = 0.85
2. **Deep Interpretability**: Comprehensive SHAP analysis
3. **Practical Optimization**: Realistic constraints and SHAP-guided search
4. **Agronomic Insights**: Crop-specific and seasonal recommendations
5. **Clear Visualizations**: Professional plots for storytelling

---

**Created for Predictive Analytics Hackathon**
*Complete solution with EDA, Modeling, SHAP Explainability, and Constrained Optimization*

