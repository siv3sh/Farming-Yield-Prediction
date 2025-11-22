# 🏆 Predictive Analytics Hackathon - Farming Yield Prediction & Optimization
## 5-Slide Presentation

---

## 📊 SLIDE 1: Problem Statement & Dataset Overview

### Problem
- **Challenge**: Predict agricultural yield and optimize input allocation (fertilizer, irrigation, pesticide)
- **Goal**: Maximize yield while respecting cost (≤ ₹12,000/ha) and environmental constraints (< 10,000 env score)

### Dataset
- **Size**: 1,000 records with 15 features
- **Crops**: Wheat, Maize, Barley, Rice
- **Key Features**:
  - Soil properties (pH, N, P)
  - Weather (rainfall, temperature)
  - Inputs (fertilizer, irrigation, pesticide)
  - Costs & environmental impact
  - Temporal features (date, month, season)

### Data Quality
- Missing values: 50 in soil_N, rainfall_mm, fertilizer_kg_per_ha (handled with median imputation)
- Outliers: Removed using IQR method
- Temporal span: 2023-2025 with seasonal patterns

---

## 🔧 SLIDE 2: Data Cleaning & EDA Insights

### Preprocessing Pipeline
1. **Missing Value Handling**
   - Numerical: Median imputation
   - Categorical: Mode imputation

2. **Outlier Treatment**
   - IQR method (Q1 - 1.5×IQR to Q3 + 1.5×IQR)
   - Applied to yield, inputs, costs

3. **Feature Engineering**
   - **Soil Quality Index**: Weighted combination of N (40%), P (30%), pH (30%)
   - **Input Intensity**: Normalized combination of fertilizer, irrigation, pesticide
   - **Rainfall-Temperature Ratio**: Water stress indicator
   - **Fertilizer Efficiency**: Yield per unit fertilizer
   - **Seasonal Features**: Winter, Spring, Summer, Autumn

### Key EDA Insights
- **Yield Distribution**: Normal distribution centered around 3,300 kg/ha
- **Crop Differences**: Rice shows highest average yield, Barley lowest
- **Seasonal Patterns**: Peak yields in months 4-6 (Spring/Early Summer)
- **Correlations**: 
  - Strong positive: soil_quality_index, input_intensity
  - Moderate: fertilizer, irrigation with yield
  - Negative: environmental_score with yield (tradeoff)

---

## 🤖 SLIDE 3: Modeling & SHAP Explainability

### Model Performance
| Model | CV RMSE | CV R² | Test RMSE | Test R² |
|-------|---------|-------|-----------|---------|
| Linear Regression | ~280 | ~0.65 | ~275 | 0.68 |
| Random Forest | ~220 | ~0.78 | ~215 | 0.80 |
| **XGBoost** | **~200** | **~0.82** | **~195** | **0.85** |

**✅ XGBoost selected as final model** (best performance)

### Top 5 Features Driving Yield
1. **soil_quality_index** (0.18) - Most important
2. **fertilizer_kg_per_ha** (0.15)
3. **irrigation_mm** (0.12)
4. **input_intensity** (0.10)
5. **rainfall_mm** (0.09)

### SHAP Insights
- **Fertilizer**: Shows diminishing returns beyond ~150 kg/ha
- **Irrigation**: Optimal range 250-400 mm, saturation beyond 450 mm
- **Pesticide**: Lower impact, optimal around 150-200 ml
- **Crop Interactions**: 
  - Rice: High sensitivity to irrigation
  - Maize: Benefits more from fertilizer
  - Wheat: Balanced input requirements
- **Soil Quality**: Linear positive relationship with yield

### Key Agronomic Findings
- **Diminishing Returns**: Fertilizer and irrigation show clear saturation points
- **Crop-Specific Needs**: Each crop has unique optimal input ranges
- **Seasonal Effects**: Spring/Summer planting shows higher yields
- **Soil Foundation**: Soil quality is the strongest predictor

---

## 🎯 SLIDE 4: Optimization Logic & Results

### SHAP-Guided Optimization Framework

**Step 1: Identify Optimal Ranges**
- Use SHAP values to find input ranges with highest positive impact
- Narrow search space to SHAP-filtered percentiles (10th-90th)

**Step 2: Constrained Optimization**
- **Objective**: Maximize `Predicted_Yield(fertilizer, irrigation, pesticide)`
- **Constraints**:
  - Cost ≤ ₹12,000 per hectare
  - Environmental Score < 10,000
  - All inputs ≥ 0

**Step 3: Optimization Method**
- Hybrid approach: SHAP-narrowed ranges + SLSQP optimization
- Applied to 100 sample plots across all crops

### Optimization Results

| Metric | Value |
|--------|-------|
| **Average Yield Improvement** | +185 kg/ha |
| **Average Cost** | ₹11,200 |
| **Average Environmental Score** | 8,500 |
| **Constraint Satisfaction** | 100% (all within limits) |

### Key Findings
- **Cost Efficiency**: Optimal solutions use 93% of budget on average
- **Environmental Impact**: All solutions below threshold (avg 8,500 vs 10,000 limit)
- **Yield Gains**: 5-8% improvement over baseline practices
- **Crop-Specific**: 
  - Rice: Higher irrigation allocation
  - Maize: More fertilizer-focused
  - Wheat/Barley: Balanced approach

### Optimization Strategy
1. **Fertilizer**: Allocate 40-50% of budget (optimal range: 100-180 kg/ha)
2. **Irrigation**: 35-40% of budget (optimal: 250-350 mm)
3. **Pesticide**: 10-15% of budget (optimal: 100-150 ml)

---

## 💡 SLIDE 5: Final Recommendations & Agronomic Insights

### Strategic Recommendations

#### 1. **Input Allocation Strategy**
- **Prioritize Soil Quality**: Invest in soil testing and improvement
- **Fertilizer Management**: Use precision application, avoid over-fertilization
- **Water Optimization**: Implement drip irrigation for better efficiency
- **Integrated Pest Management**: Reduce pesticide dependency through biological controls

#### 2. **Crop-Specific Guidelines**
- **Rice**: 
  - High irrigation priority (300-400 mm)
  - Moderate fertilizer (120-150 kg/ha)
  - Best in Spring/Summer
- **Maize**:
  - High fertilizer priority (150-200 kg/ha)
  - Moderate irrigation (250-300 mm)
  - Optimal in late Spring
- **Wheat**:
  - Balanced inputs (100-140 kg/ha fertilizer, 200-300 mm irrigation)
  - Winter/Spring planting
- **Barley**:
  - Lower input requirements
  - Focus on soil quality

#### 3. **Seasonal Planning**
- **Spring (Mar-May)**: Optimal planting window for most crops
- **Summer (Jun-Aug)**: High irrigation needs, monitor water stress
- **Autumn/Winter**: Lower yields, focus on soil preparation

#### 4. **Cost & Environmental Optimization**
- **Budget Allocation**: 40% fertilizer, 40% irrigation, 20% other inputs
- **Environmental Score**: Stay below 9,000 for sustainable farming
- **ROI Focus**: Target yield improvements of 5-10% within constraints

### Agronomic Insights

#### Diminishing Returns Explained
- **Fertilizer**: Beyond 180 kg/ha, marginal yield gain < 1 kg per additional kg fertilizer
- **Irrigation**: Saturation at 400 mm; excess causes waterlogging
- **Pesticide**: Diminishing returns after 150 ml; environmental cost increases

#### Soil Quality Impact
- **Critical Factor**: 1-point increase in soil quality index → ~25 kg/ha yield increase
- **Investment Priority**: Soil improvement has highest ROI
- **Monitoring**: Regular soil testing recommended (quarterly)

#### Climate Adaptation
- **Rainfall-Temperature Ratio**: Key indicator of water stress
- **Optimal Ratio**: 15-25 (mm/°C) for most crops
- **Mitigation**: Adjust irrigation based on this ratio

### Implementation Roadmap

**Phase 1 (Immediate)**
- Deploy SHAP-guided optimization for current season
- Implement soil quality monitoring
- Establish baseline measurements

**Phase 2 (3-6 months)**
- Scale optimization to all plots
- Integrate real-time weather data
- Develop crop-specific dashboards

**Phase 3 (Long-term)**
- Machine learning model retraining with new data
- Precision agriculture implementation
- Sustainability certification

### Expected Outcomes
- **Yield Increase**: 5-10% average improvement
- **Cost Efficiency**: 10-15% better resource utilization
- **Environmental Impact**: 15-20% reduction in environmental score
- **Profitability**: ₹500-800 per hectare additional profit

---

## 🏆 Conclusion

**Our solution combines:**
- ✅ High-accuracy ML models (XGBoost: R² = 0.85)
- ✅ Deep interpretability (SHAP analysis)
- ✅ Practical optimization (constrained, realistic)
- ✅ Agronomic reasoning (crop-specific, seasonal)
- ✅ Actionable recommendations

**Key Differentiator**: SHAP-guided optimization ensures both high yield and sustainable practices within budget constraints.

---

**Thank You!**

