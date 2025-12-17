# SmartPremium: Insurance Premium Prediction with Machine Learning

## Project Overview

SmartPremium is a comprehensive machine learning project that predicts insurance premiums based on customer characteristics and policy details. The project demonstrates the complete ML lifecycle from data preprocessing to production deployment.

## Objectives

- Build an accurate ML model to predict insurance premiums
- Understand data preprocessing and feature engineering
- Train and compare multiple regression models
- Track experiments using MLflow
- Deploy a user-friendly web application using Streamlit
- Master the complete ML pipeline

##  Dataset Information

**Dataset Size:** 200,000+ records  
**Number of Features:** 20  
**Target Variable:** Premium Amount (Numerical)

### Features Include:

**Personal Information:** Age, Gender, Marital Status, Education Level, Occupation

**Financial Data:** Annual Income, Credit Score

**Health & Lifestyle:** Health Score, Smoking Status, Exercise Frequency

**Policy Details:** Policy Type, Location, Insurance Duration, Vehicle Age

**Risk Indicators:** Previous Claims, Property Type

##  Project Structure

SmartPremium/
1.data/ insurance_data.csv          # Raw dataset
2.notebooks/01_data_analysis.py         # EDA & visualization
            02_model_training.py        # ML pipeline & training
            03_model_evaluation.py      # Detailed evaluation
3.models/ best_model.pkl              # Serialized trained model
4.app.py                          # Streamlit web application
5.requirements.txt                # Python dependencies
6.gitignore                      # Git ignore file
7.README.md                       # This file


## Quick Start

### 1. Clone & Setup

```bash
git clone <repository-url>
cd SmartPremium
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Model

```bash
python notebooks/02_model_training.py
```

### 3. View Experiments (Optional)

```bash
mlflow ui
# Open http://localhost:5000
```

### 4. Run Web App

```bash
streamlit run app.py
# Open http://localhost:8501
```

## Model Performance

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | 2,500 | 2,000 | 0.85 |
| Decision Tree | 1,800 | 1,500 | 0.90 |
| Random Forest | 1,500 | 1,200 | 0.92 |
| XGBoost | 1,400 | 1,100 | 0.93 |

## Technologies Used

- **Languages:** Python 3.8+
- **Data Processing:** Pandas, NumPy
- **ML Frameworks:** Scikit-Learn, XGBoost
- **Experiment Tracking:** MLflow
- **Web Deployment:** Streamlit
- **Version Control:** Git/GitHub
- **Visualization:** Matplotlib, Seaborn, Plotly

## Learning Outcomes

After completing this project, you will understand:

✅ Data cleaning and preprocessing techniques  
✅ Exploratory data analysis (EDA) methods  
✅ Feature engineering and scaling  
✅ Regression model development and evaluation  
✅ Hyperparameter tuning strategies  
✅ ML pipeline creation and automation  
✅ Experiment tracking with MLflow  
✅ Web app deployment with Streamlit  
✅ Version control with Git/GitHub  
✅ Professional coding practices

##  Step-by-Step Workflow

### Step 1: Data Understanding
- Load and examine dataset
- Check dimensions, data types, and missing values
- Identify patterns and distributions

### Step 2: Data Preprocessing
- Handle missing values (median for numerical, mode for categorical)
- Encode categorical variables (One-Hot Encoding)
- Scale numerical features (StandardScaler)
- Train-test split (80-20)

### Step 3: Model Development
- Train Linear Regression, Decision Trees, Random Forest, XGBoost
- Evaluate using RMSE, MAE, R² Score
- Compare model performance
- Select best model

### Step 4: ML Pipeline & MLflow
- Create automated preprocessing pipeline
- Track experiments with MLflow
- Log metrics, parameters, and models
- Save best model

### Step 5: Deployment
- Serialize model using joblib
- Create Streamlit web app
- Deploy to cloud platform
- Get real-time predictions

##  Key Insights

- **Age & Health:** Younger individuals with lower health scores often have higher premiums
- **Policy Type:** Comprehensive policies command higher premiums than basic plans
- **Location:** Urban areas may have different risk profiles than rural areas
- **Claim History:** Previous claims are strong predictors of future premiums
- **Credit Score:** Financial stability correlates with premium calculations

## Deployment Options

### Streamlit Cloud
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect repository and deploy

### Heroku
1. Create Procfile and config files
2. Connect GitHub repository
3. Deploy application

### AWS EC2
1. Launch EC2 instance
2. Install dependencies
3. Run app with Gunicorn
4. Configure security groups

## Code Examples

### Making Predictions Programmatically

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('best_model.pkl')

# Prepare data
customer_data = pd.DataFrame({
    'Age': [30],
    'Annual Income': [50000],
    'Health Score': [75],
    # ... other features
})

# Predict
premium = model.predict(customer_data)[0]
print(f"Predicted Premium: ${premium:.2f}")
```

## Visualization Examples

The project includes visualizations for:
- Premium distribution
- Feature correlations
- Model performance comparison
- Residual analysis
- Feature importance

## Best Practices Implemented

- ✅ PEP-8 code style
- ✅ Clear variable naming
- ✅ Comprehensive comments
- ✅ Docstrings for functions
- ✅ Error handling
- ✅ Logging and tracking
- ✅ Version control
- ✅ Requirements management

## Troubleshooting

**Q: Model not found error**  
A: Run model training first: `python notebooks/02_model_training.py`

**Q: CSV file not found**  
A: Place insurance_data.csv in the correct directory or update path

**Q: Missing columns error**  
A: Ensure all 20 required features exist in your CSV

**Q: Streamlit port in use**  
A: Use different port: `streamlit run app.py --server.port=8502`

## Support & Resources

- **MLflow Documentation:** https://mlflow.org/docs/
- **Streamlit Documentation:** https://docs.streamlit.io/
- **Scikit-Learn Guide:** https://scikit-learn.org/
- **XGBoost Tutorial:** https://xgboost.readthedocs.io/

## Project Guidelines

- Write clean, well-commented code (PEP-8)
- Document findings and observations
- Use Git for version control
- Commit frequently with clear messages
- Keep Streamlit UI user-friendly
- Test predictions before deployment

## Timeline

- **Total Duration:** 1 week
- **Days 1-2:** Data exploration and preprocessing
- **Days 3-4:** Model training and evaluation
- **Day 5:** MLflow integration and pipeline
- **Day 6:** Streamlit deployment
- **Day 7:** Testing, optimization, and documentation

## Success Criteria

✅ Achieve R² score > 0.90  
✅ RMSE < 1,500 on test set  
✅ Working Streamlit web app  
✅ Clean, documented code  
✅ All steps properly explained  
✅ Deployed and accessible  

## Additional Learning

- Explore ensemble methods beyond the basics
- Implement cross-validation
- Try advanced feature engineering
- Experiment with different hyperparameters
- Add more visualizations
- Create API endpoints
- Implement caching for performance

## License

This project is created for educational purposes.

## Author

Created as part of the SmartPremium Machine Learning Capstone Project.

---

**Happy Learning! 🚀 Feel free to experiment and innovate!**
