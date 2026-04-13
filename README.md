# Loan Eligibility Predictor

A Machine Learning application to predict loan eligibility using a K-Nearest Neighbors (KNN) classifier.

## Project Structure

```
├── Loan Eligibility Predictor.ipynb  # Main notebook with data exploration and model training
├── app.py                            # Streamlit web application
├── requirements.txt                  # Python dependencies
├── loan_prediction_model.pkl         # Trained KNN model (generated after running notebook)
├── scaler.pkl                        # MinMaxScaler for preprocessing (generated after running notebook)
├── feature_columns.pkl               # Feature column names (generated after running notebook)
├── train_ctrUa4K.csv               # Training dataset
└── test_lAUu6dG.csv                # Test dataset
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Notebook (Optional)

Run the Jupyter notebook to train the model and generate the `.pkl` files:

```bash
jupyter notebook "Loan Eligibility Predictor.ipynb"
```

**Important:** Execute all cells to train the model and save the required `.pkl` files.

### 3. Run the Streamlit Application

Once the model files are generated, start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

- **User-friendly Interface**: Clean and intuitive web interface
- **Real-time Predictions**: Get instant loan eligibility predictions
- **Input Validation**: Validates user input before making predictions
- **Summary Display**: Shows a detailed summary of your application

## How to Use

1. Fill in your personal and financial details in the input form
2. Select demographic information (gender, marital status, etc.)
3. Click "🔮 Predict Loan Eligibility"
4. View the prediction result and your application summary

## Model Details

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Features**: 16 engineered features based on loan application data
- **Training Data**: Cleaned and preprocessed loan application dataset
- **Preprocessing**: 
  - Square root transformation for skewed numerical features
  - MinMax scaling for feature normalization
  - One-hot encoding for categorical variables

## Input Features

### Personal Information
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term

### Demographics
- Gender (Male/Female)
- Marital Status (Yes/No)
- Number of Dependents
- Education (Graduate/Undergraduate)
- Self Employment Status
- Property Area (Rural/Semiurban/Urban)

## Prediction Output

- **✅ APPROVED**: Loan application is likely to be approved
- **❌ NOT APPROVED**: Loan application may not be approved

## Notes

- The model's accuracy depends on the quality of the training data
- Use this tool for reference purposes; actual loan decisions should be made by financial institutions
- The feature values should match the format used during model training

## Contact

For issues or questions, please refer to the notebook documentation.
