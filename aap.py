import streamlit as st
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeCV

# Load the classifier model
classifier_model_path = 'classifier_model.pkl'
with open(classifier_model_path, 'rb') as f:
    classifier_model = pickle.load(f)

# Load the regressor model
regressor_model_path = 'regressor_model.pkl'
with open(regressor_model_path, 'rb') as f:
    regressor_model = pickle.load(f)


# Function to preprocess input data
def preprocess_input(data):
    data = pd.DataFrame(data, index=[0])  # Convert input to dataframe

    label_encoder = LabelEncoder()

    for column in ['FirstTimeHomebuyer', 'Occupancy', 'Channel', 'PPM', 'PropertyState',
                   'PropertyType', 'LoanPurpose', 'NumBorrowers']:
        if column in data:
            data[column] = label_encoder.fit_transform(data[column])[0]
    
    # Calculate remaining loan term (in months)
    data['RemainingLoanTerm'] = data['OrigLoanTerm'] - data['MonthsInRepayment']

    # Calculate remaining UPB at the time of data collection
    data['RemainingUPB'] = data['OrigUPB'] * (1 - (data['MonthsInRepayment'] / data['OrigLoanTerm']))

    # Create a binary feature for high DTI ratio (e.g., DTI > 43%)
    data['HighDTI'] = (data['DTI'] > 43).astype(int)

    # Create a binary feature for high LTV ratio (e.g., LTV > 80%)
    data['HighLTV'] = (data['LTV'] > 80).astype(int)

    # Create a binary feature for single-family homes
    data['IsSingleFamily'] = (data['PropertyType'] == 'SingleFamily').astype(int)

    return data

# Function to make predictions using the classifier model
def predict_classification(data):
    # Preprocess the input data
    preprocessed_data = preprocess_input(data)
    
    # Make predictions using the classifier model
    predictions = classifier_model.predict(preprocessed_data)
    
    return predictions

# Function to make predictions using the regressor model
def predict_regression(data):
    # Preprocess the input data
    preprocessed_data = preprocess_input(data)
    
    # Make predictions using the regressor model
    predictions = regressor_model.predict(preprocessed_data)
    
    return predictions[0]

# Streamlit app
def main():
    # Add custom CSS to set the background image
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"]{
        background-color: #e5e5f7;
        opacity: 0.8;
        background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #e5e5f7 10px ), repeating-linear-gradient( #444cf755, #444cf7 );
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Set app title
    st.title("Credit Eligibility and Prepayment Risk Prediction")
    
    # Get user input
    user_input = {}  # Store user input in a dictionary
    
    # Example input fields (replace with your own input fields)
    user_input['CreditScore'] = st.number_input("Credit Score", min_value=0, max_value=1000, step=1)
    user_input['FirstTimeHomebuyer'] = st.selectbox("First Time Home Buyer?", ['Y', 'N'])
    user_input['MIP'] = st.number_input("MIP", min_value=0, max_value=100, step=1)
    user_input['Units'] = st.number_input("Units", min_value=0, max_value=4, step=1)
    user_input['Occupancy'] = st.selectbox("Occupancy", ['O', 'I', 'S'])
    user_input['OCLTV'] = st.number_input("OCLTV", min_value=1, max_value=200, step=1)
    user_input['DTI'] = st.number_input("DTI", min_value=1, max_value=100, step=1)
    user_input['OrigUPB'] = st.number_input("Original UPB", min_value=0, max_value=1000000, step=1)
    user_input['LTV'] = st.number_input("LTV", min_value=1, max_value=200, step=1)
    user_input['OrigInterestRate'] = st.number_input("Origal Interest Rate", min_value=0.0, step=0.01)
    user_input['Channel'] = st.selectbox("Channel", ['T', 'R', 'C', 'B'])
    user_input['PPM'] = st.selectbox("PPM (Y/N)", ['Y', 'N'])
    user_input['PropertyState'] = st.selectbox("Property State", ['IL', 'CO', 'KS', 'CA', 'NJ', 'WI', 'FL', 'CT', 'GA', 'TX', 'MD', 'MA', 'SC', 'WY',
                                                                  'NC', 'AZ', 'IN', 'MS', 'NY', 'WA', 'AR', 'VA', 'MN', 'LA', 'PA', 'OR', 'RI', 'UT',
                                                                  'MI', 'TN', 'AL', 'MO', 'IA', 'NM', 'NV', 'VT', 'OH', 'NE', 'HI', 'ID', 'PR', 'DC',
                                                                  'GU', 'KY', 'NH', 'SD', 'ME', 'MT', 'OK', 'WV', 'DE', 'ND', 'AK'])
    user_input['PropertyType'] = st.selectbox("Property Type", ['SF', 'PU', 'CO', 'MH', 'CP', 'LH'])
    user_input['LoanPurpose'] = st.selectbox("Loan Purpose", ['P', 'N', 'C'])
    user_input['OrigLoanTerm'] = st.number_input("Original Loan Term", min_value=100, max_value=400, step=1)
    user_input['NumBorrowers'] = st.selectbox("Number of Borrowers", ['2', '1'])
    user_input['MonthsDelinquent'] = st.number_input("Months Delinquent", min_value=0, step=1)
    user_input['MonthsInRepayment'] = st.number_input("Months In Repayment", min_value=0, step=1)
    
    
    # Perform predictions
    # Add a "Predict" button
    if st.button("Predict"):
        # Perform predictions
        classification_result = predict_classification(user_input)

        if classification_result == 0:
            st.markdown("<p style='font-size: 30px; text-align: center; color: green;' class='output'>Congratulations! You are eligible to get the credit.</p>", unsafe_allow_html=True)
            regression_result = abs(int(predict_regression(user_input)))
            st.markdown(f"<p style='font-size: 30px; text-align: center;' class='output'>The prepayment risk is: {regression_result}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size: 30px; text-align: center; color: red;' class='output'>Sorry, you are not eligible to get the credit.</p>", unsafe_allow_html=True)
        
    
# Run the app
if __name__ == '__main__':
    main()