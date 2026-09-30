import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

class Columns:

    DROP = ['Unnamed: 0','ID','SSN','Customer_ID','Name']
    NUMERIC = ['Monthly_Inhand_Salary','Num_Bank_Accounts','Num_Credit_Card',
               'Interest_Rate','Delay_from_due_date','Num_Credit_Inquiries','Credit_Utilization_Ratio',
               'Total_EMI_per_month','Age','Annual_Income','Num_of_Loan','Num_of_Delayed_Payment',
               'Changed_Credit_Limit','Outstanding_Debt','Amount_invested_monthly','Monthly_Balance',
               'Credit_History_Age_Months']
    CATEGORICAL = ['Month','Occupation','Type_of_Loan',
                   'Credit_Mix','Payment_of_Min_Amount','Payment_Behaviour']
    SUSPICIOUS_NUMERIC = ['Age','Annual_Income','Num_of_Loan','Num_of_Delayed_Payment',
                          'Changed_Credit_Limit','Outstanding_Debt','Amount_invested_monthly','Monthly_Balance']
    OHE = ['Occupation','Type_of_Loan']

class EncodingMaps:

    MONTH = {'January': 1,'February': 2,'March': 3,'April': 4,'May': 5,'June': 6,'July': 7,'August': 8}
    CREDIT_MIX = {'Bad': 0,'Standard': 1,'Good': 2}
    PAYMENT_MIN_AMOUNT = {'No': 0,'Yes': 1}
    CREDIT_SCORE = {'Poor': 0,'Standard': 1,'Good': 2}

class CleaningRules:

    HARD_LIMITS = {'Num_Bank_Accounts': (0, 100),'Num_Credit_Card': (0, 100),
                   'Num_Credit_Inquiries': (0, 1000),'Interest_Rate': (0, 100),'Age': (18, 100)}
    LOWER_ONLY = {'Delay_from_due_date': 0,'Num_of_Loan': 0}

class CreditScorePreprocssor:

    def __init__(self, test_size=0.2, random_state=42):

        self.test_size = test_size
        self.random_state = random_state
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = OneHotEncoder(drop="first",handle_unknown="ignore",sparse_output=False)
        self.outlier_bounds = {}

    def clean_and_split(self, data_path):

        df = pd.read_csv(data_path)
        X = df.drop(columns=["Credit_Score"])
        y = df["Credit_Score"]

        y = y.map(EncodingMaps.CREDIT_SCORE)
        return train_test_split(X,y,test_size=self.test_size,random_state=self.random_state,stratify=y)

    def _clean_numeric_strings(self, df):

        if 'Monthly_Balance' in df.columns:
            df['Monthly_Balance'] = (df['Monthly_Balance'].astype(str).str
                                     .replace('__-333333333333333333333333333__','',regex=False))

        for col in Columns.SUSPICIOUS_NUMERIC:
            if col not in df.columns:
                continue
            df[col] = (df[col].astype(str).str.replace('_', '', regex=False).replace('nan', np.nan))
            df[col] = pd.to_numeric(df[col],errors='coerce')

        return df

    def _convert_credit_history(self, df):

        if 'Credit_History_Age' not in df.columns:
            return df

        temp = df['Credit_History_Age'].str.extract(r'(\d+)\s+Years?\s+and\s+(\d+)\s+Months?')
        years = pd.to_numeric(temp[0], errors='coerce')
        months = pd.to_numeric(temp[1], errors='coerce')

        df['Credit_History_Age_Months'] = (years * 12 + months)
        df.drop(columns=['Credit_History_Age'],inplace=True)

        return df

    def _clean_categories(self, df):

        if 'Type_of_Loan' in df.columns:

            df['Type_of_Loan'] = (df['Type_of_Loan'].astype(str).str.split(',').str[0].str.strip())

        df.replace({'nan': np.nan,'NaN': np.nan,'NAN': np.nan,'_': np.nan,'Not Specified': np.nan,
                    'NM': np.nan,'!@9#%8': np.nan,'_______': np.nan}, inplace=True)
                    
        return df

    def _initial_cleaning(self, X):

        X = X.copy()
        cols_to_drop = [ c for c in Columns.DROP if c in X.columns ] 
        X = X.drop(columns=cols_to_drop)
        X = self._clean_numeric_strings(X)
        X = self._convert_credit_history(X)
        X = self._clean_categories(X)

        return X


    def fit(self, X):

        X = self._initial_cleaning(X)

        self.num_imputer.fit(X[Columns.NUMERIC])
        self.cat_imputer.fit(X[Columns.CATEGORICAL])
        
        X[Columns.NUMERIC] = (self.num_imputer.transform(X[Columns.NUMERIC]))
        X[Columns.CATEGORICAL] = (self.cat_imputer.transform(X[Columns.CATEGORICAL]))

        for col in Columns.NUMERIC:

            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.outlier_bounds[col] = (
                q1 - 1.5 * iqr,
                q3 + 1.5 * iqr)

        self.encoder.fit(X[Columns.OHE])

        return self

    def transform(self, X):

        X = self._initial_cleaning(X)

        X[Columns.NUMERIC] = (self.num_imputer.transform(X[Columns.NUMERIC]))
        X[Columns.CATEGORICAL] = (self.cat_imputer.transform(X[Columns.CATEGORICAL]))

        for col, (low, high) in CleaningRules.HARD_LIMITS.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=low,upper=high)

        for col, low in CleaningRules.LOWER_ONLY.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=low)

        for col in Columns.NUMERIC:
            lower, upper = self.outlier_bounds[col]
            X[col] = X[col].clip(lower=lower,upper=upper)

        X['Spending_Level'] = (X['Payment_Behaviour'].str.extract(r'(Low|High)_spent')[0]
            .map({'Low': 0,'High': 1}))
        X['Payment_Size'] = (X['Payment_Behaviour'].str.extract(r'(Small|Medium|Large)_value')[0]
            .map({'Small': 0,'Medium': 1,'Large': 2}))
        X.drop(columns=['Payment_Behaviour'],inplace=True)
        X['Month'] = X['Month'].map(EncodingMaps.MONTH)
        X['Credit_Mix'] = X['Credit_Mix'].map(EncodingMaps.CREDIT_MIX)
        X['Payment_of_Min_Amount'] = (X['Payment_of_Min_Amount'].map(EncodingMaps.PAYMENT_MIN_AMOUNT))

        encoded = self.encoder.transform(X[Columns.OHE])
        encoded_df = pd.DataFrame(encoded,columns=self.encoder.get_feature_names_out(Columns.OHE),index=X.index)
        X = pd.concat([X.drop(columns=Columns.OHE),encoded_df],axis=1)

        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)
    