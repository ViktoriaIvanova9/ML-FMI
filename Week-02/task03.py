import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def will_leave_prediction():
    telecom_df = pd.read_csv('../DATA/telecom_churn_clean.csv', index_col=0)

    X = telecom_df[['account_length', 'customer_service_calls']].values
    y = telecom_df['churn'].values

    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X, y)

    X_new = np.array([[30.0, 17.5],
                    [107.0, 24.1],
                    [213.0, 10.9]])
    
    churn_prediction = knn.predict(X_new)
    print(churn_prediction)

if __name__ == '__main__':
    will_leave_prediction()