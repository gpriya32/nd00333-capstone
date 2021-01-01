from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.model_selection import train_test_split
from azureml.core import Workspace, Dataset

subscription_id = 'f5091c60-1c3c-430f-8d81-d802f6bf2414'
resource_group = 'aml-quickstarts-132780'
workspace_name = 'quick-starts-ws-132780'

workspace = Workspace(subscription_id, resource_group, workspace_name)

ds = Dataset.get_by_name(workspace, name='heart_failure')
ds.to_pandas_dataframe()


x=ds[:,:-1]
y=ds[:,-1]

# TODO: Split data into train and test sets.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

     # dumps the run
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model,'outputs/model.joblib')

if __name__ == '__main__':
    main()
