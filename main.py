import json
# Used to run FastAPI Asynchronous.
import uvicorn
# Used for data manipulation and Numerical Operations
import numpy as np
# Used for loading data from datasets( CSV )
import pandas as pd
# Used to define the type "List"
from typing import List
# Used for building APIs in Python
from fastapi import FastAPI, HTTPException
# Used for defining request data model
from pydantic import BaseModel, root_validator
# Used for Linear Regression Model 
from sklearn.linear_model import LinearRegression


# Importing CSV file
df = pd.read_csv('example.csv')


# Split data to input and output
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

# Create and model and fit the data
model = LinearRegression()
model.fit(X, y)

# Defining the request data model for example { value: [1.0,2.3,3.1] }
class FeatureValues(BaseModel):
    values: List[float]

    @root_validator(pre=True)
    def check_size(cls, values):
        if len(values['values']) != 3:
            print(len(values['values']))
            raise ValueError('Input array length must be 3')
        return values

app = FastAPI()

@app.post("/predict")
def predict(feature_values: FeatureValues):
    try:
        # Get the input features from the API JSON format
        featVal = np.array(feature_values.values).reshape(1, -1)

        # Predict the value based on the input array
        predicted_value = model.predict(featVal)[0]
        return {"prediction": predicted_value}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




# Please run this command first to install all the requirments
# pip install -r requirements.txt


# To run the server run the following command in the cmd
# uvicorn main:app --reload