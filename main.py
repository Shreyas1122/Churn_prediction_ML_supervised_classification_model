from fastapi import FastAPI
from pydantic import BaseModel,Field
import joblib
from typing import Literal
import pandas as pd 
from fastapi.middleware.cors import CORSMiddleware

model =joblib.load("customer_churn_predictions.pkl")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#Creating a validation class for the input variables 
class Inputvariables (BaseModel):
    Age:int=Field(...,ge=0,le=100)
    Gender:Literal["Male","Female"]
    Tenure:int=Field(...,ge=0,le=2000)
    Usage_Frequency:int=Field(...,ge=0,le=1000)
    Support_Calls_days:int=Field(...,ge=0,le=1000)
    Payment_Delay_days:int=Field(...,ge=0,le=1000)
    Subscription_Type:Literal["Basic","Standard","Premium"]
    Contract_Length:Literal["Monthly","Quarterly","Annual"]
    Total_Spend:int=Field(...,ge=0,le=1000)



#creating class for the data output 
class Outputchurn(BaseModel):
    Churnoutput:int





@app.get("/")
def fun():
    return "Hello world"



@app.post("/predict")
def prediction(data:Inputvariables):
    dataframe=pd.DataFrame(data={
        "Age":data.Age,
        "Gender":data.Gender,
        "Tenure":data.Tenure,
        "Usage Frequency":data.Usage_Frequency,
        "Support Calls":data.Support_Calls_days,
        "Payment Delay":data.Payment_Delay_days,
        "Subscription Type":data.Subscription_Type,
        "Contract Length":data.Contract_Length,
        "Total Spend":data.Total_Spend
    },index=[0])

    prediction_data=model.predict(dataframe)
    print(prediction_data[0])
    return Outputchurn(Churnoutput=prediction_data[0])


    








