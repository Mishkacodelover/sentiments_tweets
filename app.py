
import uvicorn
from fastapi import FastAPI
from twitterOpinion import twitterOpinion
import numpy as np
import pickle
import pandas as pd

app = FastAPI()
pickle_in = open("grid.pkl","rb")
grid=pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.get('/{name}')
def get_name(name: str):
    return {'Welcome to my api': f'{name}'}


@app.post('/predict')
def predict_twitter(data:twitterOpinion):
    data = data.dict()
   
    tweet=data['tweet']
  
 
    prediction = grid.predict([[tweet]])
    if(prediction[0] == 0):
        prediction="The opinion is positive"
    else:
        prediction="This opinion is negative"
    return {
        'prediction': prediction
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload