# 1. Library imports
import uvicorn
from fastapi import FastAPI
from ImageNotes import ImageNote
# from main import TypePrediction, segment_image,saveResult
from main  import predictImage, saveResult

# 2. Create the app object
app = FastAPI()

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To KeVan Youtube Channel': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_crack(data:ImageNote):
    data = data.dict()
    img=data['image']
    building= data['building']
    (result,anomaly)= predictImage(img)
    if (anomaly):
        result['building']=building
        saveResult(result)     
    return {
        'anomaly':str(anomaly),
        'prediction': str(result.get('prediction',"")),
        'type': result.get('type',""),
        'image': result.get('segment_image',"")
    }


    # return {
    #     'prediction': str(pred)
    # }


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload
