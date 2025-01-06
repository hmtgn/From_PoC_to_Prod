from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from typing import List
import sys
import os

# Ajouter le chemin au modèle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from predict.predict.run import TextPredictionModel   

# Charger le modèle
model = TextPredictionModel.from_artefacts("C:\\Users\\hugo.montagnon\\Desktop\\Divers\\EPF\\From PoC to Prod\\poc-to-prod-capstone\\train\\data\\artefacts\\2024-12-11-10-53-19")

# Initialiser FastAPI
app = FastAPI(title="Text Prediction API", version="1.0.0")

# Modèle Pydantic pour valider l'entrée utilisateur
class PredictionRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    predictions: List[str]

# Endpoint pour envoyer une phrase et obtenir les tags
@app.get("/input", response_class=HTMLResponse)
async def input_form():
    """
    Formulaire HTML pour envoyer une phrase et obtenir les tags.
    """
    html_content = """
    <html>
        <head>
            <title>Text Prediction</title>
        </head>
        <body>
            <h2>Enter your phrase below:</h2>
            <form action="/predict_tags" method="post">
                <label for="text">Phrase:</label>
                <input type="text" id="text" name="text">
                <button type="submit">Submit</button>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict_tags")
async def predict_tags(text: str = Form(...)):
    """
    Endpoint pour traiter une phrase et renvoyer les tags prédits.
    """
    try:
        # Si la phrase est vide
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text provided.")

        # Effectuer la prédiction
        predictions = model.predict([text])  # Convertir en liste car le modèle attend une liste
        return {"input": text, "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint racine
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")
