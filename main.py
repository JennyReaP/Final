from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd

# Inicializar la app y el motor de plantillas
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Cargar el modelo y el codificador entrenados
encoder = joblib.load("encoder.pkl")
model = joblib.load("carEvaluation.pkl")

# Página de inicio (formulario)
@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# Ruta para procesar predicción
@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    buying: str = Form(...),
    maint: str = Form(...),
    doors: str = Form(...),
    persons: str = Form(...),
    lug_boot: str = Form(...),
    safety: str = Form(...)
):
    # Crear el DataFrame con los valores ingresados
    nuevo_auto = pd.DataFrame([{
        "buying": buying,
        "maint": maint,
        "doors": doors,
        "persons": persons,
        "lug_boot": lug_boot,
        "safety": safety
    }])

    # Codificar con el encoder entrenado
    nuevo_auto_encoded = encoder.transform(nuevo_auto)

    # Realizar la predicción
    prediccion = model.predict(nuevo_auto_encoded)[0]

    return templates.TemplateResponse("form.html", {
        "request": request,
        "result": f"Predicción: {prediccion}"
    })
