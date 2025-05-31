import joblib
import pandas as pd

# Cargar el modelo y el encoder entrenados
encoder = joblib.load('encoder.pkl')
knn = joblib.load('modelo_knn.pkl')

# Crear nuevo auto
nuevo_auto = pd.DataFrame([{
    'buying': 'low',
    'maint': 'med',
    'doors': '4',
    'persons': 'more',
    'lug_boot': 'big',
    'safety': 'high'
}])

# Codificar y predecir
nuevo_auto_encoded = encoder.transform(nuevo_auto)
prediccion = knn.predict(nuevo_auto_encoded)

print("Clase predicha:", prediccion[0])