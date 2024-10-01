import requests
import json
# Define la URL del API a donde se enviará el `prediction_data`
api_url = "http://bot.fidubit.co:3000/api/signals"  # Cambia esta URL a la de tu API PHP

# Función para enviar los datos de predicción al API
bearer_token = "Mfeqr6ufjabYGFPGGBObdr1tug1qKEkNHR76iWjkcyLjN3nrTpbvTxognQArQTQr6fpPJY0kRAYTxN+JGWCL+Q=="  # Reemplaza con tu token
# Eg: Datos de predicción ya calculados (prediction_data)
# '''prediction_data = {
#     "symbol": "EUR/USD",
#     "price": 1.1234,
#     "entry": 1.1200,
#     "stop_loss": 1.1150,
#     "tp1": 1.1300,
#     "tp2": 1.1350,
#     "tp3": 1.1400,
#     "date": "2024-09-11",
#     "trade_direction": "bullish",
# }'''

print('conenctadon a api')
# Función para enviar los datos de predicción al API con autenticación
def send_prediction_data_to_api(prediction_data):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {bearer_token}'  # Agregando el Bearer Token a los headers
    }
    try:
        response = requests.post(api_url, data=json.dumps(prediction_data), headers=headers)
        if response.status_code == 200:
            print(f"Datos enviados correctamente al API. Respuesta: {response.json()}")
        else:
            print(f"Error al enviar datos al API. Código de estado: {response.status_code}")
            print(f"Respuesta del API: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión o de solicitud: {e}")


