import requests

resp = requests.post(
    "http://127.0.0.1:61616/predict",
    json={"text": "OH Dios mío! Amo ser homosexual"}
)
print(resp.json())