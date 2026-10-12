
import requests

# رابط سيرفر الـ Flask الخاص بك
url = "http://127.0.0.1:5001/predict" 

# بيانات تجريبية للمريض
payload = {
    "features": [2, 148, 85, 28, 120, 31.5, 0.627, 50]
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
