import requests

url = 'http://localhost:9696/predict'

client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=client)
convert = response.json()

print("Response: ", convert)

if convert['convert']>= 0.5:
    print('Send email with promo')
else:
    print('dont do anything')