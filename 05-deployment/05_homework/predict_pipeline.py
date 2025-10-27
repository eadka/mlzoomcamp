import pickle


#open the pipeline file
with open('pipeline_v1.bin','rb') as f_in:
    pipeline = pickle.load(f_in)


record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

pred = pipeline.predict_proba([record])

print("The probablity: ", pred[0][1])