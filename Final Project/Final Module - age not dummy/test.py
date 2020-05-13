from absenteeism import *

model = AbsenteeismModel(scaler_file = 'scaler', model_file = 'model')

model.load_and_clean (data_file = 'Absenteeism_new_data.csv')

model_prediction = model.predicted_output()
print(model_prediction)
model_prediction.to_csv('Absenteeism_predictions.csv', index = False)