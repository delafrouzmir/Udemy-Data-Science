from absenteeism import *

# creating the model using previous scaler and trained model
model = AbsenteeismModel(scaler_file = 'scaler', model_file = 'model')

# preprocess
model.load_and_clean (data_file = 'Absenteeism_new_data.csv')

# predicting the output and saving it in file
model_prediction = model.predicted_output()
print(model_prediction)
model_prediction.to_csv('Absenteeism_predictions.csv', index = False)