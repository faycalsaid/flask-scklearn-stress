import sys
import os
import shutil
import time
import traceback
import json

from flask import Flask, request, jsonify
import pandas as pd
import joblib 

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# inputs
training_data = 'data/data.csv'

model_directory = 'model'
model_file_name_tipi_1 = '%s/model_tipi_1.pkl' % model_directory
model_file_name_tipi_4 = '%s/model_tipi_4.pkl' % model_directory

# These will be populated at training time
rfc = None
rfc_tipi_1 = None
rfc_tipi_4 = None


@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    print(json_)

    predictions = {}
 

    tipi_list = ['TIPI1','TIPI2','TIPI3','TIPI4','TIPI5','TIPI6','TIPI7','TIPI8','TIPI9','TIPI10']
    for i in tipi_list:  
        try:
            model_file_name =  f"{model_directory}/model_{i}.pkl"
            rfc = joblib.load(model_file_name)
            print(model_file_name + ' loaded')

            prediction = rfc.predict(json_)
            predictions[i] = int(prediction[0])

        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})

    # Serializing json  
    #predictions = json.dumps(predictions, indent = 4) 
    print(predictions)

    return jsonify({"predictions": predictions})

@app.route('/train', methods=['GET'])
def train():
    start = time.time()
    train()
    return_message = 'Trained in %.5f seconds' % (time.time() - start)
    return return_message


@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree('model')
        os.makedirs(model_directory)
        return 'Model wiped'

    except Exception as e:
        print(str(e))
        return 'Could not remove and recreate the model directory'

def train():
    df = pd.read_csv(training_data, sep=r'\t', engine='python')

    # Extract the columns with just the answers to the first 42 questions
    df2 = df.loc[:,::3]
    df3 = df2.iloc[:,0:42]

    # Now extracting the dataframe with personality identification questions
    # Index of TIPI1 column is 131
    df4 = df.iloc[:,131:141]
    tipi_list = df4.columns.values.tolist()

    # Extracting the participants' background info and storing the responses in a separate dataframe.
    bkgd_features = df.loc[:, ["education", "urban", "gender", "engnat", "age", "religion", "orientation","race", "voted", "married", "familysize"]]
    
    # Using for loop on the tipi list ['TIPI1','TIPI2','TIPI3','TIPI4','TIPI5','TIPI6','TIPI7','TIPI8','TIPI9','TIPI10']
    for i in tipi_list:
        print("START "+ i +" Model Train")
        # Setting target and features for basic classfication model
        df4[i] = df4[i].replace([5, 4, 3, 2, 1], 0)
        df4[i] = df4[i].replace([7, 6], 1)
        target = df4[i] 

        # Merging the background info df with the target variable
        data = pd.concat([df3, bkgd_features, target], axis=1)
        #Drop na observations
        data_final=data.dropna(how='any')
        data_final_2 = data_final.loc[~((data_final['education'] == 0) | (data_final['urban'] == 0)|(data_final['gender'] == 0)|(data_final['engnat'] == 0)|(data_final['age'] == 0)|(data_final['religion'] == 0)|(data_final['orientation'] == 0)|(data_final['race'] == 0)|(data_final['voted'] == 0)|(data_final['married'] == 0)|(data_final['familysize'] == 0))]
        data_final_2.head()

        # Normalize the feature columns
        min_max_scaler = preprocessing.MinMaxScaler()
        all_features_final = data_final_2.drop(i, axis = 1)
        target_final = data_final_2[i]
        X_minmax = min_max_scaler.fit_transform(all_features_final)
        X_train, X_test, y_train, y_test = train_test_split(X_minmax, target_final, test_size=0.2, random_state=42)

        global rfc
        rfc = RandomForestClassifier()
        start = time.time()
        rfc.fit( X_train, y_train )
        y_pred = rfc.predict( X_test )
        erreur = mean_squared_error( y_test, y_pred )
        print("Random Forrest Score", rfc.score(X_minmax, target_final))
        print("Stress mean squared error", erreur )

        model_file_name =  f"{model_directory}/model_{i}.pkl"
        print(model_file_name)
        joblib.dump(rfc, model_file_name)
        print(i +" Model Dumped")
        print("END "+ i +" Model Train")

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    tipi_list = ['TIPI1','TIPI2','TIPI3','TIPI4','TIPI5','TIPI6','TIPI7','TIPI8','TIPI9','TIPI10']
    for i in tipi_list:  
        try:
            model_file_name =  f"{model_directory}/model_{i}.pkl"
            rfc = joblib.load(model_file_name)
            print(model_file_name + ' loaded')

        except Exception as e:
            print('No '+ i +' model here')
            print('Train' + i + ' first')
            print(str(e))
            clf = None

    app.run(host='0.0.0.0', port=port, debug=True)

