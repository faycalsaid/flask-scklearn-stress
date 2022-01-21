import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify
import pandas as pd
import joblib 

app = Flask(__name__)

# inputs
training_data = 'data/data.csv'

model_directory = 'model'
model_file_name = '%s/model.pkl' % model_directory

# These will be populated at training time
model_columns = None
rfc = None


@app.route('/predict', methods=['POST'])
def predict():
    if rfc:
        try:
            json_ = request.json
            print(json_)

            prediction = list(rfc.predict(json_))

            # Converting to int from int64
            return jsonify({"prediction": list(map(int, prediction))})

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return 'no model here'


@app.route('/train', methods=['GET'])
def train():
    from sklearn.model_selection import train_test_split
    # using random forest as an example
    # can do the training separately and just update the pickles
    df = pd.read_csv(training_data, sep=r'\t', engine='python')

    # Extract the columns with just the answers to the first 42 questions
    df2 = df.loc[:,::3]
    df3 = df2.iloc[:,0:42]

    # Now extracting the dataframe with personality identification questions
    print("Index of TIPI1 column is: " + str(df.columns.get_loc("TIPI1")))
    df4 = df.iloc[:,131:141]

    # Setting target and features for basic classfication model
    df4["TIPI4"] = df4["TIPI4"].replace([5, 4, 3, 2, 1], 0)
    df4["TIPI4"] = df4["TIPI4"].replace([7, 6], 1)
    target = df4["TIPI4"] 
    features = df3

    # Extracting the participants' background info and storing the responses in a separate dataframe.
    bkgd_features = df.loc[:, ["education", "urban", "gender", "engnat", "age", "religion", "orientation","race", "voted", "married", "familysize"]]

    # Merging the background info df with the target variable
    data = pd.concat([df3, bkgd_features, target], axis=1)
    #Drop na observations
    data_final=data.dropna(how='any')
    data_final_2 = data_final.loc[~((data_final['education'] == 0) | (data_final['urban'] == 0)|(data_final['gender'] == 0)|(data_final['engnat'] == 0)|(data_final['age'] == 0)|(data_final['religion'] == 0)|(data_final['orientation'] == 0)|(data_final['race'] == 0)|(data_final['voted'] == 0)|(data_final['married'] == 0)|(data_final['familysize'] == 0))]
    data_final_2.head()

    # Normalize the feature columns
    from sklearn import preprocessing
    from sklearn.preprocessing import MinMaxScaler
    min_max_scaler = preprocessing.MinMaxScaler()
    all_features_final = data_final_2.drop("TIPI4", axis = 1)
    target_final = data_final_2["TIPI4"]
    X_minmax = min_max_scaler.fit_transform(all_features_final)
    X_train, X_test, y_train, y_test = train_test_split(X_minmax, target_final, test_size=0.2, random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import mean_squared_error

    global rfc
    rfc = RandomForestClassifier()
    start = time.time()
    rfc.fit( X_train, y_train )
    y_pred = rfc.predict( X_test )
    erreur = mean_squared_error( y_test, y_pred )
    print("Random Forrest Score", rfc.score(X_minmax, target_final))
    print("Stress mean squared error", erreur )

    joblib.dump(rfc, model_file_name)

    message1 = 'Trained in %.5f seconds' % (time.time() - start)
    message2 = 'Model training score: %s' % rfc.score(X_minmax, target_final)
    return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2) 
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


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    try:
        rfc = joblib.load(model_file_name)
        print('model loaded')

    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        clf = None

    app.run(host='0.0.0.0', port=port, debug=True)
