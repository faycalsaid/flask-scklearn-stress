
### Dependencies
- scikit-learn
- Flask
- pandas
- numpy
- joblib

```
pip install -r requirements.txt --user
```

### Running API

#### First Time
Go to root folder 

```
pip install virtualenv
```

```
python -m virtualenv venv 
```

```
venv\Scripts\activate.bat
```

```
pip install -r requirements.txt
```

```
python main.py <port>
```

#### Usual
Go to root folder 

```
Scripts\activate.bat 
```

```
python main.py <port>
```

# Endpoints
### /predict (POST)
Returns an array of predictions given a JSON object representing independent variables. Here's a sample input:
```
[
    [4,4,2,4,4,4,4,4,2,1,4,4,4,4,4,4,3,4,3,3,1,4,4,4,4,4,4,3,4,2,4,4,2,3,4,4,1,2,4,3,4,4,2,3,2,2,16,12,1,10,2,1,2]
]
```

and sample output:
```
{"prediction": [1]}
```


### /train (GET)
Trains the model. This is currently hard-coded to be a random forest model that is run on a subset of columns of the stress dataset.

### /wipe (GET)
Removes the trained model.
