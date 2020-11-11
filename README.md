# Housing Price Predictor

## Directory

```
Housing_Price_Predictor
├── model                                         
│     ├── model.h5                               <- trained keras model  
│     └── scaler.pkl                             <- sklearn standard scaler
├── Demo                                         
│     └── Demo.gif 
├── templates                               
│     └── main.html
├── app.py                                  
├── train.py                                
├── Procfile                              
├── environment.yml                               
├── requirements.txt                               
└── README.md
```
## Demo

![Demo](Demo/Demo.gif)


A live version deployed on Heroku can be found [here](https://california-housing-predict.herokuapp.com/).

## Data

* The [`California Housing Dataset`](https://scikit-learn.org/stable/datasets/index.html#california-housing-dataset) from sklearn 
contains only numerical features, as the ocean_proximity feature is removed. There is no missing value. It has 8 continuous variables and a continuous target.



## Installation

Create the virtual environent using conda. 

```bash
$ conda env create -f environment.yml
$ conda activate Housing_Price_Predictor

# then run the python scripts
$ cd src
$ python train.py
$ python app.py    # serve flask app
```



