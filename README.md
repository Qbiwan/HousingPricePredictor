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



| Features      | Description   | Median Value   | 
| ------------- |:-------------|:-------------:|
| longitude     | A measure of how far west a house is; a higher value is farther west | -118|
| latitude      | A measure of how far north a house is; a higher value is farther north     |34 |  
| housingMedianAge | Median age of a house within a block; a lower number is a newer building      | 29|
| totalRooms     | Total number of rooms within a block | 2127 |
| totalBedrooms     | Total number of bedrooms within a block     | 435|  
| population | Total number of people residing within a block      | 1166|  
| households     | Total number of households, a group of people residing within a home unit, for a block |409 | 
| medianIncome      | Median income for households within a block of houses (measured in tens of thousands of US Dollars)     | 3.5 |
| medianHouseValue | Median house value for households within a block (measured in US Dollars)     |179700 |   

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



