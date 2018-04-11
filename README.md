README

# 1. introduction
This is a solution to the exercise. This folder contains codes, data folders.
codes folder contains two python files, house_price_forecast.py and house_price_visualization.py.

house_price_forecast.py file loads data from the data folder, processes the data and predicts house
price for 2017. It generates a prediction.csv file in the data directory which will be used for
visualization by house_price_visualization.py.

house_price_visualization.py file is used to create a web service which can be used to
explore historical price and see the house price forecast in 2017.

data folder contains several data files such as house price, family data (family_data.csv)
and pre-generated prediction file (prediction.csv).
Family data contains the number of families and the average size of family in each county.

# 2. How to use
All functions and scripts (python and bash) have been tested on Mac (version 10.12.6),
python 2.7 environments. It should work in Linux environment as well without any modification.

Make sure pip and virtualenv tools have been installed in your system (Mac or Linux).
To install pip and virtualenv on Mac
    sudo easy_install pip
    pip install --upgrade pip
    pip install virtualenv

To install pip and virtualenv on linux (not tested)
    sudo apt-get install python-pip python-dev build-essential
    sudo pip install --upgrade pip
    sudo pip install --upgrade virtualenv

# 2.0 preparation
    virtualenv .virtenv
    source .virtenv/bin/activate

    # install python library dependencies
    pip install -r requirements.txt

# 2.1 run forecasting model (may skip this step and run visualization directly as
#   prediction.csv has been generated.)
    python codes/house_price_forecast.py
This script would generate a file (prediction.csv) in data folder which can be used for
visualization

# 2.2 run visualization web application
    bokeh serve codes/house_price_visualization.py
    # go to http://localhost:5006/house_price_visualization in a broswer

# or simply run
    chmod 744 run.sh
    ./run.sh
