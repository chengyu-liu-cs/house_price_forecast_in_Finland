#!/usr/bin/env bash
# author : Chengyu Liu

# make sure pip has been installed already
# install virtual virtualenv on Mac
pip install --upgrade pip
pip install virtualenv

# create virtual environment in current directory (.virtenv)
virtualenv .virtenv
source .virtenv/bin/activate

# install python library dependencies
pip install -r requirements.txt

# Running forecast model
python codes/house_price_forecast.py

# Running the application
bokeh serve codes/house_price_visualization.py
