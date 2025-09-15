# MsSTCN
How to use

The model is validated on two datasets.

Wind power forecasting

This dataset is from https://aml.engr.tamu.edu/book-dswe/dswe-datasets/. The data used here is Wind Spatio-Temporal Dataset2. Download data, put it into the './data' folder and rename it to 'wind_power.csv'. Then, run following

python train.py --name wind_power --epoch 80 --batch_size 20000 --lr 0.001 --k 5 --n_turbines 200

Wind speed forecasting

The model performance on wind speed forecasting is validated on NREL WIND dataset (https://www.nrel.gov/wind/data-tools.html). We select one wind farm with 100 turbines from Wyoming. To get data, first run

python getNRELdata.py

Then run

python train.py --name wind_speed --epoch 80 --batch_size 20000 --lr 0.001 --k 9 --n_turbines 100
