for f in main_ETT main_istanbul main_demand_forecast main_low_noise main_electricity-day main_medium_noise main_electricity-week main_syntethic_1 main_gas main_syntethic_2 main_high_noise
do
python3 $f.py 
done