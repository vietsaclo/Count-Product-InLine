import cv2
from V_02 import CountProducts as cps
import json
import os

URL_FILE_CONFIG = './project.config.json'

CONFIG = {
    "URL_LOAD_VIDEO": "./Files/Videos/products_in_line.mp4",
    "WEIGHT_NAME": "./Files/configs/tiny_13_07_2022.weights",
    "CONF_NAME": "./Files/configs/tiny_13_07_2022.cfg",
    "CLASSES_NAMES": "./Files/configs/tiny_13_07_2022.txt",
    "CORLOR": [51, 255, 102],
    "LEFT_REGION": 130,
    "RIGHT_REGION": 440,
    "TOP_REGION": 120,
    "BOTTOM_REGION": 300,
    "COPYRIGHT": "DTNCKH",
    "COPYRIGHT_COLOR": [51, 255, 102],
    "COPYRIGHT_FONT_SIZE": 0.5
}


def fun_createConfigFile():
    with open(URL_FILE_CONFIG, 'w') as f:
        json.dump(CONFIG, f)
    print('----------------------------------------------------------------------')
    print('MAYBE THE APPLICATION IS NOT WORKING CORRECT, PLEASE EDIT CONFIG FILE!')
    print('----------------------------------------------------------------------')


def fun_loadConfigFile():
    with open(URL_FILE_CONFIG, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    # Create config file if not exists
    if not os.path.exists(URL_FILE_CONFIG):
        fun_createConfigFile()

    # Load config file
    CONFIG = fun_loadConfigFile()

    # Initial App
    countPros = cps.CountProducts(CONFIG)
    
    # Start App
    countPros.fun_startVideoAndCountObject(fps= 30, frame_show_name= "Count Products", 
      # pathSave= "Files/Videos/output.mp4"
    )