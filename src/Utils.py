# -*- coding: utf-8 -*-
# @date: 6.05.2024
# @author: ikbal
# @file: Utils.py

import json

def get_json_config(config_path):
    """
    Get json config file
    :param config_path:
    :return:
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config