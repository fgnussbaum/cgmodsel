# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 18:01:36 2021

@author: Frank
"""

import json
#data = json.load("mscoco.json")


# read file
with open('mscoco/mscoco.json', 'r') as myfile:
    data=myfile.read()
    
obj = json.loads(data)

mlc_states = obj['experimentdata']['MLC_max_disc_states']
print(len(mlc_states))