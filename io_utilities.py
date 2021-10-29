#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def load_config(main_window,filename='params.txt'):
    
    list_prefix = []
    list_name   = []
    list_types  = []
    list_values = []
    
    known_types = {'int': int,'float': float,'str': str}
    
    # Load txt file
    with open(filename, 'r') as fileobj:
        for row in fileobj:
            temp = row.rstrip('\n').split(' ')
            list_prefix.append(temp[0])
            list_name.append(temp[1])
            list_types.append(known_types[temp[2]])
            list_values.append(temp[3])
    
    # Load config into MainWindow Instance
    for i in range(len(list_prefix)):
        # Main (spinboxes)
        if list_prefix[i] == 'Main':
            main_window.spinboxes[list_name[i]].setValue(list_types[i](list_values[i]))
        # UnitType (checkbox)
        elif list_prefix[i] == 'UnitType':
            main_window.params[list_name[i]] = list_types[i](list_values[i])
            main_window.unit_types[main_window.params[list_name[i]]]['checkbox'].setChecked(True)
            main_window.update_checkbox_units()
        # Slider
        elif list_prefix[i] == 'Slider':
            main_window.sliders[list_name[i]].setValue(list_types[i](list_values[i]))
        # SubSlider
        elif list_prefix[i] == 'SubSlider':
            main_window.sub_sliders[list_name[i]].setValue(list_types[i](list_values[i]))
    
    return main_window


def save_config(main_window,filename='params.txt'):
    list_prefix = []
    list_name   = []
    list_types  = []
    list_values = []
    list_save   = []
    
    # Main (spinboxes)
    for key in main_window.spinboxes.keys():
        list_prefix.append('Main')
        list_name.append(key)
        list_types.append(type(main_window.spinboxes[key].value()).__name__)
        list_values.append(main_window.spinboxes[key].value())
    # UnitType (checkbox)
    list_prefix.append('UnitType')
    list_name.append('unit_type')
    list_types.append(type(main_window.params['unit_type']).__name__)
    list_values.append(main_window.params['unit_type'])
    # Slider
    for key in main_window.sliders.keys():
        list_prefix.append('Slider')
        list_name.append(key)
        list_types.append(type(main_window.sliders[key].value()).__name__)
        list_values.append(main_window.sliders[key].value())
    # SubSlider
    for key in main_window.sub_sliders.keys():
        list_prefix.append('SubSlider')
        list_name.append(key)
        list_types.append(type(main_window.sub_sliders[key].value()).__name__)
        list_values.append(main_window.sub_sliders[key].value())
    
    # Format the txt file
    for i in range(len(list_name)):
        list_save.append(f'{list_prefix[i]} {list_name[i]} {list_types[i]} {list_values[i]}')
        
    # Save as txt file
    f = open(filename,'w')
    np.savetxt(f,list_save,fmt="%s")
    f.close()
