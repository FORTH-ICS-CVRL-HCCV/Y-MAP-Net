
"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""

import os
#-------------------------------------------------------------------------------
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
#-------------------------------------------------------------------------------
def read_json_file(file_path):
    import json
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file '{file_path}'.")
    except Exception as e:
        print(f"Error: {e}")
#-------------------------------------------------------------------------------
def checkIfPathExists(filename):
    return os.path.exists(filename)
#-------------------------------------------------------------------------------
def checkIfPathIsDirectory(filename):
    return os.path.isdir(filename) 
#-------------------------------------------------------------------------------
def checkIfFileExists(filename):
    return os.path.isfile(filename)
#-------------------------------------------------------------------------------
def convert_bytes(num):
    #This function will convert bytes to MB.... GB... strings
    step_unit = 1000.0 #1024 bad the size
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < step_unit:
            return "%3.1f %s" % (num, x)
        num /= step_unit
#-------------------------------------------------------------------------------
def inrange(min_value, value, max_value):
    return max(min_value, min(value, max_value))
#-------------------------------------------------------------------------------
