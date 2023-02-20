# This code serves the purpose to map from libraries from the parent folder that
# do ICD9 code transformations

import sys
import os

ICD9_DIR = 'icd9'
JSON_NAME = 'codes.json'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', ICD9_DIR))

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ICDMappings import ICDMappings #Simao

icdmap = ICDMappings(relative_dir="../")


from icd9_base import ICD9 # https://github.com/sirrice/icd9

tree = ICD9(f'../{ICD9_DIR}/{JSON_NAME}')

# e.g. '427.3.1' to '427.31' to match how keys are stored in the .json file
def _reaarange_icd9_code(code):
    splitted = code.split(".")
    return ".".join(splitted[:2])+splitted[2]


def _get_description(code):
    try:
        if len(code.split(".")) == 3:
            code = _reaarange_icd9_code(code)
        return tree.find(code).description
    except:
        raise ValueError("CODE FAILURE", code)

# get ICD9 description
def get_ICD9_description(code):
    c = code.split("_")[1]
    return _get_description(c.zfill(3))
    
# get CSS description
def get_css_description(code):
    return icdmap.lookup('ccstodescription', code)
