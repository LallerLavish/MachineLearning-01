from setuptools import setup,find_packages
from typing import List

Initializer='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    going to fetch data from requirements.txt file 
    '''
    requiremets=[]
    with open(file_path) as file_obj:
        requiremets=file_obj.readlines()
        requiremets=[req.replace('\n','') for req in requiremets]
        
        if Initializer in requiremets:
            requiremets.remove(Initializer)
        
    return requiremets

setup(
name='Generalised',
version='0.0.0.1',
author='Lavish Laller',
author_email='Apna@2024',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)