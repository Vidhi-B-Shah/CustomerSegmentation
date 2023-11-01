from setuptools import find_packages,setup
from typing import List

e_dot = '-e .'

def install_requirements(file_path:str)->List[str]:
    '''
    Installs all the requirements mentioned.
    '''
    requirements = []
    with open(file_path) as file:
         requirements = file.readlines()
         requirements = [req.replace("\n"," ") for req in requirements]
         
         
         if e_dot in requirements:
             requirements.remove(e_dot)




setup(
    name = 'Customer Segmentation Project',
    version= '0.0.1',
    description='Analyzing the content of an E-commerce database',
    author='Vidhi Shah' ,
    packages= find_packages(),
    install_requires=install_requirements('requirements.txt')
)