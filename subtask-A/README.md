# Subtask A Instructions
## Contact
Please contact Yitong Li at yitongli@umich.edu if you have any questions not answered in the README :)

## Running the code
- It is recommended that you use Python 3.8.7. We cannot promise that any other version will work. The Python version and cuda drivers used can be found in `main_job.sh`.
- Before running the code you must create a virtual environment and then install the dependencies listed below.
- The script can be run as `python3 subtask1.py`.
- **It is strongly recommended to run this program on a GPU, it will take hours or days otherwise. main_job.sh includes the shell script required to run the program on GreatLakes slurm.** 

## Evaluating the result
Once you run`python3 subtask1.py`, it will print out the accuracy and the Macro F1 score.
## All dependencies (default version with pip is fine or you can follow the ones below)
pip install pandas==1.3.5
numpy==1.21.4 
nltk==3.6.5
unidecode==1.3.2 
contractions==0.0.58 
sklearn==0.0 
bs4==0.0.1  
sentence_transformers==2.1.0 
xmltodict==0.12.0