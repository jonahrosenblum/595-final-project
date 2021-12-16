# Subtask B Instructions
## Contact
Please contact Jonah Rosenblum at jonaher@umich.edu if you have any questions not answered in the README :)

## Running the code
- It is recommended that you use Python 3.8.7. We cannot promise that any other version will work. The Python version and cuda drivers used can be found in `main_job.sh`.
- Before running the code you must create a virtual environment with `python3 -m venv env` and then install the dependencies listed below.
- The script can be run as `python3 subtask2.py` with a couple of flags options.
- With no flags, the script will simply train and evaluate on the SemEval dataset.
- With the `--boolq` or `-b` flag it will pretrain on the BoolQ dataset.
- With the `--multinli` of `-m` flag it will pretrain on the MultiNLI dataset.
- **It is strongly recommended to run this program on a GPU, it will take hours or days otherwise. main_job.sh includes the shell script required to run the program on GreatLakes slurm.** 

## Evaluating the result
- Once you run the script it will print out its final answers in the form `QID   Yes/No/Unsure`. Copy and paste this output to a text file, then run `./final-scorer.pl correct-yn-answers.txt YOUR_FILE` to see the final result.

## All dependencies
pip3 install absl-py==1.0.0
aiohttp==3.7.4.post0
appdirs==1.4.4
astunparse==1.6.3
async-timeout==3.0.1
attrs==19.3.0
beautifulsoup4==4.10.0
cachetools==4.2.4
certifi==2019.11.28
chardet==3.0.4
click==8.0.3
cycler==0.10.0
datasets==1.14.0
decorator==4.4.1
dill==0.3.4
distlib==0.3.1
filelock==3.3.1
flatbuffers==2.0
fsspec==2021.10.1
future==0.18.2
gast==0.4.0
gdown==4.2.0
google-auth==2.3.3
google-auth-oauthlib==0.4.6
google-pasta==0.2.0
grpcio==1.42.0
h5py==3.6.0
huggingface-hub==0.0.19
idna==2.8
importlib-metadata==4.8.2
iniconfig==1.1.1
joblib==1.1.0
keras==2.7.0
Keras-Preprocessing==1.1.2
kiwisolver==1.1.0
libclang==12.0.0
Markdown==3.3.6
matplotlib==3.1.3
more-itertools==8.2.0
multidict==5.2.0
multiprocess==0.70.12.2
networkx==2.4
numpy==1.18.1
oauthlib==3.1.1
opt-einsum==3.3.0
packaging==21.0
pandas==1.3.4
Pillow==8.4.0
pluggy==0.13.1
protobuf==3.19.0
py==1.8.1
pyarrow==5.0.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pybind11==2.4.3
pyparsing==2.4.6
PySocks==1.7.1
pytest==5.3.5
python-dateutil==2.8.1
pytz==2021.3
PyYAML==6.0
regex==2021.10.23
requests==2.22.0
requests-oauthlib==1.3.0
rsa==4.8
sacremoses==0.0.46
scikit-learn==1.0.1
scipy==1.4.1
sentencepiece==0.1.96
six==1.14.0
sklearn==0.0
soupsieve==2.2.1
tensorboard==2.7.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
tensorflow==2.7.0
tensorflow-estimator==2.7.0
tensorflow-io-gcs-filesystem==0.22.0
termcolor==1.1.0
threadpoolctl==3.0.0
tokenizers==0.10.3
toml==0.10.2
torch==1.10.0+cu113
torchaudio==0.10.0+cu113
torchvision==0.11.1+cu113
tqdm==4.62.3
transformers==4.11.3
typing-extensions==3.10.0.2
urllib3==1.25.8
virtualenv==20.4.2
wcwidth==0.1.8
Werkzeug==2.0.2
wrapt==1.13.3
xmltodict==0.12.0
xxhash==2.0.2
yarl==1.7.0
zipp==3.6.0