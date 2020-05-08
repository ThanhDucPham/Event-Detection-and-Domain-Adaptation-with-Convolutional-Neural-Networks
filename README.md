# Event-Detection-and-Domain-Adaptation-with-Convolutional-Neural-Networks
code re-implement model in paper Event Detection and Domain Adaptation with Convolutional Neural Networks

# Preprocessing data:
- Transform format data from ACE format to json format from here: https://github.com/nlpcl-lab/ace2005-preprocessing.git
- build vocabularies : re-build event vocab(34-classes), entity vocab( BIO format include pad label, id=0), word vocab( include pad token with id=0, unknow token with id =1) or just use vocabs from data folder
- use load_data_json and window_encode functions in utils.py to build data that will be fed into model

# Train model:
- hyperparameters of model are stored in Config class
- just run file runs.py

 # References:
Detection and Domain Adaptation with Convolutional Neural Networks,
Thien Huu Nguyen, Ralph Grishman<br>

