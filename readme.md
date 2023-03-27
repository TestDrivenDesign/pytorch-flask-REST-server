project requires python 3.10 installed - https://www.python.org/downloads/

once installed:

install pip:   

pip install pip

install project requirements: 

pip install -r requirements.txt

to launch live debug server locally: 
in terminal from dev path run:

export FLASK_DEBUG=1
flask run

/upload endpoint takes post requests with image sent via multipart form with key of file.

issues and roadblocks:

error handling only for bad requests, when using try except block when calling model goes straight to exception need to remedy this.

currently calls to get_skin_prediction() return only a 'success' reponse confirming model is being called. have tried to implement return of classification from model but as output is a tensor it has to be converted. attempted using tensor.to but tensor too large need to fix.

model as trained so far is returning an accuracy of ~90% on classification of individual conditions and ~95% on malignancy in validation. HOWEVER- post validation testing is returning results no better than random. need to look at input transformations and confirm  they are the same as applied to training dataloader. further testing required once resolved.