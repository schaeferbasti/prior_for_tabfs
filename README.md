# Prior Test for Feature Selection with a Tabular Foundation Model

This project is the code basis to study the design of an prior for feature selection based on metalearning and the work of nanoPFN.


## Pretraining
To pretrain nanoPFN, you need to download the prior data dump at first:
mkdir -p data/prior
curl -L -o data/prior/300k_150x5_2.h5 "https://ndownloader.figshare.com/files/58932628?private_link=63fc1ada93e42e388e63"

Secondly, run_pretraining_pipeline.py can be executed.

## Inference
In order to get the predictions, execute run_prediction_pipeline.py.
