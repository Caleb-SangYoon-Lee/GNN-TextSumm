This source codes implemented the paper "Augmenting Abstractive Text Summarization with Graph Neural Network and Transfoemr".
You can download text summarization data for training deep learning model in AI-Hub. (refer to https://aihub.or.kr/aidata/8054)

There's need to process text data as tokeninzing and to convert IDs for those tokens
 - refer to prep-data-01-sent2tokens.py to tokenization
 - refer to prep-data-02-tokens2ids.py to convert tokens to IDs
 (KoElectraBert Tokenizer was used to tokenize and convert to IDs)

Data and model directory locatioins are specified in configuratoin file such as config/config-base-v001.json

To train model you can run shell script as follows

  $ ./train.sh

You can modify config. file as follows to train with pure transformer to compare with GNN encoder.
 - "use_transformer_encoder" : true,
 - "tns_only" : true,
 - "tns_only_d_model": 512,


You can evaluate model with eval.sh after installing Rouge package like as follows.

 $ pip install pytorch-ignite
 $ ./eval.sh
