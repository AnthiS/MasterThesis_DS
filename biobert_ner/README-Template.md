# Project Title

This repo is about biomedical Named entity recognition (specifically, adverse drug reactions). The report of this project can be found here: http://www.scriptiesonline.uba.uva.nl/document/673279. 
Three traditional methods (dictionary-based, CRFs and BiLSTM) are examined and compared with a transfer learning method which uses BioBERT model. Our transfer learning method outperformed all thraditional methods using three different biomedical datasets (ADE corpus, TAC2017 corpus and Elsevier's gold set).

## Getting Started

The code regarding the traditional methods (traditional_methods.py) is based on Elsevier's project which is not possible to be published. However, explanation is given in the paper and the results can be easily reproduced. The transfer learning method (BioBERT) is based on the https://github.com/dmis-lab/biobert and https://github.com/google-research/bert repos. Cloning of these repos is mantatory. Fine-tuning of BioBERT for the NER task was followed as described in the above repo. All parameters set as default and the max_seq_length=150. Some raw data containing documents with plain and annotated text are given in the TAC2017_rawdata_sample folder. The TAC2017_BioBERT_examples folder contains output examples for the splits(i.e. TAC_1), the BioBERT output (i.e. BertResults_1) and the mapped labels to WordPiece tokens of BioBERT (i.e. Map_1). 


## Authors

* *Anthi Symeonidou*, an_syme@hotmail.com
* *Slava Sazonau*, s.sazonau@elsevier.com
* *Paul Gorth*, p.groth@uva.nl

