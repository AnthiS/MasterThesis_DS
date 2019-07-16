
"""
The following script is used for the evaluation of BioBERT results. You need to clone github "dmis-lab/biobert" 
repository.
This script is based on ner_detokenize.py in ./biocodes/ with some modifications.
The output needs to be used as input in the following command which calculates the entity-level model performance (precision, recall, F1-score):
"perl biocodes/conlleval.pl < /tmp/bioner/entity.txt". The output after running this command can be
saved in a txt/csv file.
The output file (entity.txt) is used also to calculate the token-level model performance:(see corresponding script, bioBERT_tokenlevel_evaluation.py). For an example see "BertResults_#" folders.

    path_mapped_test_tokens : the path to test tokens mapped to WordPiece tokens (use them for evaluation)
    path_mapped_test_labels : the path to test labels mapped to WordPiece tokens (use them for evaluation)
    path_predicted_test_tokens : path to "token_test.txt" file (BioBERT output, for an example see "BertResults_#" folders)
    path_predicted_test_labels : path to "label_test.txt" file (BioBERT output, for an example see "BertResults_#" folders)

"""



def main():
    parser = argparse.ArgumentParser(description='bioBERT_entitylevel_evaluation.py')
    parser.path_mapped_test_tokens('path_mapped_test_tokens', dest = 'path_mapped_test_tokens', deafult = None, type = str)
    parser.path_mapped_test_labels('path_mapped_test_labels', dest = 'path_mapped_test_labels', deafult = None, type = str)
    parser.path_predicted_test_tokens('path_predicted_test_tokens', dest = 'path_predicted_test_tokens', deafult = None, type = str)
    parser.path_predicted_test_labels('path_predicted_test_labels', dest = 'path_predicted_test_labels', deafult = None, type = str)
    parser.output_dir('output_dir', dest = 'output_dir', deafult = None, type = str)
    args = parser.parse_args()
    
    


    ### open the true BioBert Tokens and Labels ###
    golden_path_tokens = f"{path_mapped_test_tokens}/convtokens_forbert.txt"
    golden_path_labels = f"{path_mapped_test_labels}/convlabels_forbert.txt"
    
    
    # read true
    true = dict({'toks':[], 'labels':[]}) # dictionary for predicted tokens and labels.
    with open(golden_path_tokens,'r') as in_: 
        for line in in_:
            line = line.strip()
            true['toks'].append(line)
                
    with open(golden_path_labels,'r') as in_: 
        for line in in_:
            line = line.strip()
            if line in ['[CLS]','[SEP]', 'X']: # replace non-text tokens with O. This will not be evaluated.
                true['labels'].append('O')
                continue
            true['labels'].append(line)
                
    if (len(true['toks']) != len(true['labels'])): # Sanity check
        print("Error! : len(pred['toks']) != len(pred['labels']) : Please report us")
        raise
        
    bert_true = dict({'toks':[], 'labels':[]})
    for t, l in zip(true['toks'],true['labels']):
        if t in ['[CLS]','[SEP]']: # non-text tokens will not be evaluated.
            continue
        elif t[:2] == '##': # if it is a piece of a word (broken by Word Piece tokenizer)
            bert_true['toks'][-1] = bert_true['toks'][-1]+t[2:] # append pieces
        else:
            bert_true['toks'].append(t)
            bert_true['labels'].append(l)
            
            
            
    ### open the predicted BioBert Tokens and Labels ### 
    """
    For the next two following paths you use the output files of BioBERT: "token_test.txt" and "label_test.txt"
    """
    pred_token_test_path = f"{path_predicted_test_tokens}/token_test.txt"
    pred_label_test_path = f"{path_predicted_test_labels}/label_test.txt"
    
    
    # read predicted
    pred = dict({'toks':[], 'labels':[]}) # dictionary for predicted tokens and labels.
    with open(pred_token_test_path,'r') as in_: #'token_test.txt'
        for line in in_:
            line = line.strip()
            pred['toks'].append(line)
                
    with open(pred_label_test_path,'r') as in_: #'label_test_3_epoch.txt'
        for line in in_:
            line = line.strip()
            if line in ['[CLS]','[SEP]', 'X']: # replace non-text tokens with O. This will not be evaluated.
                pred['labels'].append('O')
                continue
            pred['labels'].append(line)
                
    if (len(pred['toks']) != len(pred['labels'])): # Sanity check
        print("Error! : len(pred['toks']) != len(pred['labels']) : Please report us")
        raise
        
    bert_pred = dict({'toks':[], 'labels':[]})
    for t, l in zip(pred['toks'],pred['labels']):
        if t in ['[CLS]','[SEP]']: # non-text tokens will not be evaluated.
            continue
        elif t[:2] == '##': # if it is a piece of a word (broken by Word Piece tokenizer)
            bert_pred['toks'][-1] = bert_pred['toks'][-1]+t[2:] # append pieces
        else:
            bert_pred['toks'].append(t)
            bert_pred['labels'].append(l)
        
    
    a = len(bert_pred['toks'])
    b = len(bert_pred['labels'])
    bert_true['toks'] = bert_true['toks'][:a]
    bert_true['labels'] = bert_true['labels'][:b]
    
        
    
    if (len(bert_pred['toks']) != len(bert_pred['labels'])): # Sanity check
        print("Error! : len(bert_pred['toks']) != len(bert_pred['labels']) : Please report us")
        raise
    if (len(bert_true['labels']) != len(bert_pred['labels'])): # Sanity check
        print(len(bert_true['labels']), len(bert_pred['labels']))
        print("Error! : len(bert_true['labels']) != len(bert_pred['labels']) : Please report us")
        raise
        
    
    print("The length of the true tokens is:"len(bert_true['toks']))
    print("The length of the true labels is:"len(bert_true['labels']))
    print("The length of the predicted tokens is:"len(bert_pred['toks']))
    print("The length of the predicted labels is:"len(bert_pred['labels']))
    print("All above lengths should be the same! Otherwise go back and check individually.")
    
    # Save the output in a txt file to use for the evaluation. For the evaluation run the 
    with open(f'{output_dir}/entity.txt', 'w') as out_:
        idx=0
        for ans_t in bert_true['toks']:
            if ans_t=='[SEP]':
                out_.write("\n")
            else :
                out_.write("%s %s-ADR %s-ADR\n"%(bert_pred['toks'][idx], bert_true['labels'][idx], bert_pred['labels'][idx]))
                idx+=1
            
            
if __name__ == '__main__':
    main()