import pandas as pd



"""
The following script is used for the evaluation of BioBERT results. You need to clone github "dmis-lab/biobert" 
repository. This script is based on ner_detokenize.py in ./biocodes/ with some modifications.
The output of this script is the number of all mistakes that BioBERT makes and the corresponding dataframes with all
instances.

    output : dataframes with all different types of bioBERT mistakes
"""

def main():
    parser = argparse.ArgumentParser(description='bioBERT_error_analysis.py')
    parser.path_mapped_test_tokens('path_mapped_test_tokens', dest = 'path_mapped_test_tokens', deafult = None, type = str)
    parser.path_mapped_test_labels('path_mapped_test_labels', dest = 'path_mapped_test_labels', deafult = None, type = str)
    parser.path_predicted_test_tokens('path_predicted_test_tokens', dest = 'path_predicted_test_tokens', deafult = None, type = str)
    parser.path_predicted_test_labels('path_predicted_test_labels', dest = 'path_predicted_test_labels', deafult = None, type = str)
    parser.output_dir('output_dir', dest = 'output_dir', default = None, type = str)
    args = parser.parse_args()






    ### open the true Bert Tokens and Labels ###
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
            
            
            
    ### open the predicted Bert Tokens and Labels ### 
    """
    For the next two following path you use the output files of BioBERT: "token_test.txt" and "label_test.txt"
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
    
    
    ## Create a dataframe that carries bert true tokens, predicted tokens, true labels and predicted labels 
    df = pd.DataFrame(
        {'true_toks': bert_true['toks'],
         'pred_toks': bert_pred['toks'],
         'true_labels': bert_true['labels'],
         'pred_labels' : bert_pred['labels']
        })
    
    
    # Calcuate the actual number of wrong predictions
    df_common = df[df.true_labels == df.pred_labels]
    len_difference = len(df)-len(df_common)
    print("The number of wrong predictions is:"len_difference)
    df_different = df[df.true_labels != df.pred_labels]
    
    
    
    ### "O" cases ###
    df_o = df_different[df_different['true_labels']== 'O']
    print('The number of "O" mistakes is:'len(df_o))
    df_o.to_csv(f'{output_dir}/df_o.csv',  sep=',')
    
    
    # O -> I
    df_o_i = df_o[df_o['pred_labels']== 'I']
    print("The number of O to I mistakes is:" len(df_o_i))
    df_o_i.to_csv(f'{output_dir}/df_o_i.csv',  sep=',')
   
    
    # O -> B
    df_o_b = df_o[df_o['pred_labels']== 'B']
    print("The number of O to B mistakes is:" len(df_o_b))
    df_o_b.to_csv(f'{output_dir}/df_o_b.csv',  sep=',')
    
    
    
    ### "B" cases ###
    df_b = df_different[df_different['true_labels']== 'B']
    print("The number of B mistakes is:" len(df_b))
    df_b.to_csv(f'{output_dir}/df_b.csv',  sep=',')
    
    
    # B -> I
    df_b_i = df_b[df_b['pred_labels']== 'I']
    print("The number of B to I mistakes is:"len(df_b_i))
    df_b_i.to_csv(f'{output_dir}/df_b_i.csv',  sep=',')
    
    
    # B -> O
    df_b_o = df_b[df_b['pred_labels']== 'O']
    print("The number of B to O mistakes is:"len(df_b_o))
    df_b_o.to_csv(f'{output_dir}/df_b_o.csv',  sep=',')
    
    
    ### "I" cases ###
    df_i = df_different[df_different['true_labels']== 'I']
    print('The number of I mistakes is:'len(df_i))
    df_i.to_csv(f'{output_dir}/df_i.csv',  sep=',')
    
    
    # I -> B
    df_i_b = df_i[df_i['pred_labels']== 'B']
    print('The number of I to B mistakes is:'len(df_i_b))
    df_i_b.to_csv(f'{output_dir}/df_i_b.csv',  sep=',')
    
    # I -> O
    df_i_o = df_i[df_i['pred_labels']== 'O']
    print('The number of I to O mistakes is:'len(df_i_o))
    df_i_o.to_csv(f'{output_dir}/df_i_o.csv',  sep=',')
    
if __name__ == '__main__':
    main()