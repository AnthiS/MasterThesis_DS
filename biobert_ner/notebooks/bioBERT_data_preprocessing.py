from nerds.input.brat import BratInput
from nerds.dataset.clean import retain_annotations, resolve_overlaps, clean_annotated_documents
from nerds.dataset.convert import transform_annotated_documents_to_bio_format
from nerds.util.nlp import tokens_to_pos_tags   
import pandas as pd
import os
import numpy as np
from itertools import repeat


"""
This script takes as input one data split (collection of train and test set documents) and prepares the data for BioBERT fine-tuning. 
The output is four tsv files (train, train-eval, eval, test) that are the input for BioBERT. Attention! Adjust their size accordingly. See below. For an example see "TAC_#" folders.
Another output is two txt files which have the mapped test tokens and labels (from regexp tokens/labels to WordPiece tokens/labels of test set). For an example see "Map1" folders.

    path_train : the path to the collection of training documents (after the split)
    path_test : the path to the collection of test documents (after the split)
    output_dir_train : the path to save training examples for BioBERT
    output_dir_train_eval : the path to save training-eval examples for BioBERT
    output_dir_eval : the path to save eval examples for BioBERT
    output_dir_test : the path to save test examples for BioBERT
    path_google_bert : the path to the cloned "https://github.com/google-research/bert" repository
    output_dir_mapping_test_tokens : the path to save test tokens mapped to WordPiece tokens (use them for evaluation)
    output_dir_mapping_test_labels : the path to save test labels mapped to WordPiece tokens (use them for evaluation)
    
"""
def main():
    parser = argparse.ArgumentParser(description='bioBERT_data_preprocessing.py')
    parser.path_train('path_train', dest = 'path_train', deafult = None, type = str)
    parser.path_test('path_test', dest = 'path_test', deafult = None, type = str)
    parser.label('label', dest = 'label', default = None, type = str)
    parser.output_dir_train('output_dir_train', dest = 'output_dir_train', deafult = None, type = str)
    parser.output_dir_train_eval('output_dir_train_eval', dest = 'output_dir_train_eval', deafult = None, type = str)
    parser.output_dir_eval('output_dir_eval', dest = 'output_dir_eval', deafult = None, type = str)
    parser.output_dir_test('output_dir_test', dest = 'output_dir_test', deafult = None, type = str)
    parser.path_google_bert('path_google_bert', dest = 'path_google_bert', deafult = None, type = str)
    parser.output_dir_mapping_tokens('output_dir_mapping_tokens', dest = 'output_dir_mapping_tokens', deafult = None, type = str)
    parser.output_dir_mapping_labels('output_dir_mapping_labels', dest = 'output_dir_mapping_labels', deafult = None, type = str)
    args = parser.parse_args()


    data_path_train = path_train
    data_path_test = path_test
    bi_train = BratInput(data_path_train).transform()
    bi_test = BratInput(data_path_test).transform()
    
    # Clean the data (train)
    data_train = retain_annotations(bi_train, label)
    clean_data_train = clean_annotated_documents(data_train)
    non_overlap_data_train = resolve_overlaps(clean_data_train)
    
    # Clean the data (test)
    data_test = retain_annotations(bi_test, label)
    clean_data_test = clean_annotated_documents(data_test)
    non_overlap_data_test = resolve_overlaps(clean_data_test)
    
    # Transform to BIO format
    bio_train = transform_annotated_documents_to_bio_format(non_overlap_data_train, entity_labels=label)
    bio_test = transform_annotated_documents_to_bio_format(non_overlap_data_test, entity_labels=label)
    
    
    ######################################
    ### Train, Eval and train-eval set ###
    enum1 = []
    enum2 = []
    enum3 = []
    for idx, inner in enumerate(bio_train[0], start=1):
        enum2.append(idx)
        for jdx, elt in enumerate(inner, start=1):
            enum1.append(jdx)
            enum3.append(elt)
                  
        
    # Number of tokens per sentence
    nest_token_enum = []
    for x in bio_train[0]:
        nest_token_enum.append(len(x))
        
    # Sentence ID
    numbers_ar = np.arange(1, len(non_overlap_data_train)+1)
    nest_token_enum_ar = np.asarray(nest_token_enum)
    sent_id = np.repeat(numbers_ar, nest_token_enum_ar)
    
            
    # BIO tags
    list_bio2 = []
    for x in bio_train[1]:
        list_bio2.append(x)
    flatten = lambda l: [item for sublist in l for item in sublist]
    flatten_BIO = flatten(list_bio2)
    
    
    # POS tags
    pos = tokens_to_pos_tags(enum3)
    
    
    # Sentence list with string
    sentence = []
    for i in sent_id:
        sentence.append(f'Sentence: {i}')
        
    
    # Create a dataframes (train(60/100), train_eval(80/100), eval(train_eval- train)=20/100)
    d = {'Sentence_ID':sentence,'Token_ID':enum1, 'BIO':flatten_BIO, 'POS':pos, 'Token': enum3}
    df = pd.DataFrame(d)
    df['ID'] = df.index
    
    l = df[df["Token_ID"]==1].index.tolist()
    for c, i in enumerate(l):
        dfs = np.split(df, [i+1+c])
            df = pd.concat([dfs[0], pd.DataFrame([[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]], columns=df.columns), dfs[1]], ignore_index=True)
    
    df_new = df.fillna(method='ffill')
    df_new['duplicate'] = df_new.duplicated(keep='last')
    df_new.loc[(df_new['duplicate']==True)] = [["", "", "", "", "", "", "" ]]
    df_new.drop(['Sentence_ID', "Token_ID", "POS", "ID", 'duplicate'], axis=1)
    cols = ["Token", "BIO"]
    df_new = df_new[cols]
    df_new['BIO'] = df_new['BIO'].replace({f'B_{label}' :'B', f'I_{label}' : 'I'})





    ### the size of the following datasets needs to be adjusted manually according to the nearest sentence end.
    # TRAIN
    df_train = df_new[:53970]
    df_train.to_csv(f'{output_dir_train}/train.tsv', sep = '\t', header = False, index=False)
    
    # TRAIN_EVAL
    df_new.to_csv(f'{output_dir_train_eval}/train_dev.tsv', sep = '\t', header = False, index=False)
    
    # EVAL
    dr_valid = df_new[53970:]
    dr_valid.to_csv(f'{output_dir_eval}/devel.tsv', sep = '\t', header = False, index=False)


    #####################
    ### Test set ###
    
    enum1 = []
    enum2 = []
    enum3 = []
    for idx, inner in enumerate(bio_test[0], start=1):
        #print('sentence', idx)
        enum2.append(idx)
        for jdx, elt in enumerate(inner, start=1):
            #print(jdx, elt)
            enum1.append(jdx)
            enum3.append(elt)
    
    # Number of tokens per sentence
    nest_token_enum = []
    for x in bio_test[0]:
        nest_token_enum.append(len(x))
        
    # Sentence ID
    import numpy as np
    from itertools import repeat
    
    numbers_ar = np.arange(1, len(non_overlap_data_test)+1)
    nest_token_enum_ar = np.asarray(nest_token_enum)
    sent_id = np.repeat(numbers_ar, nest_token_enum_ar)
    
    # BIO tags
    list_bio2 = []
    for x in bio_test[1]:
        list_bio2.append(x)
    flatten = lambda l: [item for sublist in l for item in sublist]
    flatten_BIO = flatten(list_bio2)
    
    
    # Create POS tags
    pos = tokens_to_pos_tags(enum3)
    
    
    # Sentence list with string
    sentence = []
    for i in sent_id:
        sentence.append(f'Sentence: {i}')
    
    
    # Create a dataframe (test)    
    d = {'Sentence_ID':sentence,'Token_ID':enum1, 'BIO':flatten_BIO, 'POS':pos, 'Token': enum3}
    df = pd.DataFrame(d)
    df['ID'] = df.index
    
    l = df[df["Token_ID"]==1].index.tolist()
    for c, i in enumerate(l):
        dfs = np.split(df, [i+1+c])
        df = pd.concat([dfs[0], pd.DataFrame([[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]], columns=df.columns), dfs[1]], ignore_index=True)
    
    df_new = df.fillna(method='ffill')
    df_new['duplicate'] = df_new.duplicated(keep='last')
    df_new.loc[(df_new['duplicate']==True)] = [["", "", "", "", "", "", "" ]]
    df_new.drop(['Sentence_ID', "Token_ID", "POS", "ID", 'duplicate'], axis=1)
    cols = ["Token", "BIO"]
    df_new = df_new[cols]
    df_new['BIO'] = df_new['BIO'].replace({f'B_{label}' :'B', f'I_{label}' : 'I'})
    
    #save the dataset in a csv file
    df_new.to_csv(f'{output_dir_test}/test.tsv', sep = '\t', header = False, index=False)




    ##########################
    #### Labels mapping ######
    """
    The following maps the original labels to WordPiece tokens that are used in BioBERT for the test set that will be used
    for the evaluation. To use this you need to clone the  github "google-research/bert" repository. The link is:
    https://github.com/google-research/bert
    """
    
    path = path_google_bert
    os.chdir(path)
    import tokenization
    from tokenization import FullTokenizer
    
    
    #### For all sentences
    # Token map will be an int -> int mapping between the `orig_tokens` index and
    # the `bert_tokens` index.
    
    
    origin_tokens = bio_test[0]
    origin_labels = bio_test[1]
    bert_tokens_all = []
    orig_to_tok_map_all = []
    
    # The "vocab.txt" file is from BERT-Base, Cased: 12-layer, 768-hidden, 12-heads , 110M parameters
    # see "https://github.com/google-research/bert"
    tokenizer = tokenization.FullTokenizer(
        vocab_file= 'vocab.txt', do_lower_case=False)
    
    
    for sent in origin_tokens:
        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append("[CLS]")
        for orig_token in sent:
            orig_to_tok_map.append(len(bert_tokens))
            bert_tokens.extend(tokenizer.tokenize(orig_token))
        bert_tokens.append("[SEP]")
        bert_tokens_all.append(bert_tokens)
        orig_to_tok_map_all.append(orig_to_tok_map)
    
    
    bert_labs_all = []
    for i in range(len(bert_tokens_all)):
        bert_labs = [None] * len(bert_tokens_all[i])
        for p, l in zip(orig_to_tok_map_all[i], origin_labels[i]):
            bert_labs.insert(p, l)
            bert_labs_choped = bert_labs[:len(bert_tokens_all[i])]
        bert_labs_all.append(bert_labs_choped)
    flat_bert_tokens = [item for sublist in bert_tokens_all for item in sublist]
    flat_bert_labels = [item for sublist in bert_labs_all for item in sublist]
    
    #Create a dataframe with flat_bert_tokens and flat_bert_labels
    df = pd.DataFrame({'bert_tokens':flat_bert_tokens, 'bert_labels': flat_bert_labels})
    df.fillna(value='X', inplace=True)
    df['bert_labels'][df.bert_tokens == '[CLS]'] = '[CLS]'
    df['bert_labels'][df.bert_tokens == '[SEP]'] = '[SEP]'
    df.bert_labels.replace([f'B_{label}', f'I_{label}'], ['B', 'I'], inplace=True)
    
    # Save in txt files
    df['bert_tokens'].to_csv(f'{output_dir_mapping_test_tokens}/convtokens_forbert.txt', header=None, index=None, sep=' ', mode='a')
    df['bert_labels'].to_csv(f'{output_dir_mapping_test_labels}/convlabels_forbert.txt', header=None, index=None, sep=' ', mode='a')
    
    
    
if __name__ == '__main__':
    main()