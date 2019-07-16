from nerds.input.brat import BratInput
from nerds.dataset.clean import retain_annotations, resolve_overlaps, clean_annotated_documents
from nerds.dataset.split import split_annotated_documents
from nerds.util.nlp import text_to_tokens
from nerds.ner import CRF, ExactMatchDictionaryNER, BidirectionalLSTM
from nerds.dataset.cv import CVSplit
from nerds.output.brat import BratOutput
from nerds.evaluate.score import annotation_precision_recall_f1score
from nerds.dataset.convert import transform_annotated_documents_to_bio_format
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd
import numpy as np


"""
This script has as input the document collection. 
It takes only sentences with less than 130 words, applies 5-fold cross validation and trains three different models: ExactMatchDictionaryNER, CRF  and BidirectionalLSTM. The model name should be given as argument.
The output is two csv files, one with token-level evaluation model performance and the second with entity-level evalluation model performance. 
"""



def main():
    parser = argparse.ArgumentParser(description='traditional_models.py')
    parser.path('path', dest = 'path', deafult = None, type = str)
    parser.label('label', dest = 'label', default = None, type = str)
    parser.output_dir('output_dir', dest = 'output_dir', default = None, type = str)
    parser.model('model', dest = 'model', default = None)
    parser.tokenlevel_file_name('tokenlevel_file_name', dest = 'tokenlevel_file_name', default = None , type = str)
    parser.entitylevel_file_name('entitylevel_file_name', dest = 'entitylevel_file_name', default = None , type = str)
    args = parser.parse_args()
    
    # Import data
    data_path = path
    ann_docs = BratInput(data_path).transform()
    data = retain_annotations(ann_docs, label)
    clean_data = clean_annotated_documents(data)
    non_overlap_data = resolve_overlaps(clean_data)

    # Split all documents collection into sentences
    sent_docs = split_annotated_documents(non_overlap_data)


    # Select sentences with less than 130 words
    short_sentences = []
    for i in sent_docs:
        tokens = text_to_tokens(i.plain_text_)
        if len(tokens)<130:
            print(len(tokens))
            short_sentences.append(i)
        
        
    ### Models ###
    
    #Cvsplit
    splitter_2 = CVSplit(strategy="random",n_folds=5)
    splits = splitter_2.make_cv_folds(short_sentences)
    
    
    train_1 = splits[1] + splits[2] + splits[3] + splits[4]
    test_1 = splits[0]
    
    train_2 = splits[0] + splits[2] + splits[3] + splits[4]
    test_2 = splits[1]
    
    train_3 = splits[0] + splits[1] + splits[3] + splits[4]
    test_3 = splits[2]
    
    train_4 = splits[0] + splits[1] + splits[2] + splits[4]
    test_4 = splits[3]
    
    train_5 = splits[0] + splits[1] + splits[2] + splits[3]
    test_5 = splits[4]
    
    ### Save the different splits
    
    BratOutput("output_dir").transform(train_1)
    BratOutput("output_dir").transform(test_1)
    
    BratOutput("output_dir").transform(train_2)
    BratOutput("output_dir").transform(test_2)
    
    BratOutput("output_dir").transform(train_3)
    BratOutput("output_dir").transform(test_3)
    
    BratOutput("output_dir").transform(train_4)
    BratOutput("output_dir").transform(test_4)
    
    BratOutput("output_dir").transform(train_5)
    BratOutput("output_dir").transform(test_5)



    np.random.seed(0)
    
    y_true = np.array([0]*400 + [1]*600)
    y_pred = np.random.randint(2, size=1000)
    
    def pandas_classification_report(y_true, y_pred):
        metrics_summary = precision_recall_fscore_support(
                y_true=y_true, 
                y_pred=y_pred)
    
        avg = list(precision_recall_fscore_support(
                y_true=y_true, 
                y_pred=y_pred,
                average='weighted'))
    
        metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
        class_report_df = pd.DataFrame(
            list(metrics_summary),
            index=metrics_sum_index)
    
        support = class_report_df.loc['support']
        total = support.sum() 
        avg[-1] = total
    
        class_report_df['avg / total'] = avg
    
        return class_report_df.T



    ### RUN the models ###
    idx = 0
    entity_level_results = []
    token_level_df = pd.DataFrame()
    
    
    for split in splits:
        test = split
        train_splits = splits[:idx] + splits[idx+1:]
        train = [item for sublist in train_splits for item in sublist]
        idx+=1
        
        #Train
        if model == 'ExactMatchDictionaryNER':
   
            model = model(entity_labels= label)
            model.fit(train)
            pred_docs = model.transform(test)
        
        if model == 'BidirectionalLSTM':
   
            model = model(entity_labels= label)
            model.fit(train)
            pred_docs = model.transform(test)
        
        
        if model == 'CRF':
   
            model = model(entity_labels= label)
            model.fit(train, max_iterations = 100)
            pred_docs = model.transform(test)
        
        
        
        
        
        #Evaluate and store (entity-level evaluation)
        metrics_1fold = []
        p, r, f = annotation_precision_recall_f1score(pred_docs, test, ann_label= label)
        print(p, r, f)
        metrics_1fold.append(p)
        metrics_1fold.append(r)
        metrics_1fold.append(f)
        entity_level_results.append(metrics_1fold)
        
        # Convert to X_test, y_test, X_pred, y_pred
        X_test, y_test = transform_annotated_documents_to_bio_format(test,  entity_labels= label)
        X_pred, y_pred = transform_annotated_documents_to_bio_format(pred_docs,  entity_labels= label)
        
        #Keep only the first y_pred of each sentence
        label_pred = []
        for i in range(len(y_pred)):
            unique = y_pred[i][:len(y_test[i])]
            label_pred.append(unique)
        
        # Flat the nested lists
        flat_y_test = [item for sublist in y_test for item in sublist]
        flat_y_pred = [item for sublist in label_pred for item in sublist]
        
        
        # Print separate for B and I (token-level evaluation)
        classes = [f'B_{label}', f'I_{label}']
        print(classification_report(flat_y_test, flat_y_pred, target_names= classes, digits=4))
        df_class_report = pandas_classification_report(y_true=flat_y_test, y_pred=flat_y_pred)
        token_level_df = token_level_df.append(df_class_report)
        
    # Save token-level evaluation report in a csv file
    token_level_df.to_csv(f'{tokenlevel_file_name}.csv',  sep=',')
    df = pd.DataFrame(entity_level_results, columns=["Precision", "Recall", "F1 measure"])
    # Save entity-level evaluation report in a csv file
    df.to_csv(f'{entitylevel_file_name}.csv')


if __name__ == '__main__':
    main()