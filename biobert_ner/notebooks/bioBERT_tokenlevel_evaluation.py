from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import os

"""
The following script computes thew token-level BioBERT performance. 
The input is a list of "entity.txt" files of the five different splits.
    path1,2,3,4,5 : The path to output of 'bioBERT_entitylevel_evaluation.py' file ("entity.txt")
The output is a csv file with token-level model performance ("tokenlevel_BERT.csv")
"""

def main():
    parser = argparse.ArgumentParser(description='bioBERT_tokenlevel_evaluation.py')
    parser.path1('path1', dest = 'path1', deafult = None, type = str)
    parser.path2('path2', dest = 'path2', deafult = None, type = str)
    parser.path3('path3', dest = 'path3', deafult = None, type = str)
    parser.path4('path4', dest = 'path4', deafult = None, type = str)
    parser.path5('path5', dest = 'path5', deafult = None, type = str)
    parser.output_dir('output_dir', dest = 'output_dir', deafult = None, type = str)
    args = parser.parse_args()


    
    # Function that creates a dataframe report
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
        class_report_df.columns = ["B", "I", "O"]
    
        support = class_report_df.loc['support']
        total = support.sum() 
        avg[-1] = total
    
        class_report_df['avg / total'] = avg
    
        return class_report_df.T
    
    # Give a list of paths (BioBERT results from all splits)
    paths = [path1, path2, path3, path4, path5]
    
    
    
    # Create dattaframe with all results
    token_level_df = pd.DataFrame()
    
    for path in paths:
        os.chdir(path)
        df = pd.read_csv('entity.txt', sep=" ", header = None)
        df.columns = ['tokens', 'y_true', 'y_pred']
        classes = ['B-ADR', 'I-ADR', 'O-ADR']
        df_class_report = pandas_classification_report(y_true=df['y_true'], y_pred=df['y_pred'])
        token_level_df = token_level_df.append(df_class_report)
        
    # Save results in a csv file   
    token_level_df.to_csv(f'{output_dir}/tokenlevel_BERT.csv',  sep=',')
    
    
    
    
if __name__ == '__main__':
    main()