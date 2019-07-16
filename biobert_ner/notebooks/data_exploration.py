from nerds.input.brat import BratInput
from nerds.dataset.split import split_annotated_documents
import collections
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import rcParams
import pandas as pd
import argparse
import numpy as np



"""
This script is for EDA.
Three output png files: label frequency, top10 label instances and text length distribution (barplots)

    label : for a specific type of label (i.e. "AdverseReaction"). This argument is used to plot the top10 label instances for a given label.
"""


def main():
    parser = argparse.ArgumentParser(description='data_exploration.py')
    parser.path('path', dest = 'path', deafult = None)
    parser.label_title('label_title', dest = 'label_title', default = None, type = str, help = 'graph title')
    parser.file_name('file_name', dest = 'file_name', type = str)
    parser.top10_title ('top10_title', dest = 'top10_title', default = None, type = str, help = 'barplot title for the top10 instances of a label' )
    parser.top10_filename('top10_filename', dest = 'top10_filename', default = None, type = str)
    parser.label('label', dest = 'label', default = None, type = str)
    args = parser.parse_args()
    
    
    
    """
    This is function to create label frequency barplot
    path: the path of the documents
    labels: is a list of all labels in this dataset
    title: a string that is the title of the barplot
    """
    # Import data
    data_path = path
    ann_docs = BratInput(data_path).transform()
    # Split the documents into sentences
    sent_docs = split_annotated_documents(ann_docs)

    # Create a list with all labels from all documents
    list_all_labels = []
    for doc in ann_docs:
        for i in doc.annotations:
            list_all_labels.append(i.label)


    # WORKS: count the frequency of an item in the list
    counter=collections.Counter(list_all_labels)
    freq_list = counter.most_common()
    list_labels = [i[0] for i in freq_list]



    # Plot the label frequency
    sns.set_context("talk")

    fig, ax = plt.subplots()
    labels = list_labels
    #labels = list(labels)
    plt.barh(range(len(freq_list)), [val[1] for val in freq_list], color="darkmagenta", orientation='horizontal', zorder=3)
    plt.yticks(range(len(freq_list)), [val[0] for val in freq_list])
    plt.yticks()
    ax.set_yticklabels(labels)
    plt.grid(axis='x')
    mpl.rcParams['grid.color'] = 'k'
    mpl.rcParams['grid.linestyle'] = ':'
    rcParams.update({'figure.autolayout': True})
    plt.xlabel('Number of occurrences')
    plt.title(f'{label_title}')
    plt.tight_layout()
    plt.savefig(f'{file_name}.png', format='png', dpi=300)

    


    
    # Calculate some statistics
    # Create a list with all text from all documents
    list_all_text = []
    for doc in ann_docs:
        for i in doc.annotations:
            list_all_text.append(i.text)
        
    # Create a list with all offset0 from all documents
    list_all_offset0 = []
    for doc in ann_docs:
        for i in doc.annotations:
            list_all_offset0.append(i.offset[0])
        
    # Create a list with all offset1 from all documents
    list_all_offset1 = []
    for doc in ann_docs:
        for i in doc.annotations:
            list_all_offset1.append(i.offset[1])

        
    ## Zip all lists
    list_data = list(zip(list_all_labels,list_all_text,list_all_offset0, list_all_offset1))
    # List to Dataframe
    df = pd.DataFrame(list_data, columns=['label', 'text', 'offset0', 'offset1'])


    # Calculate the text length and assign it to the dataframe as a new column
    df['text_length'] = df['text'].apply(len)
    # Lowercase the "text" column
    df['text'] = df['text'].str.lower()


    # Calculate the mean and median of text_length
    mean = df['text_length'].mean()
    median = df['text_length'].median()
    var = df['text_length'].var()
    std = df['text_length'].std()

    print(f"\033[1;32;35mThe mean  of the text_length is:  {mean}" '\n'
    f"The median of the text_length is:  {median}"'\n'
    f"The variance of the text_length is: {var}" '\n'
    f"The standard deviation of the text_length is:  {std}")

    # Find the min and max text length
    mini = df['text_length'].min()
    maxi = df['text_length'].max()
    print(f"\033[1;32;35mThe minimum text_length is: {mini} \n The maximum text_length is: {maxi}")



    ####################
    # Create a Top 10 (label) barplot

    # Create a subset of "AdverseReaction" label
    df_AR = df.loc[df['label'] == label]
    df_freq_text =  df_AR[['label','text']].groupby(['text'])['label'].size().nlargest(10).reset_index(name='top10')
    objects = df_freq_text['text'].values
    y_pos = np.arange(len(objects))
    performance = df_freq_text['top10'].values


    fig, ax = plt.subplots(figsize = (10,6))
    plt.barh(y_pos, performance, align='center',alpha=0.8, color = 'orange', orientation= 'horizontal')
    plt.yticks(y_pos, objects, size = 10, ha='right')
    plt.xlabel('Frequency', size = 18)
    plt.title(f'{top10_title}', size = 18)
    plt.tight_layout()
    plt.savefig(f'{top10_filename}.eps', format='eps', dpi=300)
    
    
    
    
    
    #####################
    # Create text length distribution
    # the range could be changed accordingly
    word_count_dict = dict([(range(i, i + 6), 0) for i in range(1, 67, 6)])
    for ann_doc in sent_docs:
        words = [word.strip() for word in ann_doc.plain_text_.split(' ')]
        word_count = len(words)
    
        for key in word_count_dict:
            if word_count in key:
                word_count_dict[key] += 1
    

    # make a barplot for text length distribution
    x = [str(key).split('e')[1] for key in word_count_dict.keys()]
    height = [float(v) for v in word_count_dict.values()]
    plt.bar(x=x, height=height, color='darkmagenta', zorder=3)
    plt.grid(axis='y')
    mpl.rcParams['grid.color'] = 'k'
    mpl.rcParams['grid.linestyle'] = ':'
    rcParams.update({'figure.autolayout': True})
    plt.xlabel('Word Count', fontsize=28)
    plt.ylabel('Sentence Count', fontsize=28)
    plt.title("Text Length Distribution", fontsize=32)
    plt.savefig('text_length_distribution.png', format='png')

if __name__ == '__main__':
    main()
