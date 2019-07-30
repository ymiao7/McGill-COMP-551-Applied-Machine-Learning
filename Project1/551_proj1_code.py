import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import matplotlib.pyplot as plt # visualization
import numpy as np
from numpy import linalg as LA # calculate norm
import pandas as pd
import time # time different methods
from collections import Counter # select most frequent elements in a list

# =============================================================================
# MAIN function is at the end of this file !!!
# =============================================================================

def text_modification(data):
    for x in data:
        x['text'] = x['text'].lower()
        x['text'] = x['text'].split()
    return data


# Top frequent words 
# "data" is the input list of dictionaries, and "num" is the number of top frequent words needed
def top_frequent_words(data, num):
    words = []
    for x in data:
        words = words + x['text']
        
    # Find 160 most frequent words
    Count = Counter(words) 
    most_occur = Count.most_common(num)
    
    # Get rid of the tuple in the list most_occur1 and most_occur2
    words1 = []
    for x in most_occur:
        words1.append(x[0])

#    print(words1)
    return words1


# Create the "word count features"
def text_features(data, wordList):
    word_matrix = [] # a matrix shows the frequency of the top words occurring in comments
    for x in wordList:    
        temp = []
        for y in data:
            num = y['text'].count(x)
            temp.append(num)
        word_matrix.append(temp)

    return word_matrix  



  
# Feature and label extraction 
def information_extraction(data):
    text = []
    is_root = []
    controversiality = []
    children = []
    log_children = []
    sqrt_children = []
    popularity_score = []
    
    for x in data:
        text.append(x['text'])
        if x['is_root']==False:
            is_root.append(0)
        else:
            is_root.append(1)
        controversiality.append(x['controversiality'])
        children.append(x['children'])
        log_children.append(np.log10(x['children']+1))
        sqrt_children.append(np.sqrt(x['children']))
        
        popularity_score.append(x['popularity_score'])
        
    return text, is_root, controversiality, children, log_children, sqrt_children, popularity_score


# Create dummy variable list
def onelist(n):
    listofones = [1] * n
    return listofones


# Create a new feature: number of words in each comment
def comment_words(X_text):
    X_numwords = []
    
    for x in X_text:
        X_numwords.append(len(x))
    
        
    return X_numwords


# Sort the comments according to propularity scores 
# select out the top frequent 150 words in top 50 comments 
# remove the first 100 words
# Then count the number of presence of these words in each comment
def topfreq_highscore(data, wordList):
    sorted_list = sorted(data, key=lambda k: k['popularity_score'], reverse=True) # records sorted by popularity scores
    
    # From the 50 comments with top scores, select out 150 words with top frequency
    words_150 = top_frequent_words(sorted_list[0:49], 150)
    # Remove the first 100 words since they probably have low correlations with popularity scores
    words_50 = words_150[101:]
    
    # Count the number of presence of these of words in each comment        
    count = []
    for x in data:
        temp = 0
        for words in words_50:
            if words in x['text']:
                temp += 1
        count.append(temp)
        
    return count
    

## Using closed-form
def error_function(W_est, X, y):
    # Calculate MSE
    totalError = np.dot((y-np.dot(X, W_est)).T, y-np.dot(X, W_est))
    return totalError[0][0]/len(y)


def closed_form_runner(X, y):
    XTX = X.T.dot(X)
    
    if np.linalg.det(XTX) == 0.0:
        # If the matrix is singular, this shows that some of the features 
        # are not linearly independent
        print("Singular matrix error")
        return 
    
    XTX_inv = pd.DataFrame(np.linalg.inv(XTX.values), XTX.columns, XTX.index)
    
    # get estimation of weight matrix
    W_est = XTX_inv.dot(X.T.dot(y))
    
    err = error_function(W_est, X, y)
    return err, W_est


## Using gradient descent
def gradient_descent_setup(X, y, eta_0, beta, epsilon):
    # Initialize W with zeros
    W_0 = np.zeros(shape=(X.shape[1],1))
    W_est, err_record = gradient_descent_runner(X, y, W_0, beta, eta_0, epsilon)
    err = error_function(W_est, X, y)
    return W_est, err_record, err
    
    
def gradient_descent_runner(X, y, W_0, beta, eta_0, epsilon):
    W = [W_0, 1+W_0]

    err_record = []
    
    # Calculate XTX outside the loop the reduce running time
    XTX = np.dot(X.T,X)
    if np.linalg.det(XTX) == 0.0:
        # If the matrix is singular, this shows that some of the features 
        # are not linearly independent
        print("Singular matrix error")
        return 
    
    XTy = np.dot(X.T,y)
    
    # scale the learning rate by 1/n, where n=# training points
    scale = 1/y.shape[0]
    
    i = 1
    while LA.norm(W[1]-W[0]) > epsilon:
        if i != 1:
            W[0] = W[1]
        alpha = scale * eta_0 / (1 + beta*i)
        W[1] = W[0] - 2 * alpha * (np.dot(XTX, W[0]) - XTy)
        i+=1
        
        #print(LA.norm(W[1]-W[0]))
        # Store the error 
        if i % 3 == 0:
            err_record.append(error_function(W[1], X, y))
        
        if i > 100000:
            break   
    
    return W[1], err_record


def data_preprocessing(data, word_list):
    # Create training sets
    X_text, X_root, X_contro, X_chil, X_logchil, X_sqrtchil, X_pop = information_extraction(data)
    
    # The reason I partition features into severl lists is to make it convenient when
    # we choose to discard one or more features
    ones = onelist(len(X_pop))
        
    num_words = comment_words(X_text)
    num_words_recip = np.reciprocal(num_words)
    num_words_log = np.log([x+1 for x in num_words])
    
    words_freq = topfreq_highscore(data, word_list)    
    

    # Merge feature lists into dataframe
    X_train = pd.DataFrame(
        {'is_root': X_root,
         'controversiality': X_contro,
         'children': X_chil,
         'log_children': X_logchil,
         'sqrt_children': X_sqrtchil,
         
         # New features (uncomment to see the effect)
         'words_freq': words_freq,
         'num_words_log': num_words_log,
         'num_words_recip': num_words_recip,
         
         'dummy var': ones                  
        })
    
 

# =============================================================================
#     # Add text features (uncomment to see the influence of the text features)
    word_matrix = text_features(data, word_list)   
    i = 0
    for x in word_matrix:
        col_name = word_list[i]
        X_train[col_name] = x
        i = i+1
# =============================================================================

    y = pd.DataFrame({'popularity_score': X_pop})
    
    return X_train, y

# =============================================================================
# Trainig Part
# =============================================================================

def training_runner(data, init_eta, init_beta, init_epsilon, word_list):

    X_train, y_train = data_preprocessing(data, word_list)
    # Closed-form
    start = time.time()    
    err_ls, W_ls = closed_form_runner(X_train, y_train)
    end = time.time()
    time_ls = end-start


    err_gd = []
    W_gd = 0
    
    # Gradient descent
    # (ONLY uncomment gradient descent when text features are not included!!!!!!!!!)
#    start = time.time()
#    W_gd, err_gd, err = gradient_descent_setup(X_train, y_train, init_eta, init_beta, init_epsilon)
#    end = time.time()
#    time_gd = end -start
    
# =============================================================================
# Task 3
    # Closed-form plot
    print("Time using closed form on training set:"+str.format('{0:.20f}', time_ls))
    print("Error using closed form on training set:", err_ls)   
    
    y_train = np.array(y_train.values.tolist())
    t = np.arange(0., len(y_train),1)

    fig = plt.figure()
    plt.plot(t, y_train, 'k.', t, np.dot(X_train, W_ls), 'r.')
    fig.suptitle("Closed-form on training set (in red) vs True values", fontsize=14)
    plt.xlabel('data points', fontsize=14)
    plt.ylabel('popularity score', fontsize=14)
    plt.show()
    
    # Gradient descent plot 
    # (ONLY uncomment gradient descent when text features are not included!!!!!!!!!)
#    print("Time using gradient descent on training set:"+str.format('{0:.20f}', time_gd))
#    print("Error using gradient descent on training set:", err)
#    
#    fig = plt.figure()
#    plt.plot(t, y_train, 'k.', t, np.dot(X_train, W_gd), 'r.')
#    fig.suptitle("Gradient descent fitting values (red) vs Label values", fontsize=14)
#    plt.xlabel('data points', fontsize=14)
#    plt.ylabel('popularity score', fontsize=14)
#    plt.show()
# =============================================================================
    
    return W_ls, W_gd, err_gd

# =============================================================================
# Validation Part
# =============================================================================

def validation_runner(data, W_ls, word_list):
    X_val, y_val = data_preprocessing(data, word_list)

    val_err_ls = error_function(W_ls, X_val, y_val)
    print("Error using closed form on validation set:", val_err_ls)
    
    # Plot
    y_val = np.array(y_val.values.tolist())
    t = np.arange(0., len(y_val),1)
    fig = plt.figure()
    plt.plot(t, y_val, 'k.', t, np.dot(X_val, W_ls), 'r.')
    fig.suptitle("Closed-form on validation set (in red) vs True values", fontsize=14)
    plt.xlabel('data points', fontsize=14)
    plt.ylabel('popularity score', fontsize=14)
    plt.show() 

# =============================================================================
# Test Part
# =============================================================================

def testing_runner(data, W_ls, word_list):
    X_test, y_test = data_preprocessing(data, word_list)
    
    err_ls = error_function(W_ls, X_test, y_test)
    print("Error using closed-form on test set:", err_ls)
    
    # Plot
    y_test = np.array(y_test.values.tolist())    
    t = np.arange(0., len(y_test),1)
    fig = plt.figure()
    plt.plot(t, y_test, 'k.', t, np.dot(X_test, W_ls), 'r.')
    fig.suptitle("Closed-form on test set (in red) vs True values", fontsize=14)
    plt.xlabel('data points', fontsize=14)
    plt.ylabel('popularity score', fontsize=14)
    plt.show()


# =============================================================================
# Main function of the file
# =============================================================================
def main():
    # load data
    with open("proj1_data.json") as fp:
        data = json.load(fp)

    # deal with comment text
    data = text_modification(data)
    
    # Training, Validation, Test sets split
    training, validation, test = np.split(data, [int((5/6)*len(data)), int((11/12)*len(data))])

    # get top frequent words
    wordList = top_frequent_words(training, 60) 
    
#    # Gradient descent hyparameters
    init_eta = 0.00003 # initial learning rate 
    init_beta = 0.0000001 # controls the speed of the decay
    init_epsilon = 0.000001
    
    # training part
    W_ls, W_gd, err_record = training_runner(training, init_eta, init_beta, init_epsilon, wordList)
    
    # Gradient descent error plot 
    # (ONLY uncomment when text features are not included)
#    i = 0
#    fig = plt.figure()
#    plt.plot(err_record)
#    name = "eta=" + str(init_eta) + " beta=" + str(init_beta) + " epsilon=" + str(init_epsilon)
#    fig.suptitle(name, fontsize=16)
#    plt.xlabel('iterations', fontsize=14)
#    plt.ylabel('Error', fontsize=14)
#    i = i+1    
    
    # validation part
    validation_runner(validation, W_ls, wordList)
    # test part
    testing_runner(test, W_ls, wordList)

main()


