# -*- coding: utf-8 -*-
"""
@author: Qian Shi
"""
import os
import numpy as np
from sklearn import metrics
os.environ['KERAS_BACKEND']='tensorflow'

def main():
    seq_len = 300
    seq_num = 500
    
    if not os.path.exists("data"):
        os.mkdir("data")
        
    seq_data = np.random.randint(4,size=[seq_num,seq_len])
    #print(seq_train)    
    np.savetxt('data/seq_data.csv',seq_data,fmt='%i', delimiter=",")
    
    ed_matrix = metrics.pairwise_distances(seq_data,metric=editDist).astype(int)
    #print(train_ed_matrix)
    np.savetxt('data/ed_matrix.csv',ed_matrix,fmt='%i', delimiter=",")
    
    data_pairs = []
    dist_pairs = []
    for i in range(seq_num):
        for j in range(i,seq_num):
            data_pairs += [[seq_data[i],seq_data[j]]]
            dist_pairs += [ed_matrix[i][j]]
    print(np.shape(data_pairs),np.shape(dist_pairs))
    np.save('data/data_pairs',data_pairs)
    np.save('data/dist_pairs',dist_pairs)
            
    
def editDist(seq1,seq2): 
    # Create a table to store results of subproblems 
    m = len(seq1)
    n = len(seq2)
    dp = [[0 for x in range(n+1)] for x in range(m+1)] 
  
    # Fill d[][] in bottom up manner 
    for i in range(m+1): 
        for j in range(n+1): 
  
            # If first string is empty, only option is to 
            # isnert all characters of second string 
            if i == 0: 
                dp[i][j] = j    # Min. operations = j 
  
            # If second string is empty, only option is to 
            # remove all characters of second string 
            elif j == 0: 
                dp[i][j] = i    # Min. operations = i 
  
            # If last characters are same, ignore last char 
            # and recur for remaining string 
            elif seq1[i-1] == seq2[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
  
            # If last character are different, consider all 
            # possibilities and find minimum 
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1])    # Replace 
  
    return dp[m][n] 
    
if __name__ == '__main__':
    main()
