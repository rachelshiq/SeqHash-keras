# -*- coding: utf-8 -*-
"""
@author: Qian Shi
"""
import os
import numpy as np
os.environ['KERAS_BACKEND']='tensorflow'

def main():
    seq_len = 100
    seq_num = 300
    var_seq_num = 100
    near_ed_ratio = 0.8
    far_ed_ratio = 0.2
    magnify = 64
    
    if not os.path.exists("data"):
        os.mkdir("data")
        
    seq_data = magnify*np.random.randint(4,size=[seq_num,seq_len])
    print(seq_data)    
    #np.savetxt('data/seq_data.csv',seq_data,fmt='%i', delimiter=",")
    
    #ed_matrix = metrics.pairwise_distances(seq_data,metric=editDist).astype(int)
    #print(train_ed_matrix)
    #np.savetxt('data/ed_matrix.csv',ed_matrix,fmt='%i', delimiter=",")
    
    near_ed = np.floor(seq_len*0.2)
    far_ed = np.ceil(seq_len*0.4)
    print("near_ed upper bound: ",near_ed*3,"far_ed upper bound:",far_ed*3)
        
    near_num = round(var_seq_num*near_ed_ratio)
    far_num = round(var_seq_num*far_ed_ratio)
    print("near_num per seq: ",near_num,"far_num per seq: ",far_num)
    
    data_pairs = []
    dist_pairs = []    
    for i in range(seq_num):
        if (i%10)==0:
            print("Generating pairs for sequence: ",i," to ",min(i+9,seq_num-1))
        for j in range(near_num):
            insert_num = np.random.randint(near_ed)
            #remove_num = insert_num
            sub_num = np.random.randint(near_ed)
            #print("insert_num: ",insert_num,"remove_num:",remove_num,"sub_num:",sub_num)
        
            seq_var = np.copy(seq_data[i])
            #seq_var = seq_var.reshape((1,seq_len))
            #print(np.shape(seq_var))
            #print("Base seq: ",seq_var)
            for k in range(insert_num):
                idx = np.random.randint(seq_len)
                var = np.random.randint(4)*magnify
                #print("Insert ",var,"at idx:",idx)
                seq_var = np.insert(seq_var,idx,var)
                #print("New seq: ",seq_var)
                
                idx = np.random.randint(seq_len)
                #print("Remove at idx:",idx)
                seq_var = np.delete(seq_var,idx)
                #print("New seq: ",seq_var)
                
            for k in range(sub_num):
                idx = np.random.randint(seq_len)
                var = np.random.randint(4)*magnify
                #print("Substitude ",var,"at idx:",idx)
                seq_var[idx] = var
                #print("New seq: ",seq_var)
            
            #print("Final new seq: ",seq_var)
            edit_dist = editDist(seq_data[i],seq_var)
            #print("Edit distance: ",edit_dist,"\n")
            #if edit_dist == 0:
                #continue;
            data_pairs += [[seq_data[i],seq_var]]
            dist_pairs += [edit_dist]
        
        for j in range(far_num):
            insert_num = np.random.randint(near_ed,far_ed)
            #remove_num = insert_num
            sub_num = np.random.randint(near_ed,far_ed)
            #print("insert_num: ",insert_num,"remove_num:",remove_num,"sub_num:",sub_num)
        
            seq_var = np.copy(seq_data[i])
            #seq_var = seq_var.reshape((1,seq_len))
            #print(np.shape(seq_var))
            #print("Base seq: ",seq_var)
            for k in range(insert_num):
                idx = np.random.randint(seq_len)
                var = np.random.randint(4)*magnify
                #print("Insert ",var,"at idx:",idx)
                seq_var = np.insert(seq_var,idx,var)
                #print("New seq: ",seq_var)
                
                idx = np.random.randint(seq_len)
                #print("Remove at idx:",idx)
                seq_var = np.delete(seq_var,idx)
                #print("New seq: ",seq_var)
                
            for k in range(sub_num):
                idx = np.random.randint(seq_len)
                var = np.random.randint(4)*magnify
                #print("Substitude ",var,"at idx:",idx)
                seq_var[idx] = var
                #print("New seq: ",seq_var)
            
            #print("Final new seq: ",seq_var)
            edit_dist = editDist(seq_data[i],seq_var)
            #print("Edit distance: ",edit_dist,"\n")
            #if edit_dist == 0:
                #continue;
            data_pairs += [[seq_data[i],seq_var]]
            dist_pairs += [edit_dist]
    
    np.save('data/data_pairs',data_pairs)
    np.save('data/dist_pairs',dist_pairs)
    #np.savetxt('data/dist_pairs.csv',dist_pairs,fmt='%i')
        
    #data_pairs = []
    #dist_pairs = []
    #for i in range(seq_num):
        #for j in range(i,seq_num):
            #data_pairs += [[seq_data[i],seq_data[j]]]
            #dist_pairs += [ed_matrix[i][j]]
    #print(np.shape(data_pairs),np.shape(dist_pairs))
    #np.save('data/data_pairs',data_pairs)
    #np.save('data/dist_pairs',dist_pairs)
            
    
def editDist(seq1,seq2): 
    # Create a table to store results of subproblems
    #print("seq1:",seq1,"seq2:",seq2)
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
