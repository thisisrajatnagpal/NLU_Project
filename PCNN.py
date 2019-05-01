
# coding: utf-8

# In[192]:





import gensim
import numpy as np
import collections as coll
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
import tensorflow as tf
from math import floor
from math import ceil
from sklearn.metrics import classification_report
import sklearn
import os
#length of sentence is 70
fixlen = 70
#max length of position embedding is 60 (-60~+60)
maxlen = 60

#embedding the position 
def pos_embed(x):
    if x < -maxlen:
        return 0
    if x >= -maxlen and x <= maxlen:
        return x+ maxlen + 1
    if x > maxlen:
        return 2*(maxlen+1)
    


# In[193]:





##########################################Data-Preprocessing####################################################
file = 'TRAIN_FILE.TXT'
data = open(file,mode='r').read()
test_file = 'TEST_FILE_FULL.TXT'
test_data = open(test_file,mode='r').read()

###### Reading Word-Embeddings from a pre trained embedding data################
vec = []
word2id = {}
f = open('vec.txt', encoding = 'latin1')
f.readline()
while True:
    content = f.readline()
    if content == '':
        break
    content = content.strip().split()
    word2id[content[0]] = len(word2id)
    content = content[1:]
    content = [(float)(i) for i in content]
    vec.append(content)
f.close()
word2id['UNK'] = len(word2id)
word2id['BLANK'] = len(word2id)

dim = 50 

vec.append(np.random.normal(size=dim,loc=0,scale=0.05))##########appending UNK##################
vec.append(np.random.normal(size=dim,loc=0,scale=0.05))###########appending BLANK##################


vec = np.array(vec,dtype=np.float32)
print(vec.shape)


# In[194]:



split_data = data.splitlines()
sent_data = []
sent_label = []
for i in range(int(len(split_data)/4)):
    sent = split_data[4*i]
    sent = word_tokenize(sent)
    sent_data.append(sent[2:len(sent)-2])
    sent_label.append(split_data[4*i + 1]) 
    
entity_index = np.zeros((len(sent_data),2))
for i in range(len(sent_data)):
    entity_index[i,0] = sent_data[i].index('e1') - 1
    entity_index[i,1] = sent_data[i].index('e2') - 7
    
rm_words = ['<', 'e1', '>','/e1','e2','/e2']
sent_len = np.zeros((len(sent_data)))
for i in range(len(sent_data)):
    sent_data[i] = [words.lower() for words in sent_data[i] if words not in rm_words]
    sent_len[i] = min(fixlen,len(sent_data[i]))
print(entity_index[0,:],sent_data[0])


# In[195]:


#### Defining the Index matrix for sentences
sent_index = np.zeros((len(sent_data),fixlen))
for i in range(len(sent_data)):
    for j in range(fixlen):
        if j>=sent_len[i]:
            sent_index[i,j] = word2id['BLANK']
        else:
            if sent_data[i][j] not in word2id:
                sent_index[i,j] = word2id['UNK']
            else:
                sent_index[i,j] = word2id[sent_data[i][j]]

#### Defining enity position matrix for each word
pos_matrix = np.zeros((sent_index.shape[0],sent_index.shape[1],2))
for i in range(sent_index.shape[0]):
    for j in range(sent_index.shape[1]):
        pos_matrix[i,j,0] = pos_embed(j - entity_index[i,0])
        pos_matrix[i,j,1] = pos_embed(j - entity_index[i,1])


# In[196]:


#### Preparing the test data
split_data_test = test_data.splitlines()
sent_data_test = []
sent_label_test = []
for i in range(int(len(split_data_test)/4)):
    sent = split_data_test[4*i]
    sent = word_tokenize(sent)
    sent_data_test.append(sent[2:len(sent)-2])
    sent_label_test.append(split_data_test[4*i + 1]) 
    
entity_index_test = np.zeros((len(sent_data_test),2))
for i in range(len(sent_data_test)):
    entity_index_test[i,0] = sent_data_test[i].index('e1') - 1
    entity_index_test[i,1] = sent_data_test[i].index('e2') - 7
    
rm_words = ['<', 'e1', '>','/e1','e2','/e2']
sent_len_test = np.zeros((len(sent_data_test)))
for i in range(len(sent_data_test)):
    sent_data_test[i] = [words.lower() for words in sent_data_test[i] if words not in rm_words]
    sent_len_test[i] = min(fixlen,len(sent_data_test[i]))


# In[197]:


#### Defining the Index matrix for sentences
sent_index_test = np.zeros((len(sent_data_test),fixlen))
for i in range(len(sent_data_test)):
    for j in range(fixlen):
        if j>=sent_len_test[i]:
            sent_index_test[i,j] = word2id['BLANK']
        else:
            if sent_data_test[i][j] not in word2id:
                sent_index_test[i,j] = word2id['UNK']
            else:
                sent_index_test[i,j] = word2id[sent_data_test[i][j]]

#### Defining enity position matrix for each word
pos_matrix_test = np.zeros((sent_index_test.shape[0],sent_index_test.shape[1],2))
for i in range(sent_index_test.shape[0]):
    for j in range(sent_index_test.shape[1]):
        pos_matrix_test[i,j,0] = pos_embed(j - entity_index_test[i,0])
        pos_matrix_test[i,j,1] = pos_embed(j - entity_index_test[i,1])


# In[198]:


########## Making one-hot vectors for Labels
unique_labels = sorted(list(set(sent_label)))
label_index = dict((c, i) for i, c in enumerate(unique_labels))
one_hot_label = np.zeros((len(sent_label),len(unique_labels)))
for i in range(len(sent_label)):
    one_hot_label[i,label_index[sent_label[i]]] = 1
    
one_hot_label_test = np.zeros((len(sent_label_test),len(unique_labels)))
for i in range(len(sent_label_test)):
    one_hot_label_test[i,label_index[sent_label_test[i]]] = 1


# In[ ]:





# In[199]:


###Function to get next batch during Training###############################
def get_next_batch(x_data, y_data, data_len, ent_index, pos_matrix, batch_id, batch_size):
    start = batch_id*batch_size
    end = min(start + batch_size, x_data.shape[0])
    X = x_data[start:end,:]
    Y = y_data[start:end,:]
    length = data_len[start:end]
    index = ent_index[start:end,:]
    pos = pos_matrix[start:end,:,:]
    return X,Y,length, index, pos


# In[ ]:





# In[200]:


################################TensorFlowGRaph###############################

tf.reset_default_graph()
epochs=20
batch_size=100
no_filters=130
filter_size=3*60
no_classes=19
reg_constant=.05
initializer = tf.contrib.layers.xavier_initializer()
regularizer = tf.contrib.layers.l2_regularizer(scale = reg_constant)



########placeholders#############################
x = tf.placeholder(dtype = tf.int32, shape = [None,fixlen])
y = tf.placeholder(dtype = tf.int32, shape = [None,len(unique_labels)])
x_len = tf.placeholder(dtype = tf.int32, shape = [None,]) #end
ent_index = tf.placeholder(dtype = tf.int32, shape=[None,2]) 
x_position = tf.placeholder(dtype = tf.int32, shape = [None,fixlen,2])
prob = tf.placeholder_with_default(1.0, shape=())


#########Variables########################
w1 = tf.get_variable(name = "w1", shape = [3*no_filters,no_classes], regularizer = regularizer, 
                                initializer=initializer,dtype=tf.float32) 

b1= tf.get_variable(name = "b1",shape = [1,no_classes], initializer=initializer,regularizer = regularizer,dtype=tf.float32)



# In[201]:



embeddings_word = tf.get_variable(name = "embeddings_word", initializer = vec, 
                                  regularizer = regularizer, dtype = tf.float32)
input_embedded_word = tf.nn.embedding_lookup(embeddings_word, x)
embeddings_pos = tf.get_variable(name = "embeddings_pos", shape = [2*(maxlen+1)+1,5,2], initializer = initializer, 
                                  regularizer = regularizer, dtype = tf.float32)

input_embedded_pos_1 = tf.nn.embedding_lookup(embeddings_pos[:,:,0], x_position[:,:,0])

input_embedded_pos_2 = tf.nn.embedding_lookup(embeddings_pos[:,:,1], x_position[:,:,1])

###concatenated embedding
final_input=tf.concat([input_embedded_word ,input_embedded_pos_1, input_embedded_pos_2],axis=2)


# In[202]:



############layers##################
layer_embedding = tf.reshape(final_input,[ tf.shape(x)[0] , tf.shape(x)[1] *60,1 ])

window= tf.Variable(tf.random_normal([filter_size,1,no_filters])) # filter_size X 1 X no_filters(3D)


layer_zeros=tf.zeros([ tf.shape(x)[0],60,1])

layer_embedding=tf.concat([layer_zeros,layer_embedding],axis=1)

layer_embedding=tf.concat([layer_embedding,layer_zeros],axis=1)

conv_layer  =  tf.nn.conv1d(layer_embedding, window , stride=1*60 , padding='VALID' )# no of examples X no of words X no_filters(3D)


output_list=[]
flag=[]
index_list=[]

### piecewise maxpooling for every sentence in training batch#################
for i in range(batch_size):
    last_ind=x_len[i]
    e1_ind=ent_index[i,0]
    e2_ind=ent_index[i,1]+1
    dummy=conv_layer[i,:,:]
    dummy=tf.reshape(dummy,[tf.shape(conv_layer)[1],tf.shape(conv_layer)[2]])
    to_max_pool1=dummy[:e1_ind+1 ,:]
    to_max_pool2=dummy[e1_ind+1 :e2_ind,:]
    to_max_pool3=dummy[e2_ind:last_ind,:]
    index_list.append(e1_ind)
    to_max_output1=tf.reduce_max(to_max_pool1,axis=0)
    to_max_output2=tf.reduce_max(to_max_pool2,axis=0)
    to_max_output3=tf.reduce_max(to_max_pool3,axis=0)
   
        
    dummy_output=tf.concat([to_max_output1,to_max_output2,to_max_output3],axis=0)
    
    output_list.append(dummy_output)
max_pool=tf.stack(output_list)


# In[204]:



with tf.Session() as sess:
    
    
    
    
    
    
    max_pool=tf.squeeze(max_pool)
    
    max_pool=tf.nn.tanh(max_pool)
    
    max_pool_drop=tf.nn.dropout(max_pool,keep_prob=prob)####dropout regularization################
    
    layer_dnn1 = tf.add(tf.matmul(max_pool_drop,w1),b1)
    
    logits=layer_dnn1
    
    logits_final = tf.nn.softmax(logits)  # Apply softmax to pred
    correct_prediction = tf.equal(tf.argmax(logits_final, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    
    emp_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_op = emp_loss + reg_constant*sum(reg_losses)
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_op)
     
    
    
    
    
    init =tf.global_variables_initializer()
    sess.run(init)
    
    
 
    total_batch=int(sent_index.shape[0]/batch_size)
    f1_test=[]
    
    
    
    #training of model
    for epoch in range(epochs):
        cost=0

        permute = np.random.permutation(sent_index.shape[0])
        sent_index=sent_index[permute]
        one_hot_label=one_hot_label[permute]
        entity_index=entity_index[permute]
        pos_matrix=pos_matrix[permute]
        sent_len=sent_len[permute]
        
        for i in range(total_batch):

            
            batch_x,batch_y,batch_end,batch_entity,batch_p,=get_next_batch(sent_index,one_hot_label, sent_len, 
                                                                          entity_index, pos_matrix,i, batch_size)
        
           
            c=sess.run([optimizer,loss_op],feed_dict={x:batch_x,y:batch_y,x_len:batch_end,
                                                      ent_index:batch_entity,x_position:batch_p,prob:0.5})
            
            cost+= c[1]/total_batch
            
            
        
        print(" Epoch %d "%epoch +" Training cost is %lf "%cost)
        if(epoch%5==0): #######Test F1-score print at interval of 5 epochs####################
            acc=0
            test_batch=int(sent_index_test.shape[0]/batch_size)
            y_pred=np.array([])
            y_scores=[]

            for i in range(test_batch):

                
                batch_x,batch_y,batch_end,batch_entity,batch_p=get_next_batch(sent_index_test,one_hot_label_test, sent_len_test, 
                                                                          entity_index_test, pos_matrix_test,i, batch_size)
                
                acc_batch=sess.run(logits_final,feed_dict={x:batch_x,y:batch_y,x_len:batch_end,ent_index:batch_entity,
                                                           x_position:batch_p,prob:1})
                y_pred=np.append(y_pred,np.argmax(acc_batch,axis=1))
                y_scores.append(acc_batch)
            temp  = sklearn.metrics.f1_score(np.argmax(one_hot_label_test,1)[0:2700],y_pred,average='weighted')
            f1_test.append(temp)
            print("F1 score on test data : %lf" %temp)

