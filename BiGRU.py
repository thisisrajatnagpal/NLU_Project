
# coding: utf-8

# In[149]:


import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import collections as coll

import tensorflow as tf

from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, f1_score

#length of sentence is 70
fixlen = 70
#max length of position embedding is 60 (-60~+60)
maxlen = 30

#embedding the position 
def pos_embed(x):
    if x < -maxlen:
        return 0
    if x >= -maxlen and x <= maxlen:
        return x+ maxlen + 1
    if x > maxlen:
        return 2*(maxlen+1)


# In[150]:


file = 'TRAIN_FILE.TXT'
data = open(file,mode='r').read()
test_file = 'TEST_FILE_FULL.TXT'
test_data = open(test_file,mode='r').read()

###### Reading Word-Embeddings
vec = []
word2id = {}
f = open('./origin_data/vec.txt', encoding = 'latin1')
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
vec.append(np.random.normal(size=dim,loc=0,scale=0.05))
vec.append(np.random.normal(size=dim,loc=0,scale=0.05))
vec = np.array(vec,dtype=np.float32)


# In[151]:


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


# In[127]:


print(entity_index[0,:],sent_data[0])


# In[152]:


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


# In[153]:


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


# In[154]:


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


# In[155]:


########## Making one-hot vectors for Labels
unique_labels = sorted(list(set(sent_label)))
label_index = dict((c, i) for i, c in enumerate(unique_labels))
one_hot_label = np.zeros((len(sent_label),len(unique_labels)))
for i in range(len(sent_label)):
    one_hot_label[i,label_index[sent_label[i]]] = 1
    
one_hot_label_test = np.zeros((len(sent_label_test),len(unique_labels)))
for i in range(len(sent_label_test)):
    one_hot_label_test[i,label_index[sent_label_test[i]]] = 1


# In[156]:


def get_next_batch(x_data, y_data, data_len, ent_index, pos_matrix, batch_id, batch_size):
    start = batch_id*batch_size
    end = min(start + batch_size, x_data.shape[0])
    X = x_data[start:end,:]
    Y = y_data[start:end,:]
    length = data_len[start:end]
    index = ent_index[start:end,:]
    pos = pos_matrix[start:end,:,:]
    return X,Y,length, index, pos


# In[301]:


num_neurons = 230
num_train_epochs = 20
batch_size = 100
reg_constant = 0.05
# no_filters = 50
# filter_size = 3*60
regularizer = tf.contrib.layers.l2_regularizer(scale = reg_constant)
initializer = tf.contrib.layers.xavier_initializer()


# In[304]:


########## Making tensorflow Graph
tf.reset_default_graph()

# Defining Placeholders
x = tf.placeholder(dtype = tf.int32, shape = [None,fixlen])
y = tf.placeholder(dtype = tf.int32, shape = [None,len(unique_labels)])
x_len = tf.placeholder(dtype = tf.int32, shape = [None,])
ent_index = tf.placeholder(dtype = tf.int32, shape=[None,2])
x_position = tf.placeholder(dtype = tf.int32, shape = [None,fixlen,2])
prob = tf.placeholder_with_default(1.0, shape=())

######### Variables
Dense_weights = tf.get_variable(name = "Dense_weights", shape = [2*3*num_neurons,len(unique_labels)], regularizer = regularizer, 
                                initializer=initializer) 
Dense_bias = tf.get_variable(name = "Dense_bias",shape = [len(unique_labels)], initializer=initializer)

att_mat = tf.get_variable(name = "att_mat", shape = [2*num_neurons,2*num_neurons], regularizer = regularizer, 
                            initializer = initializer)

word_att = tf.get_variable(name = "word_att", shape = [2*num_neurons,1], regularizer = regularizer, 
                            initializer = initializer)

# window = tf.get_variable(name = "window", shape = [filter_size, 1, no_filters], 
#                             initializer = initializer)

######### Embedding Layer
embeddings_word = tf.get_variable(name = "embeddings_word", initializer = vec, 
                                  regularizer = regularizer, dtype = tf.float32)
input_embedded_word = tf.nn.embedding_lookup(embeddings_word, x)

embeddings_pos1 = tf.get_variable(name = "embeddings_pos1", shape = [2*(maxlen+1)+1,5], initializer = initializer, 
                                  regularizer = regularizer, dtype = tf.float32)
embeddings_pos2 = tf.get_variable(name = "embeddings_pos2", shape = [2*(maxlen+1)+1,5], initializer = initializer, 
                                  regularizer = regularizer, dtype = tf.float32)
input_embedded_pos1 = tf.nn.embedding_lookup(embeddings_pos1, x_position[:,:,0])
input_embedded_pos2 = tf.nn.embedding_lookup(embeddings_pos2, x_position[:,:,1])
#input_embedded_pos = tf.reshape(input_embedded_pos,[tf.shape(input_embedded_pos)[0],tf.shape(input_embedded_pos)[1],-1])

#input_embedded_pos.set_shape([None, None, 10])

final_input = tf.concat([input_embedded_word,input_embedded_pos1,input_embedded_pos2],axis = 2)


# #CNN
# layer_embedding = tf.reshape(final_input,[tf.shape(x)[0],tf.shape(x)[1]*60,1])

# layer_zeros = tf.zeros([tf.shape(x)[0],60,1])

# layer_embedding = tf.concat([layer_zeros,layer_embedding],axis=1)
# layer_embedding = tf.concat([layer_embedding,layer_zeros],axis=1)

# final_input_conv = tf.nn.conv1d(layer_embedding,window,stride=1*60,padding='VALID')


######## BiGRU Cells
cell = tf.contrib.rnn.GRUCell(num_units = num_neurons, activation = tf.nn.relu)
cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=prob)
((fw_out, bw_out), (fw_state, bw_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw = cell, cell_bw = cell, sequence_length = x_len,
                                        inputs = final_input, dtype=tf.float32))
states = tf.concat((fw_state, bw_state), 1)
outputs = tf.concat([fw_out, bw_out], axis = 2)

######## Word Attention layer and piecewise concatenation
output_shape = outputs.get_shape()
output_list = []

alpha_score = []
for i in range(batch_size):
    dummy = outputs[i,0:x_len[i],:]
    dummy = tf.reshape(dummy,[x_len[i],output_shape[2]])
    eta = tf.matmul(dummy, att_mat)
    eta = tf.nn.tanh(eta)
    eta = tf.nn.dropout(eta, keep_prob = prob)
    eta = tf.matmul(eta, word_att)
    alpha_score.append(tf.nn.softmax(eta,axis = 0))
    alpha = tf.tile(tf.nn.softmax(eta,axis = 0),(1, output_shape[2]))
    out = alpha * dummy
    ##### Doing Piecewise Sum and concatenating
    p1 = tf.cond(ent_index[i,0] > 0, lambda: tf.reduce_sum(out[0:ent_index[i,0],:], axis = [0]), lambda: tf.zeros((output_shape[2]), tf.float32) )
    p2 = tf.reduce_sum(out[ent_index[i,0]:ent_index[i,1]+1,:], axis = [0])
    p3 = tf.cond(ent_index[i,1] < x_len[i]-1, lambda: tf.reduce_sum(out[ent_index[i,1]+1:,:], axis = [0]), lambda:tf.zeros((output_shape[2]), tf.float32))
    concat_out = tf.concat([p1,p2,p3], axis = 0)
    output_list.append(concat_out)
    
att_outputs = tf.nn.tanh(tf.stack(output_list))

final_output = tf.add(tf.matmul(att_outputs, Dense_weights), Dense_bias)

## Helper Objects
emp_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = final_output, labels = y))
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
total_loss = emp_loss + reg_constant*sum(reg_losses)
correct = tf.equal(tf.argmax(final_output,1) ,tf.argmax(y,1))
tag_output = tf.argmax(final_output,1)
tag_output_true = tf.argmax(y,1)
accuracy = tf.reduce_sum(tf.cast(correct,'float'))
optimizer = tf.train.AdamOptimizer().minimize(total_loss)
init = tf.global_variables_initializer()


# In[305]:


with tf.Session() as sess:
    sess.run(init) 
    f1_test = []
    sent_index1 = sent_index
    one_hot_label1 = one_hot_label
    sent_len1 = sent_len
    entity_index1 = entity_index 
    pos_matrix1 = pos_matrix
    for epochs in range(num_train_epochs):        
        #### Shuffling the data
        permute = np.random.permutation(sent_index1.shape[0])
        sent_index1 = sent_index1[permute]
        one_hot_label1 = one_hot_label1[permute]
        sent_len1 = sent_len1[permute]
        entity_index1 = entity_index1[permute] 
        pos_matrix1 = pos_matrix1[permute]
        
        epoch_loss = 0
        epoch_loss_total = 0
        for i in range(int(sent_index.shape[0]/batch_size)):
            x_batch, y_batch, x_len_batch, ent_index_batch, x_position_batch = get_next_batch(sent_index1, one_hot_label1, sent_len1, entity_index1, pos_matrix1, i, batch_size)
            _,Loss,T_loss = sess.run([optimizer,emp_loss,total_loss], feed_dict = {x : x_batch, y: y_batch, x_len: x_len_batch, ent_index: ent_index_batch, x_position: x_position_batch ,prob: 0.5})
            epoch_loss += Loss
            epoch_loss_total += T_loss
        print('Epoch', epochs, 'completed out of', num_train_epochs,'loss: ', epoch_loss, epoch_loss_total)
        
        correct_test = 0
        y_pred = np.array([])
        y_true = np.array([])
        for i in range(int(sent_index_test.shape[0]/batch_size)):
            x_batch, y_batch, x_len_batch, ent_index_batch, x_position_batch = get_next_batch(sent_index_test, one_hot_label_test, sent_len_test, entity_index_test, pos_matrix_test, i, batch_size)
            correct, output, output_true = sess.run([accuracy, tag_output, tag_output_true], 
                                        feed_dict = {x : x_batch, y: y_batch, x_len: x_len_batch,
                                        ent_index: ent_index_batch, x_position: x_position_batch})
            correct_test += correct
            y_pred = np.append(y_pred, output)
            y_true = np.append(y_true, output_true)
        f1_test.append(f1_score(y_true, y_pred, average = 'weighted'))
        print('Test_accuracy: ',correct_test/sent_index_test.shape[0], 'Test F1_Score',f1_test[epochs])
        
        if epochs==num_train_epochs - 1:
            print(classification_report(y_true, y_pred,target_names=list(label_index.keys())))    
        
    #### Finding accuracy
    correct_train = 0
    att_scores = []
    for i in range(int(sent_index1.shape[0]/batch_size)):
        x_batch, y_batch, x_len_batch, ent_index_batch, x_position_batch = get_next_batch(sent_index, one_hot_label, sent_len, entity_index, pos_matrix, i, batch_size)
        correct, att_weights = sess.run([accuracy, alpha_score], feed_dict = {x : x_batch, y: y_batch, x_len: x_len_batch, ent_index: ent_index_batch, x_position: x_position_batch})
        correct_train += correct
        att_scores += att_weights
    print('Train_accuracy: ',correct_train/sent_index.shape[0])


# In[222]:


import matplotlib.pyplot as plt
y = att_scores[149]
x = range(len(y))
plt.xticks(x, sent_data[149],rotation=90)
plt.plot(x,y,'C4', zorder=1, lw=2)
plt.scatter(x,y, s=120, zorder=2)
plt.grid(color='b', axis ='y',linestyle='-', linewidth=0.2)
plt.show()

