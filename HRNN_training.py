# Crash Catcher: DashCam Accident Detector  </br>
### Determining whether dashboard camera video contains an accident</br></br>
# I use a hierchical recurrent neural network implementation, trained on a set of videos with and without accidents, to determine whether a new video contains an accident or not.</br></br></br> 


# Load in necessary packages

## we want any plots to show up in the notebook
get_ipython().magic(u'matplotlib inline')
## has the usual packages I use
get_ipython().magic(u'run startup')
import numpy
import os
import re
import pickle
import timeit
import glob
import cv2

from skimage import transform
import skimage
from skimage import io

import sklearn
from sklearn.model_selection import train_test_split   ### import sklearn tool

import keras
from keras.preprocessing import image as image_utils
from keras.callbacks import ModelCheckpoint

rcdefaults()  ### set the defaults
matplotlib.rc('font',family='Bitstream Vera Serif')   ### I like my plots to look a certain way :)


# First, write a function to load in a video (.mp4 format, 720 by 1280 in size) from file.
# Each frame of the video will be converted to an image that can be processed.
# 
# In order to make the process (marginally) less memory-intensive, we downscale the image to
# a size of 144 pixels by 256 pixels. In addition, because the images are originally in RGB color,
# we convert to gray-scale. This also reduces the amount of memory, and while some useful information
# may be lost, the color variations from scene to scene (or dashcam to dashcam) are less important.
# Further, losing the color dimension turns a 5-D problem into a 4-D problem -- a bit more tractable.


### here is the function to load in a video from file for analysis

def load_set(videofile):
    '''The input is the path to the video file - the training videos are 99 frames long and have resolution of 720x1248
       This will be used for each video, individially, to turn the video into a sequence/stack of frames as arrays
       The shape returned (img) will be 99 (frames per video), 144 (pixels per column), 256 (pixels per row))
    '''
    ### below, the video is loaded in using VideoCapture function
    vidcap = cv2.VideoCapture(videofile)
    ### now, read in the first frame
    success,image = vidcap.read()
    count = 0       ### start a counter at zero
    error = ''      ### error flag
    success = True  ### start "sucess" flag at True

    img = []        ### create an array to save each image as an array as its loaded 
    while success: ### while success == True
        success, img = vidcap.read()  ### if success is still true, attempt to read in next frame from vidcap video import
        count += 1  ### increase count
        frames = []  ### frames will be the individual images and frames_resh will be the "processed" ones
        for j in range(0,99):
            try:
                success, img = vidcap.read()
                ### conversion from RGB to grayscale image to reduce data
                tmp = skimage.color.rgb2gray(numpy.array(img))
                ### ref for above: https://www.safaribooksonline.com/library/view/programming-computer-vision/9781449341916/ch06.html
                
                ### downsample image
                tmp = skimage.transform.downscale_local_mean(tmp, (5,5))
                frames.append(tmp)
                count+=99
            
            except:
                count+=1
                pass#print 'There are ', count, ' frame; delete last'        read_frames(videofile, name)
    
        ### if the frames are the right shape (have 99 entries), then save
        #print numpy.shape(frames), numpy.shape(all_frames)
        if numpy.shape(frames)==(99, 144, 256):
            all_frames.append(frames)
        ### if not, pad the end with zeros
        elif numpy.shape(frames[0])==(144,256):
            #print shape(all_frames), shape(frames), shape(concatenate((all_frames[-1][-(99-len(frames)):], frames)))
            #print numpy.shape(all_frames), numpy.shape(frames)
            all_frames.append(numpy.concatenate((all_frames[-1][-(99-len(frames)):], frames)))
        elif numpy.shape(frames[0])!=(144,256):
            error = 'Video is not the correct resolution.'
    vidcap.release()
    del frames; del image
    return all_frames, error



# Next, we load in the training data and randomly select and split the training and validation sets
# (some data is also set aside/not included for testing later)


img_filepath = '/pathway/to/videos' #### the filepath for the training video set
neg_all = glob.glob(img_filepath + 'negative/*.mp4')               #### negative examples - ACCV
pos_2 = glob.glob(img_filepath + 'positive/*.mp4')                 #### positive examples - ACCV
pos_1 = glob.glob(img_filepath + '../YTpickles/*.pkl')             #### positive examples - youtube
pos_all = concatenate((pos_1, pos_2))

all_files = concatenate((pos_all, neg_all))
print len(neg_all), len(pos_all)                                   #### print check



def label_matrix(values):
    '''transforms labels for videos to one-hot encoding/dummy variables'''
    n_values = numpy.max(values) + 1    ### take max value (that would be 1, because it is a binary classification), 
                                        ### and create n+1 (that would be two) sized matrix
    return numpy.eye(n_values)[values]  ### return matrix with results coded - 1 in first column for no-accident
                                        ### and a 1 in second column for an accident

labels = numpy.concatenate(([1]*len(pos_all), [0]*len(neg_all[0:len(pos_all)])))  ### create the labels for the videos
labels = label_matrix(labels)           ### make the labels into a matrix for the HRNN training



# Load in data from each video and save to (massive) data array -- should be of
# shape (L, 99, 144, 256), where L is the number of files that are going to be used.
# We use a function to load in the data differently depending on whether it is pickled
# (from youtube) or part of the ACCV dataset.


def make_dataset(rand):
    seq1 = numpy.zeros((len(rand), 99, 144, 256))   ### create an empty array to take in the data
    for i,fi in enumerate(rand):                    ### for each file...
        print (i, fi)                               ### as we go through, print out each one
        if fi[-4:] == '.mp4':
            t = load_set(fi)                        ### load in the video file using previously defined function if .mp4 file
        elif fi[-4:]=='.pkl':
            t = pickle.load(open(fi, 'rb'))         ### otherwise, if it's pickled data, load the pickle
        if shape(t)==(99,144,256):                  ### double check to make sure the shape is correct, and accept
            seq1[i] = t                             ### save image stack to array
        else:# TypeError:
            'Image has shape ', shape(t), 'but needs to be shape', shape(seq1[0]) ### if exception is raised, explain
            pass                                    ### continue loading data
    print (shape(seq1))
    return seq1



# The data is then split (with the labels created above) into training and validation sets,
# with 60% of the total set as training and 20% as validation (the remaining 20% of data is
# left as a holdout test set).
# 
# The split fractions may look a little odd, but they are essentially ensuring that the validation
# and test sets are the same size (an overall 60-20-20 for training-validation-test).


##### split data into training and validation (sets and shuffle)
x_train, x_t1, y_train, y_t1 = train_test_split(all_files, labels, test_size=0.40, random_state=0)  ### split
x_train = array(x_train); y_train = array(y_train)                          ### need to be arrays

x_testA = array(x_t1[len(x_t1)/2:]); y_testA = array(y_t1[len(y_t1)/2:])    #### test set

### valid set for model
x_testB = array(x_t1[:len(x_t1)/2]); y_test = array(y_t1[:len(y_t1)/2])    ### need to be arrays
x_test = make_dataset(x_testB)




# Below, a test was run to check whether there is signal above the noise -- fake data was
# generated from random numbers to show that the real data performed better than data/patterns
# picked up from random data.
# Thankfully, the model could barely reach 50% accuracy when run with random numbers.


#### populate data as random numbers as a sanity check
#seq3 = zeros((60,99,144,256))
#for j in range(60):   ### for each file...
#    [np.random.random((244,256)) for i in range(99)]    ### save image stack to array
#print (shape(seq3))              ### print check

#x_train2, x_test2, y_train2, y_test2 = train_test_split(seq3, labels, test_size=0.2, random_state=0)  ### split
#x_train2 = array(x_train2); y_train2 = array(y_train2)     ### need to be arrays
#x_test2 = array(x_test2); y_test2 = array(y_test2)         ### need to be arrays




# Below, the HRNN is set up to run on the train and validation set.

### the code is largely re-worked from the following resource
### https://github.com/fchollet/keras/blob/master/examples/mnist_hierarchical_rnn.py

"""HRNNs can learn across multiple levels of temporal hiearchy over a complex sequence.
Usually, the first recurrent layer of an HRNN encodes a time-dependent video (e.g. set of images)
into a vector. The second recurrent layer then encodes those vectors (encoded by the first layer) into a second layer.
# References
    - [A Hierarchical Neural Autoencoder for Paragraphs and Documents](https://arxiv.org/abs/1506.01057)
    - [Hierarchical recurrent neural network for skeleton based action recognition](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298714)
The first LSTM layer first encodes every column of pixels of shape (240, 1) to a column vector of shape (128,).
The second LSTM layer encodes then these 240 column vectors of shape (240, 128) to a image vector representing the whole image. 
A final Dense layer is added for prediction.
"""
import keras
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM

### set hyper-parameters
batch_size = 15
num_classes = 2
epochs = 30

### number of hidden layers in each NN
row_hidden = 128
col_hidden = 128

### print basic info
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

### get shape of rows/columns for each image
frame, row, col = (99, 144, 256)

### 4D input - for each 3-D sequence (of 2-D image) in each video (4th)
x = Input(shape=(frame, row, col))

encoded_rows = TimeDistributed(LSTM(row_hidden))(x)  ### encodes row of pixels using TimeDistributed Wrapper
encoded_columns = LSTM(col_hidden)(encoded_rows)     ### encodes columns of encoded rows using previous layer

### set up prediction and compile the model
prediction = Dense(num_classes, activation='softmax')(encoded_columns)
model = Model(x, prediction)
model.compile(loss='categorical_crossentropy', ### loss choice for category classification - computes probability error
              optimizer='NAdam',               ### NAdam optimization
              metrics=['accuracy'])            ### grade on accuracy during each epoch/pass

### create a filepath to save best results as we go - http://machinelearningmastery.com/check-point-deep-learning-models-keras/
### because who wants to train this crazy stuff more than once??!
i=0; filepath='HRNN_pretrained_model.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


### now we actually train - because of my laptop memory issues, this means the training data cannot
### be loaded into memory all at once because python will crash. To get around this issue, we load in 
### the whole dataset and loop through in batches of 15. 
### However, each time we pass through the entire dataset, the order of the data needs to be randomized.
### So, we shuffle the list of files during each epoch, then split into batches of 15 videos
numpy.random.seed(18247)  ### set a random seed for repeatability

for i in range(0, 30):               ### number of epochs
    c = list(zip(x_train, y_train))  ### bind the features and labels together
    random.shuffle(c)                ### shuffle the list
    x_shuff, y_shuff = zip(*c)       ### unzip list into shuffled features and labels
    x_shuff = array(x_shuff); y_shuff=array(y_shuff) ### back into arrays
    
    x_batch = [x_shuff[i:i + batch_size] for i in range(0, len(x_shuff), batch_size)] ### make features into batches of 15
    y_batch = [y_shuff[i:i + batch_size] for i in range(0, len(x_shuff), batch_size)] ### make labels into batches of 15

    for j,xb in enumerate(x_batch):  ### for each batch in the shuffled list for this epoch
        xx = make_dataset(xb)        ### load the feature data into arrays
        yy = y_batch[j]              ### set the labels for the batch
        
        model.fit(xx, yy,                            ### fit training data
                  batch_size=len(xx),                ### reiterate batch size - in this case we already set up the batches
                  epochs=1,                          ### number of times to run through each batch
                  validation_data=(x_test, y_test),  ### validation set from up earlier in notebook
                  callbacks=callbacks_list)          ### save if better than previous!

# evaluate
scores = model.evaluate(x_test, y_test, verbose=0)    ### score model
print('Test loss:', scores[0])                        ### test loss
print('Test accuracy:', scores[1])                    ### test accuracy (ROC later)




#### Checking the results - ROC curves


### first, load and compile the saved model to make predictions
model.load_weights("HRNN_pretrained_model.hdf5")
model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])

### make the holdout test dataset for prediction and comparison
x_holdout = make_dataset(x_testA)



plot([0,1],[0,1],'k:',alpha=0.5)                       ### plot the "by chance" line - the goal is to achieve better than random accuracy
ys = [y_train, y_testB, y_testA]                       ### set up labels to be iterated through
labs = ['Train', 'Valid', 'Test']                      ### set up tags to be iterated through
col = ['#4881ea', 'darkgreen', 'maroon']               ### set up colors to be iterated through
preds = []                                             ### set up prediction as empty array to populate
for i,xset in enumerate([x_train, x_testB, x_testA]):  ### iterate through each set of data
    if i==0:
        new_pred = []                                  ### for first dataset, need to iterate through each
        for k in xset:                                 ### to save memory (because we can't load the whole
            d = make_dataset([k])                      ### thing at once)
            new_pred.append(model.predict(d))          ### predictions with loaded model for each in training set
        new_pred = array(new_pred).reshape((len(new_pred),2))
    else:
        d = make_dataset(xset)                         ### can load all of valid/test datasets at once in memory
        new_pred = model.predict(d)                    ### predictions with loaded model for each valid/test dataset
    preds.append(new_pred)
    fpr, tpr, threshs = sklearn.metrics.roc_curve(ys[i][:,1], new_pred[:,1]) ### get the false pos rate and true pos rate
    plot(fpr, tpr, '-', color=col[i], alpha=0.7, lw=1.5, label=labs[i])      ### plot the ROC curve with false pos rate and true pos rate
    
    print labs[i]
    print sklearn.metrics.auc(fpr, tpr)                ### print area under curve for each set
    print sklearn.metrics.accuracy_score(ys[i][:,1], [round(j) for j in new_pred[:,1]])   ### print accuracy for each set
    print sklearn.metrics.confusion_matrix(ys[i][:,1], [round(j) for j in new_pred[:,1]]) ### print confusion matrix for each
    
xlabel('False Positive Rate'); ylabel('True Positive Rate')
plt.legend(fancybox=True, loc=4, prop={'size':10})
plt.show()



#### Examine probability range of predictions


plot([0,1],[0,1],'k:',alpha=0.5)                  ### plot the "by chance" line - trying so hard to be better than this...
for i,p in enumerate(preds):                      ### for each of the calculated predictions, make a histogram
    hist(p[:,1], bins = arange(0,1,0.05), histtype='stepfilled', color=col[i], alpha=0.7, label=labs[i]
xlabel('False Positive Rate'); ylabel('True Positive Rate')
plt.legend(fancybox=True, loc=2, prop={'size':10})
plt.show()

