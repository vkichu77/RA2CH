import cv2
import numpy as np
import math
from itertools import chain
import pandas as pd
import sklearn.preprocessing as prep 
import tensorflow as tf
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# tf.estimator.Estimator._validate_features_in_predict_input = lambda *args: None
import tensorflow.keras.layers as layers 

def dooperations2(img, saveflag, displayflag, savefile):
    blur = cv2.GaussianBlur(img,(5,5),0)
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.Canny(blur, 20, 100)
    edges = cv2.dilate(edges, kernel, iterations=1)
    # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    connectivity = 4
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity, cv2.CV_32S)
    sizes = stats[1:, -1]
    num_labels = num_labels - 1
    # your answer image
    img2 = np.zeros(labels.shape)
    # for every component in the image, you keep it only if it's above min_size
    min_size = 600

    for i in range(0, num_labels):
        if sizes[i] >= min_size:
            img2[labels == i + 1] = 255

    contours, hierarchy = cv2.findContours(img2.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    retsize = len(contours)

    return img2, retsize
    # print(sizes)

def dooperations21(img, saveflag, displayflag, savefile):
    # blur = cv2.GaussianBlur(img,(5,5),0)
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.Canny(img, 20, 100)
    edges = cv2.dilate(edges, kernel, iterations=1)
    # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    connectivity = 4
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity, cv2.CV_32S)
    sizes = stats[1:, -1]
    num_labels = num_labels - 1
    # your answer image
    img2 = np.zeros(labels.shape)
    # for every component in the image, you keep it only if it's above min_size
    min_size = 600

    for i in range(0, num_labels):
        if sizes[i] >= min_size:
            img2[labels == i + 1] = 255

    contours, hierarchy = cv2.findContours(img2.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    retsize = len(contours)

    return img2, retsize
    # print(sizes)

def doHOG(img, c1, c2, b1, b2):
    cell_size = (c1, c2)  # h x w in pixels
    block_size = (b1, b2)  # h x w in cells
    nbins = 9  # number of orientation bins

    # winSize is the size of the image cropped to an multiple of the cell size
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    # n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
    hog_feats = hog.compute(img.astype(np.uint8))
    return hog_feats

def divideblocks(img, imgmask):
    global width, height
    # Define the window size
    img = img[6:506, 6:506]
    imgmask = imgmask[6:506, 6:506]
    tmp = cv2.findNonZero(imgmask.astype(np.uint8))
    x, y, w, h = cv2.boundingRect(tmp)
    imgmask = imgmask[y:y+h, x:x+w]
    img = img[y:y + h, x:x + w]
    imgmask = cv2.resize(imgmask, (512, 512))
    img = cv2.resize(img, (512, 512))
    hog_feat = []
    c1, c2, b1, b2, windowsize_r, windowsize_c = 4, 4, 2, 2, 16, 16
    for r in range(0, imgmask.shape[0] - windowsize_r, windowsize_r):
        for c in range(0, imgmask.shape[1] - windowsize_c, windowsize_c):
            if cv2.countNonZero(imgmask[r:r + windowsize_r, c:c + windowsize_c]) > 0:
                hog_feat.append(doHOG(img[r:r + windowsize_r, c:c + windowsize_c], c1, c2, b1, b2))
                # print(hog_feat)
    # hog_feat = list(chain.from_iterable(hog_feat))
    # print(len(hog_feat))
    # c1, c2, b1, b2, windowsize_r, windowsize_c = 5, 5, 3, 3, 25, 25
    # c1, c2, b1, b2, windowsize_r, windowsize_c = 6, 6, 3, 3, 24, 24
    c1, c2, b1, b2, windowsize_r, windowsize_c = 8, 8, 4, 4, 24, 24
    for r in range(0, imgmask.shape[0] - windowsize_r, windowsize_r):
        for c in range(0, imgmask.shape[1] - windowsize_c, windowsize_c):
            if cv2.countNonZero(imgmask[r:r + windowsize_r, c:c + windowsize_c]) > 0:
                hog_feat.append(doHOG(img[r:r + windowsize_r, c:c + windowsize_c], c1, c2, b1, b2))
    # hog_feat = list(chain.from_iterable(hog_feat))
    # print(len(hog_feat))
    c1, c2, b1, b2, windowsize_r, windowsize_c = 6, 6, 4, 4, 36, 36
    for r in range(0, imgmask.shape[0] - windowsize_r, windowsize_r):
        for c in range(0, imgmask.shape[1] - windowsize_c, windowsize_c):
            if cv2.countNonZero(imgmask[r:r + windowsize_r, c:c + windowsize_c]) > 0:
                hog_feat.append(doHOG(img[r:r + windowsize_r, c:c + windowsize_c], c1, c2, b1, b2))

    hog_feat = list(chain.from_iterable(hog_feat))
    hog_feat = list(chain.from_iterable(hog_feat))
    # print(len(hog_feat))
    # print(len(hog_feat))
    # print(math.sqrt(len(hog_feat)))
    rdim = (math.ceil(math.sqrt(len(hog_feat))), math.ceil(math.sqrt(len(hog_feat))))
    s = np.zeros(math.ceil(math.sqrt(len(hog_feat))) ** 2)
    s[:len(hog_feat)] = np.asarray(hog_feat).transpose()
    hog_feat = np.reshape(s, rdim)

    # print(hog_feat.shape)
    # assert(False)
    s = []
    hog_feat = cv2.resize(hog_feat, (width, height))
    #print(hog_feat)

    # print(hog_feat.nbytes)
    # print(hog_feat.shape)
    return hog_feat


def getDataFromCSV(path, filename, typeFeat):    
    df = pd.read_csv(path + filename)
    # print(df)
    colRH = ['RH_mcp_E__ip', 'RH_pip_E__2',	'RH_pip_E__3', 'RH_pip_E__4', 'RH_pip_E__5', 'RH_mcp_E__1',	'RH_mcp_E__2', 'RH_mcp_E__3', 'RH_mcp_E__4', 'RH_mcp_E__5',	'RH_wrist_E__mc1', 'RH_wrist_E__mul', 'RH_wrist_E__nav', 'RH_wrist_E__lunate', 'RH_wrist_E__radius', 'RH_wrist_E__ulna',\
             'RH_pip_J__2', 'RH_pip_J__3', 'RH_pip_J__4', 'RH_pip_J__5', 'RH_mcp_J__1', 'RH_mcp_J__2', 'RH_mcp_J__3', 'RH_mcp_J__4', 'RH_mcp_J__5', 'RH_wrist_J__cmc3', 'RH_wrist_J__cmc4', 'RH_wrist_J__cmc5', 'RH_wrist_J__mna', 'RH_wrist_J__capnlun', 'RH_wrist_J__radcar']
    colLH = ['LH_mcp_E__ip', 'LH_pip_E__2', 'LH_pip_E__3', 'LH_pip_E__4', 'LH_pip_E__5', 'LH_mcp_E__1', 'LH_mcp_E__2', 'LH_mcp_E__3', 'LH_mcp_E__4', 'LH_mcp_E__5', 'LH_wrist_E__mc1', 'LH_wrist_E__mul', 'LH_wrist_E__nav', 'LH_wrist_E__lunate', 'LH_wrist_E__radius', 'LH_wrist_E__ulna',\
             'LH_pip_J__2', 'LH_pip_J__3', 'LH_pip_J__4', 'LH_pip_J__5', 'LH_mcp_J__1', 'LH_mcp_J__2', 'LH_mcp_J__3', 'LH_mcp_J__4', 'LH_mcp_J__5', 'LH_wrist_J__cmc3', 'LH_wrist_J__cmc4', 'LH_wrist_J__cmc5', 'LH_wrist_J__mna', 'LH_wrist_J__capnlun', 'LH_wrist_J__radcar']
    colLF = ['LF_mtp_E__ip', 'LF_mtp_E__1',	'LF_mtp_E__2', 'LF_mtp_E__3', 'LF_mtp_E__4', 'LF_mtp_E__5',\
             'LF_mtp_J__ip', 'LF_mtp_J__1', 'LF_mtp_J__2', 'LF_mtp_J__3', 'LF_mtp_J__4', 'LF_mtp_J__5']
    colRF = ['RF_mtp_E__ip', 'RF_mtp_E__1', 'RF_mtp_E__2', 'RF_mtp_E__3', 'RF_mtp_E__4', 'RF_mtp_E__5',\
             'RF_mtp_J__ip', 'RF_mtp_J__1', 'RF_mtp_J__2', 'RF_mtp_J__3', 'RF_mtp_J__4', 'RF_mtp_J__5']
    colOAnames = ['Overall_Tol', 'Overall_erosion', 'Overall_narrowing']         
    colOA = df[colOAnames]
    # print('{} {} {} {} {}'.format(len(colRH), len(colLH), len(colLF), len(colRF), len(colRH)+len(colLH)+len(colLF)+len(colRF)))

    patIDs = [path + s + '-' + typeFeat + '.jpg' for s in df['Patient_ID']]
    if typeFeat in 'RF':
        colscores = df[colRF]
        colnames = colRF
    elif typeFeat in 'RH':
        colscores = df[colRH]
        colnames = colRH
    elif typeFeat in 'LH':
        colscores = df[colLH]
        colnames = colLH
    else:
        colscores = df[colLF]
        colnames = colLF
    return patIDs, colscores, colOA, colnames, colOAnames


def getHOGForTrainData(patIDs):
    global width, height
    traindata=np.zeros((len(patIDs), width, height))
    idx = 0
    for imagepath in patIDs:
        img, imgmask = [], []
        img = cv2.resize(cv2.imread(str(imagepath), 0), (512, 512))
        imgmask, retsize = dooperations2(img.copy(), False, False, '')
        if retsize < 50:
            imgmask = []
            imgmask, retsize1 = dooperations21(img.copy(), False, False, '')
            # print("{}. {} blur {} no blur {}".format(idx, imagepath, retsize, retsize1))
        # else:
            # print("{}. {} with blur {}".format(idx, imagepath, retsize))
        hog_data = divideblocks(img, imgmask)
        traindata[idx, :, :] = hog_data
        idx = idx + 1
        
    return traindata

def standard_scale(X_train, X_test): 
    preprocessor = prep.StandardScaler().fit(X_train) 
    X_train = preprocessor.transform(X_train) 
    X_test = preprocessor.transform(X_test) 
    return X_train, X_test 
  
def get_random_block_from_data(data, batch_size): 
    start_index = np.random.randint(0, len(data) - batch_size) 
    return data[start_index:(start_index + batch_size)] 

class Encoder(tf.keras.Model):
    def __init__(self, l2_regularizer):
        super(Encoder, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)
        self.maxp1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)             
        self.maxp2 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')        
        self.conv3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)   
        self.maxp3 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.conv4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)   
        self.maxp4 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.flatten1 = tf.keras.layers.Flatten()
        self.encoded = tf.keras.layers.Dense(64, activation='softmax')
        
    def call(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        x = self.maxp3(x)
        x = self.conv4(x)
        x = self.maxp4(x)
        # print(x.shape)
        x = self.flatten1(x)
        # print(x.shape)
        x = self.encoded(x)
        # print(x.shape)
        return x

class Decoder(tf.keras.Model):
    def __init__(self, l2_regularizer):
        super(Decoder, self).__init__()        
        
        self.reshp1 = tf.keras.layers.Reshape((8, 8, 1))
        self.upsample00 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)
        self.upsample0 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv4 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)
        self.upsample1 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)
        self.upsample2 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv6 = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)    
        # self.upsample3 = tf.keras.layers.UpSampling2D((2, 2))    
        self.conv7 = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')

    def call(self, x):
        # print(x.shape)
        x = self.reshp1(x)
        x = self.upsample00(x)    
        x = self.conv3(x)
        # print(x.shape)
        x = self.upsample0(x)    
        x = self.conv4(x)
        # print(x.shape)
        x = self.upsample1(x)    
        # print(x.shape)
        x = self.conv5(x)
        # print(x.shape)
        x = self.upsample2(x)
        # print(x.shape)
        x = self.conv6(x)
        # print(x.shape)
        # x = self.upsample3(x)
        # print(x.shape)
        x = self.conv7(x)
        # print(x.shape)
        return x    


class ConvNetAutoEncoder(tf.keras.Model):
    def __init__(self, l2_regularizer):
        super(ConvNetAutoEncoder, self).__init__()
        
        self.encoder = Encoder(l2_regularizer) 
        self.decoder = Decoder(l2_regularizer) 
            
    def call(self, x, is_training=True):
        x = self.encoder(x)
        if is_training:
            x1 = self.decoder(x) 
        else:
            x1 = []
        return x, x1

def loss(x, x_bar):
    return tf.reduce_sum( tf.keras.losses.MSE(x, x_bar) )

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        encoded, reconstruction = model(inputs, True)
        loss_value = loss(targets, reconstruction)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), reconstruction, encoded

def getBlocks(X_train, st, ed, st1, ed1):
    X_train1=np.zeros((len(X_train), 128, 128))
    for x in range(len(X_train)):
        X_train1[x, :, :] = X_train[x, st:ed, st1:ed1]
    return X_train1

def getBlocks1(outputpath, outputfilename1, r, windowsize_r, c, windowsize_c):
    with open(outputpath + '{}-1.pckl'.format(outputfilename1), 'rb') as f:  # Python 3: open(..., 'rb')
        X_train, X_test = pickle.load(f)
    X_train1=np.zeros((len(X_train), windowsize_r, windowsize_c))
    for x in range(len(X_train)):
        X_train1[x, :, :] = X_train[x, r:r + windowsize_r, c:c + windowsize_c]

    X_test1=np.zeros((len(X_test), windowsize_r, windowsize_c))
    for x in range(len(X_test)):
        X_test1[x, :, :] = X_test[x, r:r + windowsize_r, c:c + windowsize_c]    

    X_train, X_test = [], []
    return X_train1, X_test1

def createAutoencoderData(typeFeat, trnpath, trnfilename, tstpath, tstfilename, outputpath, outputfilename, outputfilename1, \
                          learning_rate, l2_reg, num_epochs, batch_size, windowsize_r, windowsize_c, divval, width, height):

    
    start=datetime.now()    
    trnpatIDs, trnscores, trnscoresOA, _, _ = getDataFromCSV(trnpath, trnfilename, typeFeat)
    # print(len(trnpatIDs))
    X_train1 = getHOGForTrainData(trnpatIDs)
    # print(len(X_train1))


    tstpatIDs, tstscores, tstscoresOA, _, _ = getDataFromCSV(tstpath, tstfilename, typeFeat)
    print(len(tstpatIDs))
    X_test1 = getHOGForTrainData(tstpatIDs)
    # print(len(X_test1))
    print(datetime.now()-start)    

    with open(outputpath + '{}-1.pckl'.format(outputfilename1), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([X_train1, X_test1], f)

    X_train1, X_test1 = [], []

    width , height = int(width), int(height)

    print('autoencoder training started-{}'.format(typeFeat))
    start=datetime.now()

    resizedTrnOut = np.zeros((len(trnpatIDs), 32, 32))
    resizedTstOut = np.zeros((len(tstpatIDs), 32, 32))

    tf.keras.backend.set_floatx('float32')
    g = tf.Graph()
    for r in range(0, 600 - windowsize_r, windowsize_r):
        for c in range(0, 600 - windowsize_c, windowsize_c):
            start1=datetime.now()
            print("{} {} {} {}".format(r, r + windowsize_r, c, c + windowsize_c))
            X_train, X_test = getBlocks1(outputpath, outputfilename1, r, windowsize_r, c, windowsize_c)     

            X_train = tf.cast(np.reshape( 
                X_train, (X_train.shape[0],  
                          X_train.shape[1] * X_train.shape[2])), tf.float32) 
            X_test = tf.cast( 
                    np.reshape(X_test,  
                              (X_test.shape[0],  
                                X_test.shape[1] * X_test.shape[2])), tf.float32) 
              
            X_train, X_test = standard_scale(X_train, X_test)
            X_train = tf.reshape(X_train, (len(X_train), windowsize_r, windowsize_c, 1))
            X_test = tf.reshape(X_test, (len(X_test), windowsize_r, windowsize_c, 1))
            with g.container('experiment0'):    
                l2_regularizer = tf.keras.regularizers.l2(l2_reg)
                model = ConvNetAutoEncoder(l2_regularizer=l2_regularizer)
                optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
                global_step = tf.Variable(0)
                for epoch in range(num_epochs):
                    trnrecon = tf.Variable(0.)
                    for x in range(0, len(X_train), batch_size):
                        x_inp = X_train[x : x + batch_size, :, :, :]
                        loss_value, grads, _, _ = grad(model, x_inp, x_inp)        
                        optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)           
                        trnrecon.assign( tf.math.add( trnrecon, loss_value ) )
                    if epoch % 50 == 0:       
                        print("{},{}  Epoch: {}, Loss : {}".format(r, c, epoch, tf.math.divide(trnrecon, tf.Variable(len(X_train)).numpy()) ))   
                print("{},{}  Epoch: {}, Loss : {}".format(r, c, epoch, tf.math.divide(trnrecon, tf.Variable(len(X_train)).numpy()) ))   

            for x in range(0, len(X_test), batch_size):                
                x_inp = X_test[x : x + batch_size, :, :, :]
                tmp, _ = model(x_inp, is_training=False, training=False)
                tmp = tf.reshape(tmp, (int(batch_size), int(windowsize_r/divval), int(windowsize_c/divval))).numpy()
                resizedTstOut[x : int(x + batch_size), int(r/divval):int((r + windowsize_r)/divval), int(c/divval):int((c + windowsize_c)/divval)] = tmp

            for x in range(0, len(X_train), batch_size):                
                x_inp = X_train[x : x + batch_size, :, :, :]
                tmp, _ = model(x_inp, is_training=False, training=False)
                tmp = tf.reshape(tmp, (int(batch_size), int(windowsize_r/divval), int(windowsize_c/divval))).numpy()
                resizedTrnOut[x : int(x + batch_size), int(r/divval):int((r + windowsize_r)/divval), int(c/divval):int((c + windowsize_c)/divval)] = tmp
           
            # print( "experiment0 cleared" )
            print(datetime.now()-start1)
            print('autoencoder completed : {} {}'.format(r/divval, c/divval))
            # assert(False)

    print(datetime.now()-start)
    print('autoencoder training completed - {} All'.format(typeFeat))

    with open(outputpath + '{}-3.pckl'.format(outputfilename1), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([resizedTrnOut, resizedTstOut, trnpatIDs, trnscores, trnscoresOA, tstpatIDs, tstscores, tstscoresOA, width, height], f)

    print('autoencoder saved -{}'.format(typeFeat))
    resizedTrnOut, resizedTrnOut, trnpatIDs, trnscores, trnscoresOA, tstpatIDs, tstscores, tstscoresOA = [], [], [], [], [], [], [], []
    tf.keras.backend.clear_session()

def doRegression(typeFeat, trnpath, trnfilename, tstpath, tstfilename, outputpath, outputfilename, outputfilename1):
    def delfiles(model_dir):
        import glob        
        files = glob.glob(model_dir + "/*")
        for f in files:
            os.remove(f)
        # print('files deleted')

    with open(outputpath + '{}-3.pckl'.format(outputfilename1), 'rb') as f:  # Python 3: open(..., 'wb')
        resizedTrnOut, resizedTstOut, trnpatIDs, trnscores, trnscoresOA, tstpatIDs, tstscores, tstscoresOA, width, height = pickle.load(f)
    resizedTrnOut = np.reshape( resizedTrnOut, (resizedTrnOut.shape[0],  resizedTrnOut.shape[1] * resizedTrnOut.shape[2]))
    resizedTstOut = np.reshape(resizedTstOut,  (resizedTstOut.shape[0],  resizedTstOut.shape[1] * resizedTstOut.shape[2]))
      
    resizedTrnOut, resizedTstOut = standard_scale(resizedTrnOut, resizedTstOut)
    # print(resizedTrnOut.shape)
    # trnscores, tstscores = standard_scale(trnscores, tstscores)
    preprocessor = prep.StandardScaler().fit(trnscores) 
    trnscores = preprocessor.transform(trnscores) 
    # tstscores = preprocessor.transform(tstscores)     

    feat_cols=[x for x in range(0, resizedTrnOut.shape[1])]

    def input_fn( inp, inpscores, istraining=True):
        cont_cols = {str(k):tf.constant(inp[:,k]) for k in feat_cols}
        if istraining:
            # print(trnscores)
            return( cont_cols, tf.constant(inpscores) )
        else:
            return( tf.data.Dataset.from_tensors(cont_cols) )

    def train_input_fn():
        return input_fn(resizedTrnOut1, trnscores1)

    def test_input_fn():
        return input_fn(resizedTstOut, [], False)

    efeats=[]
    for cols in feat_cols:
        # print(cols.shape)
        column=tf.feature_column.numeric_column(str(cols))
        efeats.append(column)
    print(len(feat_cols))
    
    
    model_dir=outputpath+"tmp-{}".format(typeFeat)
    delfiles(model_dir)
    model=tf.estimator.DNNRegressor(hidden_units=[512, 31],feature_columns=efeats,  label_dimension=trnscores.shape[1], model_dir=model_dir, optimizer='Adam')

    tksz = int(resizedTrnOut.shape[0] * 0.91)
    # print(tksz)
    final_pred1 = np.zeros((tstscores.shape[0], tstscores.shape[1]))
    for ii in range(0,10):
        # delfiles(model_dir)
        # print(ii)
        npermu = np.random.permutation(resizedTrnOut.shape[0])
        npermu = npermu[:tksz]
        resizedTrnOut1 = resizedTrnOut[npermu, :]
        trnscores1 = trnscores[npermu, :]
        model.train(input_fn=train_input_fn,steps=4000)
        # train_metrics=model.evaluate(input_fn=input_func,steps=1000)
        # print('training fnshed')

        # function to return key for any value 
        def get_key(my_dict, val): 
          for value, key in my_dict.items(): 
            if val == value: 
              return key 

          return "key doesn't exist"

        preds=model.predict(input_fn=test_input_fn)
        predictions=list(preds)
        final_pred=[]
        for pred in predictions:
            final_pred.append(get_key(pred, 'predictions'))
        final_pred = preprocessor.inverse_transform(final_pred)     
        final_pred1 = final_pred1 + np.asarray(final_pred) 
        # print(final_pred1.shape)

    # print(ii)
    final_pred1 = np.divide(final_pred1, ii+1)
    out_tpl = np.nonzero(final_pred1 < 0.5)
    final_pred1[out_tpl]=0
    final_pred1 = np.around(final_pred1, decimals = 0)
    final_pred1 = final_pred1.astype(int)
    # print((np.square(tstscores - final_pred)).mean(axis=1))
    # print((np.square(tstscores - final_pred)).mean(axis=None))
    
    # mnerr = (np.square(tstscores - final_pred)).mean(axis=1)
    # print(mnerr)
    
    # print((np.square(tstscores - final_pred)).mean(axis=None))
    with open(outputpath + '{}-4.pckl'.format(outputfilename1), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(final_pred1, f)
    print('finished')
    # assert(False)


from datetime import datetime
import pickle

import sys
print(sys.version)

print(tf.__version__)
print('**'*200)
trnpath = '/train/'
trnfilename = 'training.csv'
# trnpath = '/content/gdrive/My Drive/Dataset/train/'
# trnfilename = 'training1.csv'
import os
if not os.path.exists(os.path.join(trnpath, trnfilename)):
    exit(0)
    
tstpath = '/test/'
tstfilename = 'template.csv'
# tstpath = '/content/gdrive/My Drive/Dataset/train/'
# filename = 'template.csv'

outputpath = '/output/'
outputfilename = 'predictions.csv'
# outputpath = '/content/gdrive/My Drive/output/'
# outputfilename = 'predictions.csv'

width = 512
height = 512
learning_rate = 0.001
l2_reg = 0.0001
num_epochs = 10
batch_size = 3
windowsize_r, windowsize_c, divval = 128, 128, 16

typeFeat='RH'
outputfilename1 = 'trndata-{}'.format(typeFeat)
createAutoencoderData(typeFeat, trnpath, trnfilename, tstpath, tstfilename, outputpath, outputfilename, outputfilename1,\
                      learning_rate, l2_reg, num_epochs, batch_size, windowsize_r, windowsize_c, divval, width, height)
doRegression(typeFeat, trnpath, trnfilename, tstpath, tstfilename, outputpath, outputfilename, outputfilename1)

typeFeat='RF'
outputfilename1 = 'trndata-{}'.format(typeFeat)
createAutoencoderData(typeFeat, trnpath, trnfilename, tstpath, tstfilename, outputpath, outputfilename, outputfilename1,\
                      learning_rate, l2_reg, num_epochs, batch_size, windowsize_r, windowsize_c, divval, width, height)
doRegression(typeFeat, trnpath, trnfilename, tstpath, tstfilename, outputpath, outputfilename, outputfilename1)

typeFeat='LH'
outputfilename1 = 'trndata-{}'.format(typeFeat)
createAutoencoderData(typeFeat, trnpath, trnfilename, tstpath, tstfilename, outputpath, outputfilename, outputfilename1,\
                      learning_rate, l2_reg, num_epochs, batch_size, windowsize_r, windowsize_c, divval, width, height)
doRegression(typeFeat, trnpath, trnfilename, tstpath, tstfilename, outputpath, outputfilename, outputfilename1)

typeFeat='LF'
outputfilename1 = 'trndata-{}'.format(typeFeat)
createAutoencoderData(typeFeat, trnpath, trnfilename, tstpath, tstfilename, outputpath, outputfilename, outputfilename1,\
                      learning_rate, l2_reg, num_epochs, batch_size, windowsize_r, windowsize_c, divval, width, height)
doRegression(typeFeat, trnpath, trnfilename, tstpath, tstfilename, outputpath, outputfilename, outputfilename1)


df = pd.read_csv(tstpath + tstfilename)
typeFeat='RH'
outputfilename1 = 'trndata-{}'.format(typeFeat)
_, _, _, colnames, colOAnames = getDataFromCSV(tstpath, tstfilename, typeFeat)
with open(outputpath + '{}-4.pckl'.format(outputfilename1), 'rb') as f:  # Python 3: open(..., 'wb')
  final_pred1 = pickle.load(f)
# print(len(colnames))
# print(final_pred1.shape)  
for ii in range(0, final_pred1.shape[0]):        
    df.loc[ii, colnames] = final_pred1[ii,:]

typeFeat='RF'
outputfilename1 = 'trndata-{}'.format(typeFeat)
_, _, _, colnames, colOAnames = getDataFromCSV(tstpath, tstfilename, typeFeat)
with open(outputpath + '{}-4.pckl'.format(outputfilename1), 'rb') as f:  # Python 3: open(..., 'wb')
  final_pred1 = pickle.load(f)
for ii in range(0, final_pred1.shape[0]):        
    df.loc[ii, colnames] = final_pred1[ii,:]

typeFeat='LH'
outputfilename1 = 'trndata-{}'.format(typeFeat)
_, _, _, colnames, colOAnames = getDataFromCSV(tstpath, tstfilename, typeFeat)
with open(outputpath + '{}-4.pckl'.format(outputfilename1), 'rb') as f:  # Python 3: open(..., 'wb')
  final_pred1 = pickle.load(f)
for ii in range(0, final_pred1.shape[0]):        
    df.loc[ii, colnames] = final_pred1[ii,:]

typeFeat='LF'
outputfilename1 = 'trndata-{}'.format(typeFeat)
_, _, _, colnames, colOAnames = getDataFromCSV(tstpath, tstfilename, typeFeat)
with open(outputpath + '{}-4.pckl'.format(outputfilename1), 'rb') as f:  # Python 3: open(..., 'wb')
  final_pred1 = pickle.load(f)
for ii in range(0, final_pred1.shape[0]):        
    df.loc[ii, colnames] = final_pred1[ii,:]

tmp = np.zeros((final_pred1.shape[0], 3))
tmp[:,1] = np.sum(df.values[:, 4:47], axis=1)
tmp[:,2] = np.sum(df.values[:, 48:], axis=1)
tmp[:,0] = tmp[:,1]+tmp[:,2]
tmp = tmp.astype(int)
for ii in range(0, final_pred1.shape[0]):        
    df.loc[ii, colOAnames] = tmp[ii,:]
df.to_csv(outputpath + outputfilename, index=False)
print('predictions.csv saved')

