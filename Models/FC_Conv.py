####### Importing Libraries
import numpy as np
import tensorflow as tf                                                                              

####### FC-Conv
rate_regularizer = 1e-5
class SC_Conv2D(tf.keras.layers.Layer):

    """ 
    This is inhe1rited class from keras.layers and shall be instatition of self-calibrated convolutions
    """
    
    def __init__(self,num_filters,kernel_size,num_features,pool_size,conv2_kernel_size):
    
        #### Defining Essentials
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_features = num_features # Number of Channels in Input
        self.pool_size = pool_size # Pool Size of the Average Pooling Layer
        self.conv2_kernel_size = conv2_kernel_size # Size of Local Attention i.e. Conv2 Filter Kernel

        #### Defining Layers
        self.conv2 = tf.keras.layers.Conv2D(self.num_features/2,self.conv2_kernel_size,kernel_regularizer=tf.keras.regularizers.l2(rate_regularizer),dtype='float32',activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(self.num_features/2,self.kernel_size,padding='same',kernel_regularizer=tf.keras.regularizers.l2(rate_regularizer),dtype='float32',activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(self.num_filters/2,self.kernel_size,padding='same',activation='linear',kernel_regularizer=tf.keras.regularizers.l2(rate_regularizer),dtype='float32')
        self.conv1 = tf.keras.layers.Conv2D(self.num_filters/2,self.kernel_size,padding='same',activation='linear',kernel_regularizer=tf.keras.regularizers.l2(rate_regularizer),dtype='float32')
    
        self.BN_X4 = tf.keras.layers.BatchNormalization()
        self.BN_X1 = tf.keras.layers.BatchNormalization()

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'num_features': self.num_features,
            'pool_size' : self.pool_size, 
            'conv2_kernel_size': self.conv2_kernel_size
        })
        return config
    
    
    def call(self,X):
       
        """
          INPUTS : 1) X - Input Tensor of shape (batch_size,H,W,num_features)
          OUTPUTS : 1) X - Output Tensor of shape (batch_size,H,W,num_features)
        """
        
        #### Dimension Extraction
        b_s = (X.shape)[0] 
        H = (X.shape)[1]
        W = (X.shape)[2]
        num_features = (X.shape)[3]
        
        #### Channel-Wise Division
        X_attention = X[:,:,:,0:int(self.num_features/2)]
        X_global = X[:,:,:,int(self.num_features/2):]
        
        #### Self Calibration Block

        ### Local Feature Detection
        
        ## Down-Sampling
        X_down_sampled = tf.keras.layers.AveragePooling2D(self.pool_size,strides=(1,1))(X_attention)

        ## Convoluting Down Sampled Image
        X_down_conv = self.conv2(X_down_sampled)
    
        ## Up-Sampling
        X_local_upsampled = tf.keras.layers.experimental.preprocessing.Resizing(X_attention.shape[1],X_attention.shape[2],interpolation='bilinear')(X_down_conv)
        
        ## Local-CAM
        X_local = tf.keras.layers.Add()([X_local_upsampled,X_attention])

        ## Local Importance 
        X_2 = tf.keras.activations.sigmoid(X_local)

        ### Self-Calibration

        ## Global Convolution
        X_3 = self.conv3(X_attention)

        ## Attention Determination
        X_attention = tf.math.multiply(X_2,X_3)

        #### Self-Calibration Feature Extraction
        X_4 = self.conv4(X_attention)
        X_4 = self.BN_X4(X_4)
        X_4 = tf.keras.layers.ReLU()(X_4)

        #### Normal Feature Extraction
        X_1 = self.conv1(X_global)
        X_1 = self.BN_X1(X_1)
        X_1 = tf.keras.layers.ReLU()(X_1)

        #### Concatenating and Returning Output
        return (tf.keras.layers.concatenate([X_1,X_4],axis=3))