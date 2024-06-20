####### Importing Libraries
import numpy as np
import tensorflow as tf                                                                              

####### CAM
class CAM_Module(tf.keras.layers.Layer):
    
    "Custom Layer for Computing Channel Attention,inherited from keras.layers base class"
    
    def __init__(self,num_features,beta):
        
        ##### Defining Instatiations
        super().__init__()
        self.num_features = num_features # Number of channels in the input 
        self.beta = beta # Weighting factor for Attention
                 
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_features': self.num_features,
            'beta' : self.beta
        })
        return config       
    
    def call(self,X):
       
        """
          INPUTS : 1) X - Input Tensor of shape (batch_size,H,W,num_features)
          OUTPUTS : 1) X - Output Tensor of shape (batch_size,H,W,num_features)
        """
        
        ##### Dimension Extraction
        b_s = (X.shape)[0] 
        H = (X.shape)[1]
        W = (X.shape)[2]
        N = int(H*W)
        C_val = self.num_features
        
        ##### Channel Attention Computation
        
        A1 = tf.keras.layers.Reshape((N,C_val))(X)
        A2 = tf.keras.layers.Reshape((N,C_val))(X)
        A2 = tf.keras.layers.Permute((2,1))(A2)
        
        Q = tf.linalg.matmul(A2,A1)
        Q = tf.keras.layers.Softmax(axis=1)(Q)
        Q = tf.linalg.matmul(A1,Q)
        Q = tf.keras.layers.Reshape((H,W,C_val))(Q)
        
        M = tf.keras.layers.Add()([self.beta*Q,X])
        return M