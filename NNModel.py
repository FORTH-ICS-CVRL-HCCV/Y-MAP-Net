
"""
Author : "Ammar Qammaz"
Copyright : "2024 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"
"""

import numpy as np
import keras
import tensorflow as tf
import time
from keras import layers, models
from keras.layers import Input, Layer, DepthwiseConv2D, UpSampling2D, Flatten, Dropout, SpatialDropout2D, Conv2D, ZeroPadding2D, Conv2DTranspose, Cropping2D, Concatenate, AvgPool2D, BatchNormalization, ReLU, Reshape, Dense, Add, UpSampling2D, MaxPooling2D, AveragePooling2D, Activation, MaxPool2D, Concatenate, Permute
from keras.models import Model
#-------------------------------------------------------------------------------
def retrieveModelOutputDimensions(model): 
    try:
      output_layer = model.get_layer(name="hm") #heatmap_output
    except:
      print("Could not find heatmap output layer, assuming that it is the last layer")
      output_layer = model.layers[0]  # Assuming the output layer is the first/ layer
      #output_layer = model.layers[-1]  # Assuming the output layer is the last layer
    output_shape = output_layer.output.shape
    output_size  = (output_shape[1],output_shape[2])
    numberOfHeatmaps = output_shape[3]
    print("Number of Heatmaps is ", numberOfHeatmaps)
    print("Output Shape is ", output_size)
    return output_shape[1],output_shape[2],output_shape[3]
#----------------------------------------------------------------------------------------
def load_keypoints_model(model_path):
    from tools import bcolors
    from NNLosses import HeatmapDistanceMetric, GloVeMSELoss, HeatmapCoreLoss, WeightedBinaryCrossEntropy, HeatmapDistanceMetricPartial, CustomTopKCategoricalAccuracy
    print(bcolors.OKGREEN,"Loading %s model.. " % model_path,bcolors.ENDC)
    try:
       #Regular keras loading until V3 that breaks
       import keras
       doModelCompilation = True #<- Does this have any effect on loading speed?

       print("Keras is NOW loading the saved model..")
       start      = time.time()
       model = keras.saving.load_model(model_path, custom_objects={
                                                        'HeatmapDistanceMetric':HeatmapDistanceMetric
                                                       ,'HeatmapDistanceMetricPartial':HeatmapDistanceMetricPartial
                                                       ,'GloVeMSELoss':GloVeMSELoss
                                                       ,'WeightedBinaryCrossEntropy':WeightedBinaryCrossEntropy
                                                       ,'HeatmapCoreLoss':HeatmapCoreLoss
                                                       ,'CustomTopKCategoricalAccuracy':CustomTopKCategoricalAccuracy
                                                     }, compile=doModelCompilation)#, safe_mode=True)
       seconds    = time.time() - start
       print("Loading the model took ",seconds," seconds..")

       input_layer  = model.layers[0]  # Assuming the input layer is the first layer

       #This works when we have one output
       #output_layer = model.layers[-1]  # Assuming the output layer is the last layer
       # Access the layer by name
       try:
         output_layer = model.get_layer(name="hm") #heatmap_output
       except:
         print("Could not find heatmap output layer, assuming that it is the last layer")
         output_layer = model.layers[0]  # Assuming the output layer is the first/ layer
         #output_layer = model.layers[-1]  # Assuming the output layer is the last layer
       

       input_shape  = input_layer.output.shape
       output_shape = output_layer.output.shape
       # Check the shape of the input layer
       input_size = (input_shape[1],input_shape[2]) #This is a little dodgie
       # Get the output layer of the model

       print("Output Shape ",output_shape)
       output_size = (output_shape[1],output_shape[2])
       numberOfHeatmaps = output_shape[3]
    except Exception as e:
       #Fallback to TF2 saved_model loading
       print(bcolors.FAIL,"An exception occurred trying to load keras model:", str(e),bcolors.ENDC)
       print(bcolors.OKGREEN,"Falling back to TF saved_model loader.. ",bcolors.ENDC)
       #model_path_dir = os.path.dirname(path)
       model = tf.saved_model.load(model_path)
       signatures = model.signatures
       signature_keys = signatures.keys()
       if 'serving_default' in signature_keys:
         signature_key = 'serving_default'
       else:
         signature_key = list(signature_keys)[0]  # Use the first signature if 'serving_default' is not available
       input_info   = signatures[signature_key].inputs
       output_info  = signatures[signature_key].outputs
       input_shape  = input_info[0].shape
       output_shape = output_info[0].shape
       input_size = (input_shape[1], input_shape[2])
       output_size = (output_shape[1], output_shape[2])
       numberOfHeatmaps = output_shape[0]  # Assuming the first dimension represents the number of heatmaps

    print("Input shape is ",input_size)
    print("Number of Heatmaps is ", numberOfHeatmaps)
    print("Output Shape is ", output_size)

    return model,input_size,output_size,numberOfHeatmaps
#-------------------------------------------------------------------------------
def test_model_IO(model):
    try:
        model.save('test.keras')
        load_keypoints_model('test.keras')
        import os
        os.system("rm test.keras")
    except Exception as e:
        raise ValueError('Failed testing model IO')
#-------------------------------------------------------------------------------
class ReshapeTiles(tf.keras.layers.Layer):
    def __init__(self, tile_height, tile_width, tile_depth, **kwargs):
        super(ReshapeTiles, self).__init__(**kwargs)
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.tile_depth = tile_depth

    def call(self, inputs):
        # Compute dynamic batch size
        batch_size = tf.shape(inputs)[0]
        num_tiles_h = tf.shape(inputs)[1]  # Height-wise tiles
        num_tiles_w = tf.shape(inputs)[2]  # Width-wise tiles

        # Flatten tile dimensions into a new batch dimension
        return tf.reshape(inputs, (batch_size * num_tiles_h * num_tiles_w, self.tile_height, self.tile_width, self.tile_depth))
#-------------------------------------------------------------------------------
# Corrected ReshapeBack layer
class ReshapeBack(tf.keras.layers.Layer):
    def __init__(self, num_tiles, tile_features, **kwargs):
        super(ReshapeBack, self).__init__(**kwargs)
        self.num_tiles = num_tiles
        self.tile_features = tile_features

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0] // (self.num_tiles * self.num_tiles)
        return tf.reshape(inputs, (batch_size, self.num_tiles, self.num_tiles, self.tile_features))
#-------------------------------------------------------------------------------
# Custom TileExtractor layer
class TileExtractor(tf.keras.layers.Layer):
    def __init__(self, tile_size, **kwargs):
        super(TileExtractor, self).__init__(**kwargs)
        self.tile_size = tile_size

    def call(self, inputs):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.tile_size, self.tile_size, 1],
            strides=[1, self.tile_size, self.tile_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        batch_size = tf.shape(inputs)[0]
        num_tiles = inputs.shape[1] // self.tile_size  # Assuming square input
        return tf.reshape(patches, [batch_size, num_tiles, num_tiles, self.tile_size, self.tile_size, 3])
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def positional_encoding(seq_len, model_dim):
    pos = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, model_dim, 2) * -(np.log(10000.0) / model_dim))
    pos_encoding = np.zeros((seq_len, model_dim))
    pos_encoding[:, 0::2] = np.sin(pos * div_term)
    pos_encoding[:, 1::2] = np.cos(pos * div_term)
    return pos_encoding
#-------------------------------------------------------------------------------
class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super(PositionalEncodingLayer, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        # Custom positional encoding logic here

    def call(self, inputs):
        # Forward logic for the layer
        return inputs + self.positional_encoding

    def get_config(self):
        # Include the custom arguments in the config
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "d_model": self.d_model,
        })
        return config
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def add_token_output(bridge_output, input_layer, numTokens, activation='leaky_relu', dropoutRate=0.2, nextTokenStrength=0.7, layer_width=1228, D=300, depth=7, dropoutDepth=4, use_learnable_residuals=False):
    """
    Adds a token output branch to the network based on the bridge output and input layer,
    including convolutional and max-pooling layers before the dense layers.
    
    Args:
    - bridge_output: The output of the U-Net bridge (last layer before decoder).
    - input_layer: The original input layer to add skip connections from.
    - numTokens: The number of tokens to predict.
    - activation: The activation function to use.
    - dropoutRate: Dropout rate for the token output layers (if any).
    
    Returns:
    - token_output: The final token output layer.
    """ 
    
    nextTokenStrength = max(1.0,nextTokenStrength)

    # Reshape only if the input is not already flattened
    if len(bridge_output.shape) > 2:  # Check if the output needs reshaping
        x = layers.Reshape((layer_width,))(bridge_output)
    else:
        x = bridge_output #Else use it without reshape

    #pos_encoding = positional_encoding(numTokens, 512)
    #x_with_pos = x + pos_encoding

    # Initialize a variable to hold the previous output for residual connections
    prev_output = None
    prev_glove  = None

    # Define multiple outputs, one for each embedding
    glove_outputs = []

    for i in range(numTokens):

        thisTokenDropout = dropoutRate/(i+1)
        #scaling_factor = layers.Dense(1, activation='sigmoid')(x)  # Learnable scaling factor
        
        # Sequential dense layers for each embedding
        x_token = layers.Dense(layer_width, activation=activation, name="Y_l1_t%u"%i)(x)
        x_token = layers.LayerNormalization()(x_token)
         
        if (thisTokenDropout>0.01):
          x_token = layers.Dropout(thisTokenDropout)(x_token) #Staggered dropout that gets lower as we progress to next tokens

        # Residual connection: add the previous token's output to the current one
        x_token_residual = x_token
        #Residual connection makes all tokens repeat the same thing 
        if prev_output is not None:

        # Learnable residual mixing
            if i > 0 and use_learnable_residuals==True:
                residual_factor_x    = layers.Dense(1, activation='sigmoid', name="res_factor_x_t%u"%i)(x_token_residual)
                residual_factor_prev = layers.Dense(1, activation='sigmoid', name="res_factor_prev_t%u"%i)(prev_output)
                x_token_residual     = layers.Multiply()([x_token_residual, residual_factor_x])
                prev_output          = layers.Multiply()([prev_output, residual_factor_prev])
            elif i > 0 and use_learnable_residuals==False:
                x_token_residual = keras.layers.Rescaling(1.0-nextTokenStrength, offset=0.0, name="Y_resrescale_t%u"%i)(x_token_residual) #<- Scale down importance of previous value
                prev_output      = keras.layers.Rescaling(nextTokenStrength,     offset=0.0, name="Y_rescale_t%u"%i)(prev_output) #<- Scale down importance of previous value
            #--------------------------------------------------------------------------------------------------------------------------------------------------------------------
            x_token_residual = layers.Dense(layer_width, activation=activation, name="Y_l2X_t%u"%i)(prev_output)
            x_token_residual = layers.LayerNormalization()(x_token_residual)
            #x_token_residual = layers.Dropout(dropoutRate)(x_token) #<- make it noisy ?
            #x_token_residual = layers.Dropout(dropoutRate)(x_token_residual) #<- make it noisy ?
            x_token_residual = layers.Add()([x_token, x_token_residual])
        else:
            #Also add the residual part for the first token(!)
            x_token_residual = layers.Dense(layer_width, activation=activation, name="Y_l2F_t%u"%i)(x_token)
            x_token_residual = layers.LayerNormalization()(x_token_residual)
            x_token_residual = layers.Dropout(thisTokenDropout)(x_token_residual)

        for d in range(3, depth + 1):  # Loop from depth 3 to the minimum of (depth, 7)
          x_token_residual = layers.Dense(layer_width, activation=activation, name=f"Y_l{d}_t{i}")(x_token_residual)
          x_token_residual = layers.LayerNormalization()(x_token_residual)
          # Apply dropout only for depth 3 and 4
          #if (d <= dropoutDepth) and (thisTokenDropout>0.01):
          x_token_residual = layers.Dropout(thisTokenDropout)(x_token_residual)
        #-------------------------------------------
        connect_to_next = x_token_residual #This is the next residual
        
        if prev_glove is not None:
           x_token = layers.Dense(D, activation=activation, name="Y_pre_final_combined_t%u"%i)(x_token_residual)
           x_token = layers.LayerNormalization()(x_token)
           x_token = layers.Add()([x_token, prev_glove])
        else:
           x_token = layers.Dense(D, activation=activation, name="Y_pre_final_t%u"%i)(x_token_residual)
           x_token = layers.LayerNormalization()(x_token)


        #This should be a tanh activation but in order for the appended network to have tanh try linear
        glove_output = layers.Dense(D, activation='tanh', name="t%02u"%i)(x_token)
        prev_glove   = glove_output
        #glove_output = layers.Multiply()([glove_output, scaling_factor])
        glove_outputs.append(glove_output)

        # Store the current token's output to be used as residual in the next iteration
        prev_output = connect_to_next #glove_output or connect_to_next

    """
    #EXPERIMENT
    exp = x
    j=0
    for i in range(numTokens):
    #    for j in range(i + 1, min(i+2,numTokens)):  # Avoid duplicate pairs
                    exp = layers.Dense(layer_width, activation=activation, name="Y_l1_t%02u_t%02u"% (i, j))(exp)
                    exp = layers.LayerNormalization()(exp)
                    exp = layers.Dropout(0.1)(exp) #Staggered dropout that gets lower as we progress to next tokens
                    exp = layers.Dense(layer_width, activation=activation, name="Y_l2_t%02u_t%02u"% (i, j))(exp)
                    exp = layers.LayerNormalization()(exp)
                    exp = layers.Dropout(0.1)(exp) #Staggered dropout that gets lower as we progress to next tokens
                    exp = layers.Dense(layer_width, activation=activation, name="Y_l3_t%02u_t%02u"% (i, j))(exp)
                    exp = layers.LayerNormalization()(exp)
                    exp = layers.Dropout(0.1)(exp) #Staggered dropout that gets lower as we progress to next tokens
                    exp = layers.Dense(layer_width, activation=activation, name="Y_l4_t%02u_t%02u"% (i, j))(exp)
                    exp = layers.LayerNormalization()(exp)
                    exp = layers.Dropout(0.1)(exp) #Staggered dropout that gets lower as we progress to next tokens
                    #exp_output = layers.Dense(D, activation='tanh', name="t%02u_t%02u" % (i, j))(exp)
                    exp_output = layers.Dense(D, activation='tanh', name="d%02u" % (i))(exp)
                    glove_outputs.append(exp_output)
    """

    return glove_outputs
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def append_final_multihot_layer(glove_outputs, bridge_output=None, D=300, TokensOut=8 ,Classes=2037, layer_width = 2048, depth = 2, activation='leaky_relu', dropoutRate=0.2, use_attention=False):
    # Concatenate all glove outputs into a single vector
    
    concatenated_output = layers.Concatenate()(glove_outputs)

    if use_attention:
        concatenated_output = layers.Reshape((TokensOut, D))(concatenated_output)  # Ensure (batch, seq_len, feature_dim)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=D//4)(concatenated_output, concatenated_output)
        concatenated_output = layers.Concatenate()([concatenated_output, attn])
    else:
        concatenated_output = layers.Reshape((D * TokensOut,))(concatenated_output)

    concatenated_output = layers.Flatten()(concatenated_output)

    #This actually works worse..
    if (bridge_output is None):
        print("Not connecting to bridge")
        x = concatenated_output
    else:
        bridge_output_flat  = layers.Flatten()(bridge_output)
        intermediateADapter = layers.Dense(layer_width, activation=activation, name="tmh_adapter")(concatenated_output)
        concatenated_output = layers.Add()([bridge_output_flat, intermediateADapter])
        x = concatenated_output
    
    for d in range(depth):  # Loop for desired depth (starting from 1) 
          x = layers.Dense(layer_width, activation=activation, name=f"tm{d}")(x)
          #layer_width = 2048 #<- If uncommented forces all layers after first to have this width 
          x = layers.LayerNormalization()(x)
          x = layers.Dropout(dropoutRate)(x)

    #Add a linear layer
    #x = layers.Dense(layer_width, activation='linear', name="tokens_scale")(x)

    # Add a single Dense layer of size D * maxtokens as the final output #NOT TokensOut
    token_output = layers.Dense(Classes, activation='sigmoid', name="tokens_multihot")(x)
 
    return token_output
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def append_descriptor_layer(bridge_layer=None, layer_width = 768, activation='leaky_relu', dropoutRate=0.1, depth = 1):
    # Concatenate all glove outputs into a single vector
    concatenated_output = layers.Flatten()(bridge_layer)
    x = concatenated_output  
    
    for d in range(depth):  # Loop for desired depth (starting from 1) 
          x = layers.Dense(layer_width, activation=activation, name=f"desc{d}")(x)
          x = layers.LayerNormalization()(x)
          x = layers.Dropout(dropoutRate)(x)
 
    # Add a single Dense layer of size D * maxtokens as the final output #NOT TokensOut
    token_output = layers.Dense(layer_width, activation='linear', name="descriptors")(x)
 
    return token_output
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def build_resnethybrid_cnn(input_shape, #num_classes, 
                           activation='leaky_relu',
                           gloveLayers = 7,
                           multihotLayers = 2,
                           bridgeLayerWidthCompatibility=0,
                           dropoutRate=0.3,
                           nextTokenStrength=0.3,
                           numTokens = 8,
                           numClasses = 2037,
                           use_learnable_residuals=True,
                           useDescriptors=False):
    from keras import layers, models
    import tensorflow as tf

    # Load ResNet50 without the top layers (include_top=False)

    base_model = tf.keras.applications.ResNet50(
                                                #name="pretrained",
                                                include_top=False,
                                                weights='imagenet',
                                                input_shape=input_shape,
                                                pooling='avg'  # Global Average Pooling
                                               )

    """
    base_model = tf.keras.applications.ConvNeXtSmall( 
    #model_name="pretrained",
    include_top=False,
    include_preprocessing=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax')
    """

    # Freeze the ResNet50 base model
    base_model.trainable = False

    # Create a new model
    inputs = layers.Input(shape=input_shape)

    # Apply the ResNet base model
    b = base_model(inputs)
    
    # Reshape only if the input is not already flattened
    #if len(b.shape) > 2:  # Check if the output needs reshaping
    #    b = layers.Reshape((2048,))(b) #The output of the resnet is 2048

    if len(b.shape) > 2:  # Check if the output needs reshaping
        b = layers.Reshape((7*7*768,))(b) #The output of the resnet is 2048
        #b = layers.Dense(2048, activation='gelu', name="scaledown")(b)
   
    if bridgeLayerWidthCompatibility!=0:
       print("Adding a compatibility layer (layer width=",bridgeLayerWidthCompatibility,") after resnet to make it more compatible with Y-MAP ")
       dim_product = bridgeLayerWidthCompatibility
       b = layers.Dense(int(bridgeLayerWidthCompatibility), activation='linear', name="resnet_width_adapter")(b)
     
    if (useDescriptors):
       descriptor_outputs = append_descriptor_layer(bridge_layer=b, layer_width = 768) 


    glove_outputs = add_token_output(b, inputs, numTokens, activation=activation, dropoutRate=dropoutRate, nextTokenStrength=nextTokenStrength, layer_width=dim_product, depth=gloveLayers, use_learnable_residuals=use_learnable_residuals)
    number_of_glove_output_tokens = len(glove_outputs)  #This should be the same as numTokens but make sure
 
    token_output = append_final_multihot_layer(glove_outputs,  bridge_output=b, D=300, TokensOut=number_of_glove_output_tokens ,Classes=numClasses, layer_width=dim_product, depth=multihotLayers)
     
    #-------------------------------
    modelOutputs = list()
    if (useDescriptors):
      modelOutputs.append(descriptor_outputs)
    modelOutputs.append(glove_outputs)
    modelOutputs.append(token_output)
    #-------------------------------
      
    model = Model(inputs, modelOutputs, name="ResnetBasedModel")

    # Print model summary
    model.summary()
    
    return model
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class AddGaussianNoise(tf.keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super(AddGaussianNoise, self).__init__()
        self.stddev = stddev

    def call(self, inputs, training=False):
        if training:
            noise = tf.random.normal(tf.shape(inputs), mean=0.0, stddev=self.stddev)
            return inputs + noise
        return inputs
#-------------------------------------------------------------------------------
def convnext_block(input, name, num_filters, activation='gelu', dropoutRate=0.0):
    shortcut = input
    
    # Depthwise convolution (like in ConvNeXt)
    x = DepthwiseConv2D(kernel_size=7, padding='same', name=f"dwconv_{name}_{num_filters}")(input)
    x = layers.LayerNormalization()(x)
    
    # Pointwise convolution (1x1) for feature transformation
    x = Conv2D(4 * num_filters, kernel_size=1, padding='same', name=f"pwconv1_{name}_{num_filters}")(x)
    x = Activation(activation)(x)
    
    # Another 1x1 convolution to bring it back to original num_filters
    x = Conv2D(num_filters, kernel_size=1, padding='same', name=f"pwconv2_{name}_{num_filters}")(x)
    
    # Residual connection
    x = Add(name=f"residual_{name}_{num_filters}")([shortcut, x])
    
    # Optional dropout
    if dropoutRate > 0.0:
        x = Dropout(dropoutRate, name=f"dropout_{name}_{num_filters}")(x)
    
    return x
#-------------------------------------------------------------------------------
def conv_block(input, name, num_filters, activation='leaky_relu', dropoutRate=0.0):
    # Save input for residual connection
    shortcut = input

    x = Conv2D(num_filters, 3, padding="same", name="conv2D_In_%s_%u_%s" % (name,num_filters,activation))(input)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = Conv2D(num_filters, 3, padding="same", name="conv2D_Out_%s_%u_%s" % (name,num_filters,activation))(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    # Optionally add dropout
    if dropoutRate > 0.0:
        x = SpatialDropout2D(dropoutRate, name="dropout_%s_%u_%u" % (name,num_filters, int(dropoutRate * 100)))(x)

    # Residual connection: if the input and output dimensions match, add them
    if shortcut.shape[-1] == num_filters:
        x = Add(name="residual_In_%s_%u_%s" % (name,num_filters,activation))([shortcut, x])
    else:
        # If dimensions do not match, apply a 1x1 convolution to match them
        shortcut = Conv2D(num_filters, kernel_size=1, padding='same')(shortcut)
        x = Add(name="residual_In_%s_%u_%s" % (name,num_filters,activation))([shortcut, x])

    return x
#-------------------------------------------------------------------------------
def encoder_block(input, num_filters, activation, dropoutRate=0.0,depth=None):
    #x = convnext_block(input, "encoder", num_filters, activation="gelu", dropoutRate=dropoutRate)
    if depth==None:
       name = "encoder"
    else:
       name = "encoder_d%u" % depth

    x = conv_block(input, name, num_filters, activation=activation, dropoutRate=dropoutRate)
    p = AveragePooling2D((2, 2))(x)  # Use AveragePooling2D
    return x, p
#-------------------------------------------------------------------------------
def decoder_block(input, skip_features, num_filters, activation, namePrefix="decoder"):
    # Transposed convolution (upsampling)
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same", name="conv2DT_%s_%u_%s" % (namePrefix, num_filters, activation))(input)

    # Check if spatial dimensions mismatch
    if (skip_features.shape[1] != x.shape[1]) or (skip_features.shape[2] != x.shape[2]):
        print(f"Incompatible skip {skip_features.shape[1]}, {skip_features.shape[2]} with {x.shape[1]}, {x.shape[2]}")
        crop_height = skip_features.shape[1] - x.shape[1]
        crop_width = skip_features.shape[2] - x.shape[2]

        # Crop skip_features if necessary
        if crop_height != 0 or crop_width != 0:
            skip_features = Cropping2D(cropping=((crop_height // 2, crop_height - (crop_height // 2)),
                                                (crop_width // 2, crop_width - (crop_width // 2))))(skip_features)

    # Concatenate upsampled input with cropped skip features
    x_concat = Concatenate(name="concat_%s_%u_%s" % (namePrefix, num_filters, activation))([x, skip_features])

    # Pass through the convolution block after concatenation
    x_conv = conv_block(x_concat, "decoder", num_filters, activation=activation)

    # Add residual connection (input `x` before conv_block) back to the output of the conv_block
    x_residual = Add(name="residual_%s_%u_%s" % (namePrefix, num_filters, activation))([x, x_conv])

    return x_residual
#-------------------------------------------------------------------------------
#A more powerful block in the bridge, such as dilated convolutions or even an atrous spatial pyramid pooling (ASPP) module, which can capture multi-scale context better.
def bridge_block(input, num_filters, activation, layers = 3):
    
    x = Conv2D(num_filters, 3, padding="same", dilation_rate=1, activation=activation, name="BridgeStart")(input)
    x = BatchNormalization()(x)
    
    if (layers>=2):
      x = Conv2D(num_filters, 3, padding="same", dilation_rate=2, activation=activation, name="BridgeMid")(x)
      x = BatchNormalization()(x)

    if (layers>=3):
      x = Conv2D(num_filters, 3, padding="same", dilation_rate=4, activation=activation, name="BridgeFinish")(x)
      x = BatchNormalization()(x)

    return x
#-------------------------------------------------------------------------------
def build_unet(inputHeight, 
               inputWidth, 
               inputChannels,
               outputWidth,
               outputHeight,
               numKeypoints,
               num16BitHeatmaps=1,
               numTokens = 0,
               numClasses = 2037,
               minHeatmapValue=-120,
               maxHeatmapValue= 120,
               growthBase=1.4,#2 default
               baseChannels=64,
               pixelwiseChannels=0,
               encoderRepetitions=7,
               decoderRepetitions=7,
               midSectionRepetitions=3,
               gloveLayers = 7,
               multihotLayers = 2,
               use_learnable_residuals=True,
               bridgeRatio=1.0,
               forceBridgeSize=0,
               activation='leaky_relu',
               gaussianNoiseSTD=0.0,
               dropoutRate=0.0,
               nextTokenStrength=0.5,
               quantize=False,
               serial="",
               useDescriptors=False):

    #Uncomment to go "single
    ###if (num16BitHeatmaps>0):
    #      DEACTIVATED 16BIT without changing 8bit
    #      numKeypoints = numKeypoints + num16BitHeatmaps
    ###       num16BitHeatmaps  = 0

    input_shape  = (inputWidth, inputHeight, inputChannels)
    output_shape = (outputWidth, outputHeight, numKeypoints)

    inputs       = Input(input_shape, name="input")
    after_input  = keras.layers.Rescaling(scale=1.0/255.0, name="float_scaling")(inputs)

    #Fixed : https://github.com/keras-team/keras/issues/19589 ?     
    if (gaussianNoiseSTD>0.0):
        after_input = AddGaussianNoise(gaussianNoiseSTD, name='training_gaussian_noise')(after_input)

    # Encoder blocks
    encoder_layers = []
    p = after_input
    for i in range(encoderRepetitions):
        thisLayerFilters = int(baseChannels * (growthBase**i))
        if (thisLayerFilters<numKeypoints):
              thisLayerFilters = numKeypoints #Always have at least the number of outputs to not strangle them
        s, p = encoder_block(p, thisLayerFilters, activation, dropoutRate=dropoutRate, depth=i)
        encoder_layers.append(s)

    # UNET Bridge layer
    #b = conv_block(p, "bridge", int(bridgeRatio * baseChannels * (growthBase**encoderRepetitions)), activation=activation) #<- Standard one layer bridge
    if (forceBridgeSize==0):
          forceBridgeSize = int(bridgeRatio * baseChannels * (growthBase**encoderRepetitions))
    b = bridge_block(p, forceBridgeSize, activation=activation, layers=midSectionRepetitions)           #<- CHANGE More complex bridge


    if (useDescriptors):
       descriptor_outputs = append_descriptor_layer(bridge_layer=b, layer_width = 768) 

    # Decoder blocks
    decoder_layers = [b]
    for i in range(decoderRepetitions):
        thisLayerFilters = int(baseChannels * 2**(encoderRepetitions-i-1))
        if (thisLayerFilters<numKeypoints):
              thisLayerFilters = numKeypoints #Always have at least the number of outputs to not strangle them
        d = decoder_block(decoder_layers[-1], encoder_layers[-(i+1)], thisLayerFilters, activation)
        decoder_layers.append(d)

    #Final layer calculations 
    print("Out layer ",decoder_layers[-1].shape)
    UNETDimW       = decoder_layers[-1].shape[1]
    UNETDimH       = decoder_layers[-1].shape[2]
    print("Out layer dims ",UNETDimW,",",UNETDimH)
    print("Requested final layer ",outputWidth,",",outputHeight)
    stride_amount = 2
    if (outputHeight>=UNETDimH) or (outputWidth>=UNETDimW):
         stride_amount = 1
         print("Force stride=(1,1) to keep network size (Rule 1)")
    if (outputHeight-UNETDimH<0) or (outputWidth-UNETDimW<0):
         stride_amount = 1
         print("Force stride=(1,1) to keep network size (Rule 2)")

    #Heatmap output size and magnitude control
    #----------------------------------------------------------------------
    #Using tanh to force output in [-1..1] avoid over/underflow of data

    # Adding the pixelwise convolution (1x1 convolution) as the last layer
    if (pixelwiseChannels!=0):
      pixelwise = Conv2D(pixelwiseChannels, kernel_size=(1, 1), activation=activation, name="pixelwise")(decoder_layers[-1]) #activation = None ?
    else:
      pixelwise = decoder_layers[-1]
   
    heatmap_output      = Conv2D(filters=numKeypoints, kernel_size=1, padding="same", activation="tanh", name="hm_tanh_8bit")(pixelwise) # Why where there -> strides=(stride_amount, stride_amount)
    # Maintain the desired output dimensions
    if (outputHeight != heatmap_output.shape[1]) or (outputWidth != heatmap_output.shape[2]):
            print("Incompatible final skip ", heatmap_output.shape[1], ",", heatmap_output.shape[2])
            print(" with ", outputHeight, ",", outputWidth)
            crop_height = heatmap_output.shape[1] - outputHeight
            crop_width  = heatmap_output.shape[2] - outputWidth
            if (crop_height<0) or (crop_width<0): 
               print("Network has smaller output than what was requested..")
            else:
               heatmap_output = Cropping2D(cropping=((crop_height // 2, crop_height - (crop_height // 2)), (crop_width // 2, crop_width - (crop_width // 2))) , name="final_crop")(heatmap_output)
    #Scale back to min/max Heatmap values ( [-120..120] )
    scale_factor   = max(abs(minHeatmapValue),abs(maxHeatmapValue))
    heatmap_output = keras.layers.Rescaling(scale=scale_factor,name="hm")(heatmap_output) #int8_scaling / heatmap_output
    print("Final Scale Factor is ",scale_factor)

    # 16Bit output (if it is enabled..)
    #----------------------------------------------------------------------
    heatmaps_16bit_declared = False
    heatmaps_16bit = None
    if (num16BitHeatmaps>0):
        heatmaps_16bit_declared = True
        heatmaps_16bit = Conv2D(filters=num16BitHeatmaps, kernel_size=1, padding="same", activation="tanh", name=f"hm_tanh_16bit", strides=(stride_amount, stride_amount))(decoder_layers[-1])
        if (outputHeight != heatmaps_16bit.shape[1]) or (outputWidth != heatmaps_16bit.shape[2]):
                crop_height = heatmaps_16bit.shape[1] - outputHeight
                crop_width  = heatmaps_16bit.shape[2] - outputWidth
                if (crop_height >= 0) and (crop_width >= 0):
                    heatmaps_16bit = Cropping2D(cropping=((crop_height // 2, crop_height - (crop_height // 2)), (crop_width // 2, crop_width - (crop_width // 2))), name=f"final_crop_16bit")(heatmaps_16bit)
        scale_factor_16b = 32767 #/ scale_factor # This should be something around ~273
        heatmaps_16bit = keras.layers.Rescaling(scale=scale_factor_16b, name="hm_16b")(heatmaps_16bit) 
        print("Final 16-bit Scale Factor is ",scale_factor_16b) #32767.0

    # Token output (if it is enabled..)
    #----------------------------------------------------------------------
    modelOutputs = []
    if (numTokens>0):
      #numTokens = 1 #TEST tokens_to_classes
      #numTokens = 8 #TEST tokens_to_glove
 
      #This works until Keras 3.4.0 and Tensorflow 2.17.0
      #modelOutputs = add_token_output(b, inputs, numTokens, activation=activation, dropoutRate=dropoutRate)     
      #token_output = append_final_multihot_layer(modelOutputs, D=300, TokensOut=numTokens ,Classes=2037)
      #modelOutputs.append(token_output)


      # Get shape of tensor b
      b_shape = b.shape  # Returns a tuple-like object
      # Compute the product of spatial and channel dimensions
      dim_product = 1
      for dim in b_shape[1:]:  # Ignore batch size (b_shape[0] is None)
          if dim is not None:
              dim_product *= dim
      print("Bridge dimension is ",dim_product, ", FYI this will be used as the layer width for glove/token outputs!")


      modelOutputs  = list()
      glove_outputs = add_token_output(b, inputs, numTokens, activation=activation, dropoutRate=dropoutRate, nextTokenStrength=nextTokenStrength, layer_width=int(dim_product), depth=gloveLayers, use_learnable_residuals=use_learnable_residuals)
      number_of_glove_output_tokens = len(glove_outputs) #This should be the same as numTokens but make sure
      token_output  = append_final_multihot_layer(glove_outputs, bridge_output=b,  D=300, TokensOut=number_of_glove_output_tokens ,Classes=numClasses, layer_width = int(dim_product), depth=multihotLayers) #  
      
      if (heatmaps_16bit_declared):
          modelOutputs.append(heatmap_output)
          modelOutputs.append(heatmaps_16bit)
          if (useDescriptors):
             modelOutputs.append(descriptor_outputs)
          modelOutputs.extend(glove_outputs)
          modelOutputs.append(token_output)
          model = Model(inputs,modelOutputs, name="Y-Net-v%s"%serial)
      else:
          modelOutputs.append(heatmap_output)
          if (useDescriptors):
             modelOutputs.append(descriptor_outputs)
          modelOutputs.extend(glove_outputs)
          modelOutputs.append(token_output)
          model = Model(inputs,modelOutputs, name="Y-Net-v%s"%serial)
    else:
      if (heatmaps_16bit_declared):
          modelOutputs.append(heatmap_output)
          modelOutputs.append(heatmaps_16bit)
          model = Model(inputs, modelOutputs, name="U-Net-v%s"%serial)
      else:
          model = Model(inputs, heatmap_output, name="U-Net-v%s"%serial)

    # Quantization aware model ( if it is enabled..)
    #----------------------------------------------------------------------
    if quantize:
        try:
            import tensorflow_model_optimization as tfmot
            model = tfmot.quantization.keras.quantize_model(model)
            print("Using quantized model..!")
        except Exception as e:
            print("An exception occurred:", str(e))
            print("Could not quantize model..!")
            print("https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/python/core/quantization/keras/quantize_aware_activation.py")

    model.summary()
    return model
#============================================================================================
#============================================================================================
# Main Function
#============================================================================================
#============================================================================================
if __name__ == '__main__':
     inputWidth     = 210
     inputHeight    = 210
     inputChannels  = 3
     print("Image size %ux%u:%u" % (inputWidth,inputHeight,inputChannels) )

     outputWidth    = 96
     outputHeight   = 96
     heatmaps       = 19
     print("Heatmap size %ux%u:%u" % (outputWidth,outputHeight,heatmaps) )

            #MB     #KB   # Bytes
     GPU = 49140 * 1024  * 1024
     sampleSize = (inputWidth * inputHeight * inputChannels) + (outputWidth * outputHeight * heatmaps) 
 
     COCO17trainval = 123287
     COCO17trainvalOnlyHuman = 65000

     print("GPU size ",GPU," bytes")
     print("Sample size ",sampleSize," bytes")
     print("Sample Number that fits : ",int(GPU/sampleSize)," ")
     print("COCO17 train+val takes : %0.2f %% of an A6000" % (100 * float(COCO17trainval*sampleSize/GPU)) )
     print("COCO17 train+val (only humans) takes : %0.2f %% an A6000" % (100 * float(COCO17trainvalOnlyHuman*sampleSize/GPU)) )

     GPU = 16376 * 1024  * 1024
     print("COCO17 train+val takes : %0.2f %% of an RTX4080" % (100 * float(COCO17trainval*sampleSize/GPU)) )
     print("COCO17 train+val (only humans) takes : %0.2f %% an RTX4080" % (100 * float(COCO17trainvalOnlyHuman*sampleSize/GPU)) )

     GPU = 8192 * 1024  * 1024
     print("COCO17 train+val takes : %0.2f %% of an GTX1070" % (100 * float(COCO17trainval*sampleSize/GPU)) )
     print("COCO17 train+val (only humans) takes : %0.2f %% an GTX1070" % (100 * float(COCO17trainvalOnlyHuman*sampleSize/GPU)) )
