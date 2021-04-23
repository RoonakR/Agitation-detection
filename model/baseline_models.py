from keras.layers import Dense, TimeDistributed, LSTM, Bidirectional, Input, Flatten, UpSampling2D
from keras import Sequential, Model, utils
from keras.optimizers import Adam
from keras import initializers
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3


def deepBiLSTM(timesteps, n_features, n_class, n_layer, n_units, drop_out, lr, loss):
    # define model
    model = Sequential()
    model.add(Bidirectional(LSTM(units=n_units, return_sequences=True, recurrent_dropout=drop_out), 
                            input_shape=(timesteps, n_features)))
    for i in range(1, n_layer):
        model.add(Bidirectional(LSTM(units=n_units, return_sequences=True, recurrent_dropout=drop_out)))
    model.add(Flatten())
    # in binary classification there is no difference between softmax and sigmoid but for multi-class we should use softmax for
    # cross-entropy loss
    model.add(Dense(n_class, activation="sigmoid"))  # a dense layer as suggested by neuralNer
    model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=['acc'])
    model.summary()

    return model


def deepLSTM(timesteps, n_features, n_class, n_layer, n_units, drop_out, lr, loss, active='relu'):
    # define model
    model = Sequential()
    model.add(LSTM(n_units, activation=active, return_sequences=True, input_shape=(timesteps, n_features)))
    for i in range(1, n_layer):
        model.add(LSTM(n_units, activation=active, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(n_class, activation="sigmoid"))
    model.compile(optimizer=Adam(lr=lr), loss=loss, metrics=['acc'])
    model.summary()

    return model


def ResNet_model(n_feat=8, n_channel=3, n_class=2, lr=0.0001, loss='binary_crossentropy', 
                 metrics=['accuracy'], n_layers=1, n_units=50):
     
    inp = Input(shape=(n_feat, n_feat, n_channel))
    up_inp = UpSampling2D((4, 4))(inp)
    res_model = ResNet50(weights='imagenet', include_top=False, input_tensor=up_inp)
    res_model.trainable = False
    res_output = res_model.output
    
    x = Flatten()(res_output)
    for i in range(0, n_layers):
        x = Dense(n_units, activation='relu', kernel_initializer="random_normal",
                          bias_initializer=initializers.Zeros())(x)
    x = Dense(n_class, activation='sigmoid', kernel_initializer="random_normal",
                          bias_initializer=initializers.Zeros(), name='predictions')(x)
    # create model
    model = Model(inputs=inp, outputs=x)
    opt = Adam(learning_rate=lr)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    # summarize model
    #model.summary()
    return model


def VGG_model(n_feat=8, n_channel=3, n_class=2, lr=0.0001, loss='binary_crossentropy', 
                 metrics=['accuracy'], n_layers=1, n_units=50):
    
    inp = Input(shape=(n_feat, n_feat, n_channel))
    up_inp = UpSampling2D((4, 4))(inp)
    vgg_model = VGG16(weights='imagenet', include_top=False, input_tensor=up_inp)
    vgg_model.trainable = False
    vgg_output = vgg_model.output

    x = Flatten(name='flatten')(vgg_output)
    for i in range(0, n_layers):
        x = Dense(n_units, activation='relu', kernel_initializer="random_normal",
                          bias_initializer=initializers.Zeros())(x)
    x = Dense(n_class, activation='sigmoid', kernel_initializer="random_normal",
                          bias_initializer=initializers.Zeros(), name='predictions')(x)
    model = Model(inputs=inp, outputs=x)
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
        
    # summarize model
    #model.summary()
    return model


def Inception_model(n_feat=8, n_channel=3, n_class=2, lr=0.0001, loss='binary_crossentropy', 
                 metrics=['accuracy'], n_layers=1, n_units=50):
    
    inp = Input(shape=(n_feat, n_feat, n_channel))
    up_inp = UpSampling2D((16, 16))(inp)
    inc_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=up_inp)
    inc_model.trainable = False
    inc_output = inc_model.output
    
    x = Flatten()(inc_output)
    for i in range(0, n_layers):
        x = Dense(n_units, activation='relu', kernel_initializer="random_normal",
                              bias_initializer=initializers.Zeros())(x)
    x = Dense(n_class, activation='sigmoid', kernel_initializer="random_normal",
                          bias_initializer=initializers.Zeros(), name='predictions')(x)
    # create model
    model = Model(inputs=inp, outputs=x)
    opt = Adam(learning_rate=lr)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    # summarize model
    #model.summary()
    return model