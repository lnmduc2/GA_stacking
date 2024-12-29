from __init__ import *

def MobileNet_Model(img_size=128, isSummary=False):
    with tf.device('/GPU:0'):
        base_model = MobileNet(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))
        base_model.trainable = False

        model = Sequential()
        model.add(base_model)

        model.add(BatchNormalization())

        model.add(Dropout(0.45))
        model.add(Dense(220, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adamax(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        if isSummary:
            print(model.summary())

    return model


def ResNet50_Model(img_size=128, isSummary=False):
    with tf.device('/GPU:0'):
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))
        base_model.trainable = False

        model = Sequential()
        model.add(base_model)

        model.add(BatchNormalization())

        model.add(Dropout(0.45))
        model.add(Dense(220, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adamax(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        if isSummary:
            print(model.summary())

    return model


def VGG16_Model(img_size=128, isSummary=False):
    with tf.device('/GPU:0'):
        base_model = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))
        base_model.trainable = False

        model = Sequential()
        model.add(base_model)

        model.add(BatchNormalization())

        model.add(Dropout(0.45))
        model.add(Dense(220, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adamax(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        if isSummary:
            print(model.summary())

    return model


def Xception_Model(img_size=128, isSummary=False):
    with tf.device('/GPU:0'):
        base_model = Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))
        base_model.trainable = False

        model = Sequential()
        model.add(base_model)

        model.add(BatchNormalization())

        model.add(Dropout(0.45))
        model.add(Dense(220, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adamax(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        if isSummary:
            print(model.summary())

    return model


def LGBM_Model():
    return LGBMClassifier() 

def XGB_Model():
    return XGBClassifier() 

def CatBoost_Model():
    return CatBoostClassifier(verbose=0) 

def LogisticRegression_Model():
    return LogisticRegression()

def RandomForestClassifier_Model():
    return RandomForestClassifier()

def AdaBoostClassifier_Model():
    return AdaBoostClassifier()

def KNN_Model():
    return KNeighborsClassifier()

MODEL_FACTORY = {
    ## DL models
    'VGG16': VGG16_Model,
    'Xception': Xception_Model,
    'ResNet50': ResNet50_Model,
    'MobileNet': MobileNet_Model,
    # ML models
    'LogisticRegression': LogisticRegression_Model,
    'RandomForestClassifier': RandomForestClassifier_Model,
    'KNN': KNN_Model,
    'AdaBoostClassifier': AdaBoostClassifier_Model,
    'LGBM': LGBM_Model,
    'XGB': XGB_Model,
    'CatBoost': CatBoost_Model
}