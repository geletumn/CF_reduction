#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Cross-fusion model architecture

from tensorflow import keras
from tensorflow.keras import layers

img_size = (384,1248)
ldr_size= (384,1248) # projected lidar image size
class_num = 3 # model output channels


# Fusion: weighting
class c_fusion_wt (layers.Layer):
    def __init__(self,**kwargs):
        super(c_fusion_wt, self).__init__(**kwargs)
    
    def build (self, input_shape):
        self.w=self.add_weight(shape=(1,), initializer="zero", trainable=True,name='weight') 
        
    def call(self, layer_1):
        return layer_1*self.w 

    def get_config(self):
        config = super(c_fusion_wt, self).get_config()
        return config


# cross fusion  model architecture
def get_model(img_size, ldr_size, num_classes):
    
    inputs_img=keras.Input(shape=img_size+(3,), name='rgb')
    inputs_ldr=keras.Input(shape=ldr_size+(3,), name='lidar')
    
    #x: image processing brach
    #y: lidar processing brach
    
    # Encoder
    #B1: Block 1
    x=layers.ZeroPadding2D(padding=1, name='Block1_rgb_zp')(inputs_img)
    x=layers.Conv2D(32,4,strides=2,activation='elu', name='Block1_rgb_conv')(x)
    y=layers.ZeroPadding2D(padding=1, name='Block1_lidar_zp')(inputs_ldr)
    y=layers.Conv2D(32,4,strides=2,activation='elu', name='Block1_lidar_conv')(y)
    # Fusion
    x_fsn=c_fusion_wt(name='Block1_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block1_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block1_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block1_lidar_cf')([y_fsn,y])
    #B2: Block 2
    x=layers.ZeroPadding2D(padding=1,name='Block2_rgb_zp')(x_fsn)
    x=layers.Conv2D(32,3,strides=1,activation='elu',name='Block2_rgb_conv')(x) 
    y=layers.ZeroPadding2D(padding=1,name='Block2_lidar_zp')(y_fsn)
    y=layers.Conv2D(32,3,strides=1,activation='elu', name='Block2_lidar_conv')(y) 
    # Fusion
    x_fsn=c_fusion_wt(name='Block2_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block2_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block2_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block2_lidar_cf')([y_fsn,y])
    #B3: Block 3
    x=layers.ZeroPadding2D(padding=1,name='Block3_rgb_zp')(x_fsn)
    x=layers.Conv2D(64,4,strides=2,activation='elu', name='Block3_rgb_conv')(x)  
    y=layers.ZeroPadding2D(padding=1, name='Block3_lidar_zp')(y_fsn)
    y=layers.Conv2D(64,4,strides=2,activation='elu', name='Block3_lidar_conv')(y)   
    # Fusion
    x_fsn=c_fusion_wt(name='Block3_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block3_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block3_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block3_lidar_cf')([y_fsn,y])
    #B4: Block 4
    x=layers.ZeroPadding2D(padding=1, name='Block4_rgb_zp')(x_fsn)
    x=layers.Conv2D(64,3,strides=1,activation='elu', name='Block4_rgb_conv')(x) 
    y=layers.ZeroPadding2D(padding=1, name='Block4_lidar_zp')(y_fsn)
    y=layers.Conv2D(64,3,strides=1,activation='elu', name='Block4_lidar_conv')(y)
    # Fusion
    x_fsn=c_fusion_wt(name='Block4_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block4_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block4_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block4_lidar_cf')([y_fsn,y])
    #B5: Block 5
    x=layers.ZeroPadding2D(padding=1,name='Block5_rgb_zp')(x_fsn)
    x=layers.Conv2D(128,4,strides=2,activation='elu', name='Block5_rgb_conv')(x) 
    y=layers.ZeroPadding2D(padding=1, name='Block5_lidar_zp')(y_fsn)
    y=layers.Conv2D(128,4,strides=2,activation='elu', name='Block5_lidar_conv')(y)  
    # Fusion
    x_fsn=c_fusion_wt(name='Block5_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block5_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block5_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block5_lidar_cf')([y_fsn,y])
    
    # Context module
    #B6: Block 6
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(1,1),activation='elu', name='Block6_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block6_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(1,1),activation='elu', name='Block6_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block6_lidar_dp')(y)
    # Fusion
    x_fsn=c_fusion_wt(name='Block6_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block6_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block6_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block6_lidar_cf')([y_fsn,y])
    #B7: Block 7
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(1,1),activation='elu', name='Block7_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block7_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(1,1),activation='elu', name='Block7_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block7_lidar_dp')(y)
    # Fusion
    x_fsn=c_fusion_wt(name='Block7_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block7_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block7_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block7_lidar_cf')([y_fsn,y])
    #B8: Block 8
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(1,2),activation='elu', name='Block8_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block8_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(1,2),activation='elu', name='Block8_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block8_lidar_dp')(y)
    # Fusion
    x_fsn=c_fusion_wt(name='Block8_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block8_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block8_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block8_lidar_cf')([y_fsn,y])
    #B9: Block 9
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(2,4),activation='elu', name='Block9_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block9_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(2,4),activation='elu', name='Block9_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block9_lidar_dp')(y)
    # Fusion
    x_fsn=c_fusion_wt(name='Block9_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block9_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block9_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block9_lidar_cf')([y_fsn,y])
    #B10: Block 10
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(4,8),activation='elu', name='Block10_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block10_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(4,8),activation='elu', name='Block10_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block10_lidar_dp')(y)
    # Fusion
    x_fsn=c_fusion_wt(name='Block10_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block10_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block10_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block10_lidar_cf')([y_fsn,y])
    #B11: Block 11
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(8,16),activation='elu', name='Block11_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block11_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(8,16),activation='elu', name='Block11_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block11_liar_dp')(y)
    # Fusion
    x_fsn=c_fusion_wt(name='Block11_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block11_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block11_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block11_lidar_cf')([y_fsn,y])
    #B12: Block 12
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(16,32),activation='elu', name='Block12_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block12_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(16,32),activation='elu', name='Block12_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block12_lidar_dp')(y)
    # Fusion
    x_fsn=c_fusion_wt(name='Block12_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block12_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block12_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block12_lidar_cf')([y_fsn,y])
    #B13: Block 13
    x=layers.Conv2D(128,3,padding="same",dilation_rate=(1,1),activation='elu', name='Block13_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block13_rgb_dp')(x)
    y=layers.Conv2D(128,3,padding="same",dilation_rate=(1,1),activation='elu', name='Block13_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block13_lidar_dp')(y)
    # Fusion
    x_fsn=c_fusion_wt(name='Block13_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block13_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block13_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block13_lidar_cf')([y_fsn,y])
    #B14: Block 14
    x=layers.Conv2D(128,1,padding="same",activation='elu', name='Block14_rgb_conv')(x_fsn)
    x=layers.Dropout(0.25, name='Block14_rgb_dp')(x)
    y=layers.Conv2D(128,1,padding="same",activation='elu', name='Block14_lidar_conv')(y_fsn)
    y=layers.Dropout(0.25, name='Block14_lidar_dp')(y)
    # Fusion
    x_fsn=c_fusion_wt(name='Block14_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block14_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block14_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block14_lidar_cf')([y_fsn,y])
    # Decoder
    #B15: Block 15
    x=layers.Conv2DTranspose(64,4,strides=2,activation='elu', padding='same', name='Block15_rgb_convtp')(x_fsn)
    y=layers.Conv2DTranspose(64,4,strides=2,activation='elu', padding='same', name='Blodck15_lidar_convtp')(y_fsn)
    # Fusion
    x_fsn=c_fusion_wt(name='Block15_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block15_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block15_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block15_lidar_cf')([y_fsn,y])
    #B16: Block 16
    x=layers.ZeroPadding2D(padding=1,name='Block16_rgb_zp')(x_fsn)
    x=layers.Conv2D(64,3,strides=1,activation='elu', name='Block16_rgb_conv')(x)
    y=layers.ZeroPadding2D(padding=1, name='Block16_lidar_zp')(y_fsn)
    y=layers.Conv2D(64,3,strides=1,activation='elu', name='Block_16_lidar_conv')(y)
    # Fusion
    x_fsn=c_fusion_wt(name='Block16_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block16_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block16_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block16_lidar_cf')([y_fsn,y])
    #B17: Block 17
    x=layers.Conv2DTranspose(32,4,strides=2,activation='elu', padding='same', name='Block17_rgb_convtp')(x_fsn)
    y=layers.Conv2DTranspose(32,4,strides=2, activation='elu', padding='same', name='Block17_lidar_convtp')(y_fsn)
    # Fusion
    x_fsn=c_fusion_wt(name='Block17_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block17_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block17_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block17_lidar_cf')([y_fsn,y])
    #B18: Block 18
    x=layers.ZeroPadding2D(padding=1, name='Block18_rgb_zp')(x_fsn)
    x=layers.Conv2D(32,3,strides=1,activation='elu', name='Block18_rgb_conv')(x)
    y=layers.ZeroPadding2D(padding=1, name='Block18_lidar_zp')(y_fsn)
    y=layers.Conv2D(32,3,strides=1,activation='elu', name='Block18_lidar_conv')(y)
    # Fusion
    x_fsn=c_fusion_wt(name='Block18_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block18_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block18_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block18_lidar_cf')([y_fsn,y])
    #B19: Block 19
    x=layers.Conv2DTranspose(8,4,strides=2, activation='elu', padding='same', name='Block19_rgb_convtp')(x_fsn)
    y=layers.Conv2DTranspose(8,4,strides=2,activation='elu', padding='same', name='Block19_lidar_convtp')(y_fsn)
    # Fusion
    x_fsn=c_fusion_wt(name='Block19_lidar_cfw')(y)
    x_fsn=layers.Add(name='Block19_rgb_cf')([x_fsn,x])
    y_fsn=c_fusion_wt(name='Block19_rgb_cfw')(x)
    y_fsn=layers.Add(name='Block19_lidar_cf')([y_fsn,y])
    #B20: Block 20
    x=layers.ZeroPadding2D(padding=1, name='Block20_rgb_zp')(x_fsn)
    x=layers.Conv2D(class_num,3,strides=1, name='Block20_rgb_conv')(x) 
    y=layers.ZeroPadding2D(padding=1, name='Block20_lidar_zp')(y_fsn)
    y=layers.Conv2D(class_num,3,strides=1, name='Block20_lidar_conv')(y) 
    # Fusion
    y_fsn=c_fusion_wt(name='Block20_lidar_cfw')(y)
    x_fsn=c_fusion_wt(name='Block20_rgb_cfw')(x)
    fsn=layers.Add(name='final_cf')([y_fsn,x_fsn])
    #B21: Block 21
    output=layers.Activation('softmax', name='output')(fsn)
        
    # Define the model
    model=keras.Model(inputs=[inputs_img, inputs_ldr], outputs=output)
    return model

#Build model
model = get_model(img_size, ldr_size, class_num)
model.summary()
#keras.utils.plot_model(model, show_shapes=True)

# save model
model.save('model_arch_cf')

