import numpy as np
import pandas as pd
from PIL import Image
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
from keras.models import model_from_json
from scipy.optimize import fmin_l_bfgs_b
import pickle
import time
import os
from datetime import datetime

# comment out following line if running on computer with no gpu
K.tensorflow_backend._get_available_gpus()
os.environ['KMP_DUPLICATE_LIB_OK']='True'

content_path = 'data/template/'
input_style_image_name = 'input_style.jpg'
input_content_image_name = 'input_content.jpg'
generated_image_name = 'output.jpg'
file_output_transitional_prefix = 'transition_image'


loadWeightsFromDisk = True
saveTransitionalImages = True
shouldSaveModel = True
saveModelAfter = 20
last_saved = saveModelAfter - 1
save_timestamp = None
history_is_blank = False
# current and total number of iterations (defaults: 0, 600)
iteration = 0
iterations = 700

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Specify paths for 1) content image 2) style image and 3) generated image
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

cImPath = content_path + 'imgs/' + input_content_image_name
sImPath = content_path + 'imgs/' + input_style_image_name
genImOutputPath = content_path + 'imgs/' + generated_image_name


print('\n\n\n\n========> Starting Style Transfer using content from ' +  content_path + ' <======== \n\n\n\n')
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Image processing
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
targetHeight = 512
targetWidth = 512
targetSize = (targetHeight, targetWidth)

cImageOrig = Image.open(cImPath)
cImageSizeOrig = cImageOrig.size
cImage = load_img(path=cImPath, target_size=targetSize)
cImArr = img_to_array(cImage)
cImArr = K.variable(preprocess_input(np.expand_dims(cImArr, axis=0)), dtype='float32')

sImage = load_img(path=sImPath, target_size=targetSize)
sImArr = img_to_array(sImage)
sImArr = K.variable(preprocess_input(np.expand_dims(sImArr, axis=0)), dtype='float32')

gIm0 = np.random.randint(256, size=(targetWidth, targetHeight, 3)).astype('float64')
gIm0 = preprocess_input(np.expand_dims(gIm0, axis=0))
gIm0_dims = gIm0.size

gImPlaceholder = K.placeholder(shape=(1, targetWidth, targetHeight, 3))

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## Define loss and helper functions
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

def get_feature_reps(x, layer_names, model):
    featMatrices = []
    for ln in layer_names:
        selectedLayer = model.get_layer(ln)
        featRaw = selectedLayer.output
        featRawShape = K.shape(featRaw).eval(session=tf_session)
        N_l = featRawShape[-1]
        M_l = featRawShape[1]*featRawShape[2]
        featMatrix = K.reshape(featRaw, (M_l, N_l))
        featMatrix = K.transpose(featMatrix)
        featMatrices.append(featMatrix)
    return featMatrices

def get_content_loss(F, P):
    cLoss = 0.5*K.sum(K.square(F - P))
    return cLoss

def get_Gram_matrix(F):
    G = K.dot(F, K.transpose(F))
    return G

def get_style_loss(ws, Gs, As):
    sLoss = K.variable(0.)
    for w, G, A in zip(ws, Gs, As):
        M_l = K.int_shape(G)[1]
        N_l = K.int_shape(G)[0]
        G_gram = get_Gram_matrix(G)
        A_gram = get_Gram_matrix(A)
        sLoss+= w*0.25*K.sum(K.square(G_gram - A_gram))/ (N_l**2 * M_l**2)
    return sLoss

def get_total_loss(gImPlaceholder, alpha=1, beta=10000.0):
    F = get_feature_reps(gImPlaceholder, layer_names=[cLayerName], model=gModel)[0]
    Gs = get_feature_reps(gImPlaceholder, layer_names=sLayerNames, model=gModel)
    #TODO: define a variable called contentLoss and set it equal to get_content_loss(F, P)
    
    #TODO: define a variable called styleLoss and set it equal to get_style_loss(ws, Gs, As)
    
    #TODO: cacluate totalLoss using alpha*contentLoss + beta*styleLoss
    
    return totalLoss

def calculate_loss(gImArr):
    """
    Calculate total loss using K.function
    """
    
    global iteration
    if (shouldSaveModel):
        saveModel(gImArr)
    iteration += 1
    if gImArr.shape != (1, targetWidth, targetWidth, 3):
        gImArr = gImArr.reshape((1, targetWidth, targetHeight, 3))
    loss_fcn = K.function([gModel.input], [get_total_loss(gModel.input)])

    return loss_fcn([gImArr])[0].astype('float64')

def get_grad(gImArr):
    """
    Calculate the gradient of the loss function with respect to the generated image
    """
    if gImArr.shape != (1, targetWidth, targetHeight, 3):
        gImArr = gImArr.reshape((1, targetWidth, targetHeight, 3))
    grad_fcn = K.function([gModel.input], K.gradients(get_total_loss(gModel.input), [gModel.input]))
    grad = grad_fcn([gImArr])[0].flatten().astype('float64')
    return grad

def postprocess_array(x):
    # Zero-center by mean pixel
    if x.shape != (targetWidth, targetHeight, 3):
        x = x.reshape((targetWidth, targetHeight, 3))
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x

def reprocess_array(x):
    x = np.expand_dims(x.astype('float64'), axis=0)
    x = preprocess_input(x)
    return x

def save_original_size(x, target_size=cImageSizeOrig, output_path=genImOutputPath):
    xIm = Image.fromarray(x)
    xIm = xIm.resize(target_size)
    xIm.save(output_path)
    return xIm

def saveModel(gImArr):
    global iteration, iterations, saveModelAfter, saveTransitionalImages, save_timestamp
    if (iteration % saveModelAfter == 0 and (iteration > last_saved or history_is_blank)):
        print('saving weights after iteration ', iteration)
        cModel.save_weights(content_path + 'model/cModel-' + str(iteration) + '.h5')
        sModel.save_weights(content_path + 'model/sModel-' + str(iteration) + '.h5')
        gModel.save_weights(content_path + 'model/gModel-' + str(iteration) + '.h5')
        pickle.dump(gImArr, open(content_path + 'model/gImArr-' + str(iteration) + '.p', 'wb'))
        print('weights saved')

        if(saveTransitionalImages):
            xOut = postprocess_array(gImArr)
            output_path = content_path + 'transitional/' + str(file_output_transitional_prefix) + '-' + str(iteration) + '.jpg'
            xIm = save_original_size(xOut, output_path=output_path)
            print('Image saved')

        history  = open(content_path + 'model/history.txt', 'a')
        if (save_timestamp != None):
            remaining_mins = ((time.time() - save_timestamp) / (60)) * ((iterations - iteration) / saveModelAfter) 
            print('minutes remaining until program finishes: ', str(remaining_mins))
            print(time.time(), save_timestamp, iterations, iteration, saveModelAfter)
        save_timestamp = time.time()
        dt_object = datetime.fromtimestamp(save_timestamp)
        history.write(str(iteration) + ' ' + str(dt_object) + '\n')
        history.close()
        print('model/history.txt updated')

        remove_files = [content_path + 'model/cModel-' + str(iteration - saveModelAfter) + '.h5',
                        content_path + 'model/sModel-' + str(iteration - saveModelAfter) + '.h5',
                        content_path + 'model/gModel-' + str(iteration - saveModelAfter) + '.h5',
                        content_path + 'model/gImArr-' + str(iteration - saveModelAfter) + '.p']
        for filePath in remove_files:
            if os.path.exists(filePath):
                os.remove(filePath)
        print('old model files deteted')
        
        



tf_session = K.get_session()
cModel = VGG16(include_top=False, weights='imagenet', input_tensor=cImArr)
sModel = VGG16(include_top=False, weights='imagenet', input_tensor=sImArr)
gModel = VGG16(include_top=False, weights='imagenet', input_tensor=gImPlaceholder)
x_val = gIm0.flatten()

if (loadWeightsFromDisk):
    # load current iteration
    fileHandle = open (content_path + 'model/history.txt',"r" )
    lineList = fileHandle.readlines()
    fileHandle.close()
    if (len(lineList) != 0): 
        # get last line of document, Calculate loss is called 6 times at start up
        last_line = lineList[len(lineList)-1]
        iteration = int(last_line.split()[0]) # Get the iteration number
        last_saved = iteration
        iteration -= 3
        iterations -= iteration 
        
        print('\nloaded history, starting up at iteration: ', iteration)
        print('iterations left: ', iterations)
        print('previous iteration / last saved: ', last_line)
        
        # load model weights
        cModel.load_weights(content_path + 'model/cModel-' + str(last_saved) + '.h5')
        sModel.load_weights(content_path + 'model/sModel-' + str(last_saved) + '.h5')
        gModel.load_weights(content_path + 'model/gModel-' + str(last_saved) + '.h5')
        x_val = pickle.load(open(content_path + 'model/gImArr-' + str(last_saved) + '.p', 'rb'))


        print('weights loaded from disk')
    else:
        print('model/history.txt is empty, starting from scratch.')
        history_is_blank = True

cLayerName = 'block4_conv2'
sLayerNames = [
                'block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                #'block5_conv1'
                ]

P = get_feature_reps(x=cImArr, layer_names=[cLayerName], model=cModel)[0]
As = get_feature_reps(x=sImArr, layer_names=sLayerNames, model=sModel)
ws = np.ones(len(sLayerNames))/float(len(sLayerNames))



start = time.time()
xopt, f_val, info= fmin_l_bfgs_b(calculate_loss, x_val, fprime=get_grad,
                            maxiter=iterations, disp=True)
xOut = postprocess_array(xopt)
xIm = save_original_size(xOut)
print('Image saved')
end = time.time()
print('Time taken: {}'.format(end-start))