
import matplotlib.pyplot as plt
import tensorflow as tf
import glob

# jpgFilenamesList = glob.glob('145592*.jpg')
# cnt=0
def printImages(images,outputs,dic, fileName='Val_images'):
    pngFilenamesList = glob.glob(dic +'/' + 'Val_images_*.png')
    cnt=0
    for fname in pngFilenamesList:
        x=fname.split('_')[2].split('.')[0]
        wrkCnt=int(x)+1
        cnt= max((cnt,wrkCnt))
    grid_col = 8
    grid_row = 2
    f, axarr = plt.subplots(grid_row, grid_col, figsize=(grid_col*2, grid_row*2))
    i = 0
    for row in range(0, grid_row, 2):
        for col in range(grid_col):
            im=tf.cast(images[i],dtype=tf.float32)
            impred=tf.cast(outputs[i],dtype=tf.float32)
            axarr[row,col].imshow(im)
            axarr[row,col].axis('off')
            axarr[row+1,col].imshow(impred)
            axarr[row+1,col].axis('off')        
            i += 1
    f.tight_layout( 0.1, h_pad=0.2, w_pad=0.1)        
    plt.savefig(dic +'/' + fileName + "_" + str(cnt))
    cnt +=1
    plt.close()