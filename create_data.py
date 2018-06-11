import os
import sys
import numpy as np
import tensorflow as tf
from scipy import signal
from numpy.fft import fft as FFT;
from numpy.fft import ifft as IFFT;
from numpy.fft import fftfreq as FREQS;

files = []
labels = []
imageDir = "../../data/"
for filename in os.listdir(imageDir):
	if filename.startswith("nfibers109_617"):		
		try:
			delay = 0;
			with open(imageDir + filename) as f:
				for line in f:
					delay = float(line.split("=")[1])
					break
			labels.append(delay)
			files.append(filename)

		except:
			continue

c = list(zip(files, labels))
np.random.shuffle(c)
files, labels = zip(*c)

train_files = files[0:int(0.85*len(files))]
train_labels = labels[0:int(0.85*len(labels))]

val_files = files[int(0.85*len(files)):int(0.925*len(files))]
val_labels = labels[int(0.85*len(labels)):int(0.925*len(labels))]

test_files = files[int(0.925*len(files)):]
test_labels = labels[int(0.925*len(labels)):]

def load_image(filename):
	image = np.loadtxt(imageDir + filename)
	FFT_PLACEHOLDER = np.zeros((image.shape[0], 128, 2))
        
	FFT_IMAGE = FFT(image,axis=1)
        
	FFT_PLACEHOLDER[:,:,0] = np.abs(FFT_IMAGE)[:,:128];
	FFT_PLACEHOLDER[:,:,1] = np.diff(np.unwrap(np.angle(FFT_IMAGE), axis = 1), axis = 1)[:,:128];
        
	return FFT_PLACEHOLDER

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

train_filename = 'train.tfrecords'
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)
for i in range(len(train_files)):
	if not i % 1000:
        	print('Train data: {}/{}'.format(i, len(train_files)))
        	sys.stdout.flush()
    
	img = load_image(train_files[i])
	label = train_labels[i]
    
	feature = {'train/label': _float_feature(label), 'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    
	example = tf.train.Example(features=tf.train.Features(feature=feature))
    
	writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()

val_filename = 'val.tfrecords'
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(val_filename)
for i in range(len(val_files)):
        if not i % 1000:
                print('Val data: {}/{}'.format(i, len(val_files)))
                sys.stdout.flush()

        img = load_image(val_files[i])
        label = val_labels[i]

        feature = {'val/label': _float_feature(label), 'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

test_filename = 'test.tfrecords'
# open the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)
for i in range(len(test_files)):
        if not i % 1000:
                print('Test data: {}/{}'.format(i, len(test_files)))
                sys.stdout.flush()

        img = load_image(test_files[i])
        label = test_labels[i]

        feature = {'test/label': _float_feature(label), 'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()

