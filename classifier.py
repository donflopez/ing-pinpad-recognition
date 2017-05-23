import sys
import tensorflow as tf
import numpy
import PIL
from PIL import Image, ImageFilter, ImageEnhance

class Classifier:

    x = tf.placeholder(tf.float32, [None, 10800])
    W = tf.Variable(tf.zeros([10800, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    saver = tf.train.Saver()
    keep_prob = tf.placeholder(tf.float32)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "model/mymodel")

    prediction = tf.argmax(y, 1)
    def getNumber(self, fil):
        img = Image.open(fil).convert('RGB')
        data = img.getdata()
        narr = numpy.array(data)
        narr = narr.flatten()
        narr = [ (255-j)*1.0/255.0 for j in narr]
        res = self.prediction.eval(feed_dict={self.x: [narr], self.keep_prob: 0.5}, session=self.sess)
        number = str(res[0])
        return number