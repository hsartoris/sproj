import numpy as np
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import prettify
import math
from SeqData2 import seqData2

SAVE_CKPT = True
# if you set this to False it will break
TBOARD_LOG = True

runNumber = 6
batchSize = 64
timesteps = 200
baseRate = .0001
initLearningRate = .05
#initLearningRate = 0.01 - baseRate
trainingSteps = 10000
epochLen = 100
prefix = "dataSmall"
pretty = prettify.pretty()
logPath = "/home/hsartoris/tflowlogs/"


b = timesteps   # time dimension subsampling. ignored in this test case as we are using 200 step chunks
# metalayers. let's try restricting to 1
d = 2
n = 3           # number of neurons

_data = tf.placeholder(tf.float32, [None, b, n])
#_data = tf.placeholder(tf.float32, [b, n])
_labels = tf.placeholder(tf.float32, [None, 1, n * n])
#_labels = tf.placeholder(tf.float32, [1, n * n])
dropout = tf.placeholder(tf.float32)

weights = { 'layer0': tf.Variable(tf.random_normal([d, 2*b])), 
        'layer2_in': tf.Variable(tf.random_normal([d, 2*d])), 
        'layer2_out': tf.Variable(tf.random_normal([d, 2*d])), 
        'final' : tf.Variable(tf.random_normal([1,d])) }

#weights = [tf.Variable(tf.random_normal([2*b, d])), tf.Variable(tf.random_normal([d, d])),  tf.Variable(tf.random_normal([d, 1]))]
#biases = [tf.Variable(tf.random_normal([d])), tf.Variable(tf.random_normal([1]))]
# biases not currently in use
biases = { 'layer0' : tf.Variable(tf.random_normal([d])), 'final' : tf.Variable(tf.random_normal([1])) }

expand = np.array([[1]*n + [0]*n*(n-1)])
for i in range(1, n):
    expand = np.append(expand, [[0]*n*i + [1]*n + [0]*n*(n-1-i)], 0)

expand = tf.constant(expand, tf.float32)

tile = np.array([([1] + [0]*(n-1))*n])
for i in range(1, n):
    tile = np.append(tile, [([0]*i + [1] + [0]*(n-1-i))*n], 0)

tile = tf.constant(tile, tf.float32);

def batchModel(x, weights):
    # effectively layer 0
    # this is a party
    total = tf.concat([tf.einsum('ijk,kl->ijl',x,expand), tf.tile(x,[1,1,n])], 1)
    #print(total.get_shape().as_list())
    layer1 = tf.nn.relu(tf.einsum('ij,kjl->kil', weights['layer0'], total))
    #print(layer1.get_shape().as_list())
    print("First layer output size:",layer1.get_shape().as_list())
    return layer1

def layer2batch(x, weights):
    a_total = tf.concat([tf.einsum('ijk,kl->ijl', tf.einsum('ijk,kl->ijl', x, tf.transpose(expand)), expand), x], 1)
    a_total = tf.einsum('ij,ljk->lik', weights['layer2_out'], a_total)
    b_total = tf.concat([x, tf.einsum('ijk,kl->ijl', tf.einsum('ijk,kl->ijl', x, tf.transpose(tile)), tile)], 1)
    b_total = tf.einsum('ij,ljk->lik', weights['layer2_in'], b_total)
    return tf.nn.relu(tf.add(a_total, b_total))

def finalBatch(x, weights):
    return tf.einsum('ij,ljk->lik', weights['final'], x)

#X, Y = iterator.get_next()
global_step = tf.Variable(0, trainable=False)

with tf.name_scope("rate"):
    #learningRate = tf.train.inverse_time_decay(initLearningRate, global_step, 1, .003)
    #learningRate = tf.train.exponential_decay(initLearningRate, global_step, 500, .4)
    #learningRate = tf.train.polynomial_decay(0.001, global_step, trainingSteps, .0001)
    learningRate = tf.placeholder(tf.float32, shape=[])

logits = finalBatch(layer2batch(batchModel(_data, weights), weights), weights)
#logits = final(layer2(model(_data, weights), weights), weights);
with tf.name_scope("Model"):
    #pred = tf.nn.softmax(logits)
    pred = tf.nn.tanh(logits)

with tf.name_scope("Loss"):
    #lossOp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=_labels))
    lossOp = tf.reduce_mean(tf.losses.mean_squared_error(_labels, pred, reduction=tf.losses.Reduction.NONE))
    #lossOp = tf.reduce_mean(tf.losses.hinge_loss(_labels, pred, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS))
    #lossOp = tf.reduce_sum(tf.losses.absolute_difference(_labels, pred, reduction=tf.losses.Reduction.NONE))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
optimizer = tf.train.AdamOptimizer(initLearningRate)
#optimizer = tf.train.MomentumOptimizer(initLearningRate, .001)
#optimizer = tf.train.AdagradOptimizer(initLearningRate)

with tf.name_scope("Optimizer"):
    trainOp = optimizer.minimize(lossOp, global_step=global_step)

correct = tf.equal(tf.argmax(pred, 1), tf.argmax(_labels, 1))

with tf.name_scope("Accuracy"):
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

if TBOARD_LOG:
    #rateSum = tf.summary.scalar("learn_rate", learningRate)
    lossSum = tf.summary.scalar("train_loss", lossOp)
    accSum = tf.summary.scalar("train_accuracy", accuracy)

if SAVE_CKPT:
    saver = tf.train.Saver()

trainMaxIdx = 1280
validMaxIdx = 1600
testMaxIdx  = 2000
training = seqData2(0, trainMaxIdx, prefix)
validation = seqData2(trainMaxIdx, validMaxIdx, prefix)
testing = seqData2(validMaxIdx, testMaxIdx, prefix)

with tf.Session() as sess:
    sess.run(init)

    if TBOARD_LOG:
        # initialize log writers
        summWriter = tf.summary.FileWriter(logPath + "/train" + str(runNumber), graph=tf.get_default_graph())
        validWriter = tf.summary.FileWriter(logPath + "/validation" + str(runNumber))

    if len(sys.argv) > 1:
        if not SAVE_CKPT: saver = tf.train.Saver()
        saver.restore(sess, sys.argv[1])
        testData = testing.data
        testLabels = testing.labels
        print("Accuracy on validation data:", sess.run(accuracy, feed_dict={_data: testData, _labels: testLabels}))
        testX, testY, _ = testing.next(1)
        print(sess.run(pred, feed_dict={_data: testX, _labels: testY}))
        print(weights['final'].eval())
        sys.exit()

    for step in range(trainingSteps):
        global_step += 1
        #print(weights['final'].eval())
        #if step < 1500: lr = baseRate + (initLearningRate * math.pow(.4, step/500.0))
        #else: lr = baseRate + (initLearningRate / (1 + .00975 * step))
        #lr = initLearningRate
        batchX, batchY, batchId = training.next(batchSize)
    # see status.1
        #sess.run(trainOp, feed_dict={_data: batchX,_labels: batchY, learningRate: lr})
        sess.run(trainOp, feed_dict={_data: batchX,_labels: batchY})
        if batchId == trainMaxIdx or step == 1:
            # end of epoch as signaled by SeqData
            # calculate current loss on training data
            #currRate, tLoss, tAcc, loss= sess.run([rateSum, lossSum, accSum, lossOp], 
            tLoss, tAcc, loss= sess.run([lossSum, accSum, lossOp], 
                    feed_dict={_data: batchX, _labels: batchY})
            # see status.1
            #        feed_dict={_data: batchX, _labels: batchY, learningRate: lr})

            # calculate validation loss
            validX, validY, _ = validation.next(batchSize*2)
            vloss, vacc, loss, acc= sess.run([lossSum, accSum, lossOp, accuracy], 
                    feed_dict={_data: validX, _labels: validY})

            if TBOARD_LOG:
                # log various data
                 validWriter.add_summary(vloss, step)
                 validWriter.add_summary(vacc, step)
                 summWriter.add_summary(tLoss, step)
                 summWriter.add_summary(tAcc, step)
            print("Step " + str(step) + ", batch loss = " + "{:.4f}".format(loss) + 
                   ", accuracy = " + "{:.3f}".format(acc))
           #print(weights['final'].eval())
        #pretty.arrow(step%epochLen, epochLen)
        pretty.arrow(batchId, trainMaxIdx)

        if step % 500 == 0 and SAVE_CKPT:
            save = saver.save(sess, "/home/hsartoris/tflowlogs/checkpoints" + 
                    str(runNumber) + "/trained" + str(step) + ".ckpt")
            print("Saved checkpoint " + str(step))

    if SAVE_CKPT:
        save = saver.save(sess, "/home/hsartoris/tflowlogs/checkpoints" + str(runNumber) + "/trained.ckpt")
        print("Training complete; model saved in file %s" % save)
    testData = testing.data
    testLabels = testing.labels
    print("Immediate OOM:", sess.run(accuracy, feed_dict={_data: testData, _labels: testLabels}))
