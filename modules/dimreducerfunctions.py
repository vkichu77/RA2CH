import tensorflow as tf
import tensorflow.keras.layers as layers
from modelclasses import ConvNetAutoEncoderDimReduction
from datetime import datetime

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        encoded, reconstruction = model(inputs)
        loss_value = loss(targets, reconstruction)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), reconstruction, encoded

def loss(x, x_bar):
    return tf.reduce_sum(tf.keras.losses.MSE(x, x_bar))


def saveDimReductionAutoEncoderModels(trn_data, outputmodelfiledir, learning_rate, l2_reg, num_epochs, batch_size, ckptlist):

    random.seed(555)
    np.random.seed(seed=1111)
    tf.random.set_seed(seed=2345)

    start = datetime.now()
    if not os.path.exists(outputmodelfiledir):
        os.makedirs(outputmodelfiledir)

    print('DIM eduction Autoencoder started LR-{}, l2-{}, num_epochs-{}, batch_size-{}'.format(
        learning_rate, l2_reg, num_epochs, batch_size))
    tf.keras.backend.set_floatx('float32')
    g = tf.Graph()
    with g.container('experiment0'):
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        l2_regularizer = tf.keras.regularizers.l2(l2_reg)
        ConvNetDR = ConvNetAutoEncoderDimReduction(
            l2_regularizer=l2_regularizer, name='3_layer_DR')
        ckpt = tf.train.Checkpoint(step=tf.Variable(
            1), optimizer=optimizer, net=ConvNetDR)
        manager = tf.train.CheckpointManager(
            ckpt, outputmodelfiledir, max_to_keep=1)

        # dfval = None
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print("Save DR Restored from {}".format(manager.latest_checkpoint))
            ckpt.step.assign_add(1)
            # dfval = pd.read_csv(outputmodelfiledir + 'model.csv')
        else:
            print("Save DR Initializing from scratch.")

        dfvalues = []
        global_step = tf.Variable(0)
        for epoch in range(1, num_epochs+1):
            trnrecon = tf.Variable(0., dtype=tf.float32)
            for x in range(0, trn_data.shape[0], batch_size):
                if x + batch_size > trn_data.shape[0]:
                    x_inp = trn_data[x: x +
                                     trn_data.shape[0] - batch_size, :, :, :]
                else:
                    x_inp = trn_data[x: x + batch_size, :, :, :]
                # print(x_inp.shape)
                loss_value, grads, _, _ = grad(ConvNetDR, x_inp, x_inp)
                optimizer.apply_gradients(
                    zip(grads, ConvNetDR.trainable_variables), global_step)
                trnrecon.assign(tf.math.add(trnrecon, loss_value))
            avgLoss = trnrecon.numpy() / np.ceil(trn_data.shape[0]/batch_size)
            # print('{} {}'.format(epoch, int(ckpt.step)))
            if int(ckpt.step) in ckptlist:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(
                    int(ckpt.step), save_path))
                print("loss {:1.2f}".format(avgLoss))
                dataset = pd.DataFrame(
                    dfvalues, columns=['epoch', 'loss', 'cktp'])
                if os.path.exists(outputmodelfiledir + 'model.csv'):
                    dfval = pd.read_csv(outputmodelfiledir + 'model.csv')
                    dataset = dfval.append(dataset, ignore_index=True)
                dataset.to_csv(outputmodelfiledir + 'model.csv', index=False)
                dfvalues = []
                deltfeventoutfiles(outputmodelfiledir)
            if int(ckpt.step) % 100 == 0:
                print("Epoch: {:5g} Loss: {:.2f} : {:5g}".format(
                    epoch, avgLoss, int(ckpt.step)))
            dfvalues.append([epoch, avgLoss, int(ckpt.step)])
            ckpt.step.assign_add(1)

        # print("Epoch: {}, Loss : {}: {}".format(epoch, avgLoss, int(ckpt.step)))

        tf.saved_model.save(ConvNetDR, outputmodelfiledir + 'model')
        # dataset = pd.DataFrame({'epoch': dfvalues[:, 0], 'Loss': dfvalues[:, 1], 'cktp': dfvalues[:, 2]})
        # if dfval is not None:
        #     dataset = dfval.append(dataset, ignore_index=True)
        # dataset.to_csv(outputmodelfiledir + 'model.csv', index=False)
        print('model saved')

    deltfeventoutfiles(outputmodelfiledir)
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    print('Dim Reduction autoencoder completed --{}'.format(datetime.now()-start))


def applyDimReductionAutoEncoderModels(trn_data, tst_data, outputfilename, outputmodelfiledir, learning_rate, l2_reg):

    random.seed(555)
    np.random.seed(seed=1111)
    tf.random.set_seed(seed=2345)

    start = datetime.now()
    batch_size = 1
    # trn_data, tst_data = trn_data.astype(np.float32), tst_data.astype(np.float32)
    width, height = tf.shape(trn_data)[2], tf.shape(trn_data)[1]
    print('Applying Autoencoder Models started LR-{}, l2-{}, batch_size-{}'.format(
        learning_rate, l2_reg, batch_size))
    tf.keras.backend.set_floatx('float32')
    g = tf.Graph()
    with g.container('experiment0'):
        ConvNetDR = tf.saved_model.load(outputmodelfiledir + 'model')

        trnNoiseOut = np.zeros((trn_data.shape[0], int(
            height/4), int(width/4), 1), dtype=np.float32)
        trnNoiseOut1 = np.zeros(
            (trn_data.shape[0], height, width, 1), dtype=np.float32)
        trnLossValue = 0.0
        print(trn_data.dtype)
        for x in range(0, trn_data.shape[0]):
            x_inp = tf.reshape(trn_data[x, :, :, :],
                               shape=(1, height, width, 1))
            x_inp = tf.convert_to_tensor(x_inp, dtype=tf.float32, name='x')
            tmp1, tmp = ConvNetDR(x_inp)
            loss_value = loss(x_inp, tmp)
            trnNoiseOut[x, :, :, :] = tf.reshape(
                tmp1, shape=(1, int(height/4), int(width/4), 1)).numpy()
            trnNoiseOut1[x, :, :, :] = tmp
            trnLossValue = trnLossValue + loss_value.numpy()
        trnLossValue = trnLossValue/trn_data.shape[0]

        tstNoiseOut = np.zeros((tst_data.shape[0], int(
            height/4), int(width/4), 1), dtype=np.float32)
        tstLossValue = 0.0
        for x in range(0, tst_data.shape[0]):
            x_inp = tf.reshape(tst_data[x, :, :, :],
                               shape=(1, height, width, 1))
            x_inp = tf.convert_to_tensor(x_inp, dtype=tf.float32, name='x')
            tmp1, tmp = ConvNetDR(x_inp)
            loss_value = loss(x_inp, tmp)
            tstNoiseOut[x, :, :, :] = tf.reshape(
                tmp1, shape=(1, int(height/4), int(width/4), 1)).numpy()
            tstLossValue = tstLossValue + loss_value.numpy()
        tstLossValue = tstLossValue/tst_data.shape[0]

        print("loss1-{:1.4f} loss2-{:1.4f}".format(trnLossValue, tstLossValue))
        saveToPickleFile(outputfilename + '.pckl', trnNoiseOut,
                         tstNoiseOut, trnLossValue, tstLossValue)
        saveToPickleFile(outputfilename + '1.pckl', trnNoiseOut1,
                         tstNoiseOut, trnLossValue, tstLossValue)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    print('Dim Reduction Apply autoencoder completed --{}'.format(datetime.now()-start))

def Init_Save_DimReductionAutoEncoder_Models(tFarray, trnpath, trnfilename, tstpath, tstfilename, outputpath, mymodelspath, myparams):

    start = datetime.now()
    for typeFeat in tFarray:
        if typeFeat:
            datafilename = outputpath + myparams[typeFeat]['finalfileNR']
            trn_data, _, trnLossValue, _ = readFromPickleFile(
                datafilename + '.pckl')
            trn_data = np.clip(np.divide(trn_data, 255), 0, 1)
            print(trn_data.shape)
            print('{}-{}'.format(myparams[typeFeat]
                  ['width'], myparams[typeFeat]['height']))
            if not myparams[typeFeat]['onlyApply'] and not myparams[typeFeat]['unZipFolder']:
                for ii in range(myparams[typeFeat]['learnRate'].shape[0]):
                    inputmodelfiledir = mymodelspath + \
                        myparams[typeFeat]['targetprefixes'][ii][0]
                    outputmodelfiledir = outputpath + \
                        myparams[typeFeat]['targetprefixes'][ii][0]
                    print(inputmodelfiledir)
                    print(outputmodelfiledir)
                    saveDimReductionAutoEncoderModels(trn_data, outputmodelfiledir, myparams[typeFeat]['learnRate'][ii], myparams[typeFeat]['l2reg'][ii],
                                                      myparams[typeFeat]['num_epochs'], myparams[typeFeat][
                        'batch_size'], myparams[typeFeat]['ckptlist']
                        )

            _, tst_data, _, _ = readFromPickleFile(datafilename + '.pckl')
            tst_data = np.clip(np.divide(tst_data, 255), 0, 1)
            print(tst_data.shape)
            for ii in range(myparams[typeFeat]['learnRate'].shape[0]):
                inputmodelfiledir = mymodelspath + \
                    myparams[typeFeat]['targetprefixes'][ii][0]
                outputmodelfiledir = outputpath + \
                    myparams[typeFeat]['targetprefixes'][ii][0]
                outputfilename = outputpath + \
                    myparams[typeFeat]['targetprefixes'][ii][1]
                print(outputfilename)
                if not myparams[typeFeat]['onlyApply'] and myparams[typeFeat]['unZipFolder']:
                    unZipfiles([inputmodelfiledir + 'model'],
                               [outputmodelfiledir + 'model'])

                applyDimReductionAutoEncoderModels(trn_data, tst_data, outputfilename, outputmodelfiledir,
                                                   myparams[typeFeat]['learnRate'][ii], myparams[typeFeat]['l2reg'][ii])

                if not myparams[typeFeat]['onlyApply'] and myparams[typeFeat]['unZipFolder']:
                    delfiles(outputmodelfiledir + 'model')

                saveImgs2(typeFeat, trnpath, trnfilename, outputfilename)

        print('Dim Reduction autoencoder Models completed --{}'.format(datetime.now()-start))
        