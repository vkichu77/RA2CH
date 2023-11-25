import tensorflow as tf
import tensorflow.keras.layers as layers
from modelclasses import ConvNetAutoEncoderNoiseRemover
from datetime import datetime

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        encoded, reconstruction = model(inputs)
        loss_value = loss(targets, reconstruction)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), reconstruction, encoded

def loss(x, x_bar):
    return tf.reduce_sum(tf.keras.losses.MSE(x, x_bar))

def trainNoiseRemoverAutoEncoderModels(trn_data, outputmodelfiledir, learning_rate, l2_reg, num_epochs, batch_size, ckptlist):
    random.seed(555)
    np.random.seed(seed=1111)
    tf.random.set_seed(seed=2345)

    start = datetime.now()
    if not os.path.exists(outputmodelfiledir):
        os.makedirs(outputmodelfiledir)

    print('Save NR Autoencoder started LR-{}, l2-{}, num_epochs-{}, batch_size-{}'.format(
        learning_rate, l2_reg, num_epochs, batch_size))
    tf.keras.backend.set_floatx('float32')
    g = tf.Graph()
    with g.container('experiment0'):
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        l2_regularizer = tf.keras.regularizers.l2(l2_reg)
        ConvNetNR = ConvNetAutoEncoderNoiseRemover(
            l2_regularizer=l2_regularizer, name='3_layer_NR')
        ckpt = tf.train.Checkpoint(step=tf.Variable(
            1), optimizer=optimizer, net=ConvNetNR)
        manager = tf.train.CheckpointManager(
            ckpt, outputmodelfiledir, max_to_keep=1)

        ckpt.restore(manager.latest_checkpoint).expect_partial()
        dfval = None
        if manager.latest_checkpoint:
            print("Save NR Restored from {}".format(manager.latest_checkpoint))
            ckpt.step.assign_add(1)
            dfval = pd.read_csv(outputmodelfiledir + 'model.csv')
        else:
            print("Save NR Initializing from scratch.")

        dfvalues = np.zeros((num_epochs, 3))
        global_step = tf.Variable(0)
        for epoch in range(1, num_epochs+1):
            trnrecon = tf.Variable(0., dtype=tf.float32)
            for x in range(0, trn_data.shape[0], batch_size):
                if x + batch_size > trn_data.shape[0]:
                    x_inp = trn_data[x: x +
                                     trn_data.shape[0] - batch_size, :, :, :]
                else:
                    x_inp = trn_data[x: x + batch_size, :, :, :]
                loss_value, grads, _, _ = grad(ConvNetNR, x_inp, x_inp)
                optimizer.apply_gradients(
                    zip(grads, ConvNetNR.trainable_variables), global_step)
                trnrecon.assign(tf.math.add(trnrecon, loss_value))
            avgLoss = trnrecon.numpy() / np.ceil(trn_data.shape[0]/batch_size)
            if int(ckpt.step) in ckptlist:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(
                    int(ckpt.step), save_path))
                print("loss {:1.2f}".format(avgLoss))
                deltfeventoutfiles(outputmodelfiledir)
                
            if int(ckpt.step) % 100 == 0:
                print("Epoch: {:5g} Loss: {:.2f} : {:5g}".format(
                    epoch, avgLoss, int(ckpt.step)))
            dfvalues[epoch-1, :] = [epoch, avgLoss, int(ckpt.step)]
            ckpt.step.assign_add(1)

        tf.saved_model.save(ConvNetNR, outputmodelfiledir + 'model')
        dataset = pd.DataFrame(
            {'epoch': dfvalues[:, 0], 'Loss': dfvalues[:, 1], 'cktp': dfvalues[:, 2]})
        if dfval is not None:
            dataset = dfval.append(dataset, ignore_index=True)
        dataset.to_csv(outputmodelfiledir + 'model.csv', index=False)
        print('model saved')

    deltfeventoutfiles(outputmodelfiledir)
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    print('autoencoder completed -{}--{}'.format(outputmodelfiledir, datetime.now()-start))


def applyNoiseRemoverAutoEncoderModels(trn_data, tst_data, outputfilename, outputmodelfiledir, learning_rate, l2_reg):

    random.seed(555)
    np.random.seed(seed=1111)
    tf.random.set_seed(seed=2345)

    start = datetime.now()
    batch_size = 1
    width, height = tf.shape(trn_data)[2], tf.shape(trn_data)[1]

    print('Apply NR Autoencoder started, LR-{}, l2-{}, width-{}, height-{}'.format(
        learning_rate, l2_reg, width, height))
    tf.keras.backend.set_floatx('float32')
    g = tf.Graph()
    with g.container('experiment0'):
        ConvNetNR = tf.saved_model.load(outputmodelfiledir + 'model')
        trnNoiseOut = np.zeros(
            (trn_data.shape[0], height, width, 1), dtype=np.float32)
        trnLossValue = 0.0
        for x in range(0, trn_data.shape[0]):
            x_inp = tf.reshape(trn_data[x, :, :], shape=(1, height, width, 1))
            x_inp = tf.convert_to_tensor(x_inp, dtype=tf.float32, name='x')
            [], tmp = ConvNetNR(x_inp)
            loss_value = loss(x_inp, tmp)
            trnNoiseOut[x, :, :, :] = tmp.numpy()
            trnLossValue = trnLossValue + loss_value.numpy()
        trnLossValue = trnLossValue/trn_data.shape[0]

        tstNoiseOut = np.zeros(
            (tst_data.shape[0], height, width, 1), dtype=np.float32)
        tstLossValue = 0.0
        for x in range(0, tst_data.shape[0]):
            x_inp = tf.reshape(tst_data[x, :, :], shape=(1, height, width, 1))
            x_inp = tf.convert_to_tensor(x_inp, dtype=tf.float32, name='x')
            [], tmp = ConvNetNR(x_inp)
            loss_value = loss(x_inp, tmp)
            tstNoiseOut[x, :, :, :] = tmp.numpy()
            tstLossValue = tstLossValue + loss_value.numpy()
        tstLossValue = tstLossValue/tst_data.shape[0]

        print("loss1-{:1.4f} loss2-{:1.4f}".format(trnLossValue, tstLossValue))
        saveToPickleFile(outputfilename + '.pckl', trnNoiseOut,
                         tstNoiseOut, trnLossValue, tstLossValue)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    print('autoencoder completed -{}--{}'.format(outputfilename, datetime.now()-start))



def Init_Save_NoiseRemoverAutoEncoder_Models(tFarray, trnpath, trnfilename, tstpath, tstfilename, outputpath, mymodelspath, myparams):

    start = datetime.now()
    for typeFeat in tFarray:
        if typeFeat:
            trn_data, _, _ = getInputData(
                typeFeat, trnpath, trnfilename, myparams[typeFeat]['width'], myparams[typeFeat]['height'])
            trn_data = tf.reshape(trn_data, [
                                  trn_data.shape[0], myparams[typeFeat]['height'], myparams[typeFeat]['width'], 1])
            print(tf.shape(trn_data))
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
                    saveNoiseRemoverAutoEncoderModels(trn_data, outputmodelfiledir, myparams[typeFeat]['learnRate'][ii], myparams[typeFeat]['l2reg'][ii],
                                                      myparams[typeFeat]['num_epochs'], myparams[typeFeat][
                        'batch_size'], myparams[typeFeat]['ckptlist'])

            trn_data = tf.reshape(trn_data, [tf.shape(trn_data)[
                                  0], myparams[typeFeat]['height'], myparams[typeFeat]['width']])
            tst_data, _, _ = getInputData(
                typeFeat, tstpath, tstfilename, myparams[typeFeat]['width'], myparams[typeFeat]['height'])
            print(tf.shape(trn_data))
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

                applyNoiseRemoverAutoEncoderModels(trn_data, tst_data, outputfilename, outputmodelfiledir,
                                                   myparams[typeFeat]['learnRate'][ii], myparams[typeFeat]['l2reg'][ii])

                if not myparams[typeFeat]['onlyApply'] and myparams[typeFeat]['unZipFolder']:
                    delfiles(outputmodelfiledir + 'model')

                # saveImgs( typeFeat, trnpath, trnfilename, outputfilename )

    print('Noise Rem autoencoder Models completed - NR--{}'.format(datetime.now()-start))


