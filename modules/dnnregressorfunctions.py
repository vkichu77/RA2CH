import cv2
import tensorflow as tf
import os
import pickle
from datetime import datetime
import sys
import tensorflow.keras.layers as layers
from utils import getDataFromCSV, getInputData, saveToPickleFile, unZipfiles, delfiles, deltfeventoutfiles, copymodelfiles, clearfiles


def doRegression(typeFeat, trnpath, trnfilename, readfilename, finaloutfile, model_dir, csvfile, myparams, 
    rndnumb, loadModelFlag=False, applyOnly=False):

    random.seed(534*rndnumb)
    np.random.seed(seed=1345*rndnumb)
    tf.random.set_seed(seed=4975*rndnumb)

    start = datetime.now()

    BATCH_SIZE = myparams[typeFeat]['batch_size']
    hidden_units = myparams[typeFeat]['hidden_units']
    STEPS_PER_EPOCH = myparams[typeFeat]['steps_per_epoch']
    learning_rate = myparams[typeFeat]['learning_rate']
    save_checkpoints_steps = myparams[typeFeat]['save_checkpoints_steps']
    ROTATE_ANGLE = myparams[typeFeat]['ROTATE_ANGLE']
    SHIFT_VALUE = myparams[typeFeat]['SHIFT_VALUE']
    momentum = myparams[typeFeat]['momentum']
    dropout = myparams[typeFeat]['dropout']
    freq_loss_print = myparams[typeFeat]['freq_loss_print']
    freq_drv_trash_clear = myparams[typeFeat]['freq_drv_trash_clear']
    EPOCHS = myparams[typeFeat]['num_epochs']
    temp_model_dir = myparams[typeFeat]['temp_model_dir']

    X_train1, _, _, _ = readFromPickleFile(readfilename)
    _, trnscores, _ = getDataFromCSV(trnpath, trnfilename, typeFeat)

    height, width = X_train1.shape[1], X_train1.shape[2]

    trnscores = trnscores.astype(np.float32)
    X_train1 = np.reshape(X_train1, (X_train1.shape[0], height, width))
    X_train1 = np.trunc(np.clip((X_train1 * 255), 0, 255))

    if STEPS_PER_EPOCH < 0:
        STEPS_PER_EPOCH = int(np.ceil((X_train1.shape[0]*2)/BATCH_SIZE))

    print("""batch_size: {batch_size}, hidden_units: {hidden_units}, steps_per_epoch: {steps_per_epoch},
    learning_rate: {learning_rate}, save_checkpoints_steps:{save_checkpoints_steps}, ROTATE_ANGLE: {ROTATE_ANGLE},
    SHIFT_VALUE: {SHIFT_VALUE}, momentum: {momentum}, dropout: {dropout}, freq_loss_print: {freq_loss_print},
    freq_drv_trash_clear: {freq_drv_trash_clear}, num_epochs: {num_epochs}""".format(**myparams[typeFeat]))
    print('STEPS_PER_EPOCH: {}, train shape-{}'.format(STEPS_PER_EPOCH, X_train1.shape))

    tf.keras.backend.set_floatx('float32')
    g = tf.Graph()
    with g.container('experiment0'):

        def random_rotate_image(imagedata, rotate_angle):
            angle = np.random.uniform(rotate_angle * -1, rotate_angle)
            return cv2.resize(cv2.warpAffine(imagedata,
                                             cv2.getRotationMatrix2D(
                                                 tuple(np.array([imagedata.shape[0], imagedata.shape[1]]) / 2), angle, 1.0),
                                             (imagedata.shape[1], imagedata.shape[0])),
                              imagedata.shape)

        def random_shift_image(imagedata, shift_value):
            if random.uniform(0, 1) < 0.5:
                s1 = random.randrange(shift_value * -1, shift_value)
            else:
                s1 = 0
            if random.uniform(0, 1) < 0.5:
                s2 = random.randrange(shift_value * -1, shift_value)
            else:
                s2 = 0
            # print('{} {}'.format(s1,s2))
            return cv2.warpAffine(imagedata,
                                  np.float32([[1, 0, s1], [0, 1, s2]]),
                                  (imagedata.shape[1], imagedata.shape[0]))

        def training_input_fn_with_dataaug(features, labels, batch_size, epochs):
            data_aug_factor = 2
            tmp1 = np.zeros(
                (features.shape[0] * data_aug_factor, features.shape[1], features.shape[2]), dtype=np.float32)

            idx = 0
            for jj in range(data_aug_factor):
                for ii in range(0, features.shape[0]):
                    tmp1[idx, :, :] = np.cast['float32'](
                        random_rotate_image(features[ii, :, :], ROTATE_ANGLE))
                    tmp1[idx, :, :] = np.cast['float32'](
                        random_shift_image(tmp1[idx, :, :], SHIFT_VALUE))
                    idx = idx + 1

            tmp1 = np.divide(tmp1, 255)
            features = np.clip(tmp1, 0, 1)
            tmp1 = []
            tmp1scores = np.zeros(
                (labels.shape[0]*2, labels.shape[1]), dtype=np.float32)
            tmp1scores[0:labels.shape[0], 0:labels.shape[1]] = labels
            tmp1scores[labels.shape[0]:labels.shape[0]
                       * 2, 0:labels.shape[1]] = labels
            labels = tmp1scores
            tmp1scores = []
            return training_input_fn_without_dataaug(features, labels, batch_size, epochs)

        def training_input_fn_without_dataaug(features, labels, batch_size, epochs):
            features = np.reshape(
                features, (features.shape[0], features.shape[1]*features.shape[2]))
            buff_size = features.shape[0]*features.shape[1]
            features = {str(k): tf.constant(
                features[:, k]) for k in range(0, features.shape[1])}
            dataset = tf.data.Dataset.from_tensor_slices(
                (features, tf.constant(labels)))
            dataset = dataset.shuffle(buff_size).repeat(
                epochs).batch(batch_size)
            return (dataset)

        def training_input_fn2():
            features = np.clip(np.divide(X_train1, 255), 0, 1)
            features = np.reshape(
                features, (features.shape[0], features.shape[1]*features.shape[2]))
            features = {str(k): tf.constant(
                features[:, k]) for k in range(0, features.shape[1])}
            return tf.data.Dataset.from_tensors((features, tf.constant(trnscores)))

        def test_input_fn():
            features = np.clip(np.divide(X_test1, 255), 0, 1)
            features = np.reshape(
                features, (features.shape[0], features.shape[1]*features.shape[2]))
            features = {str(k): tf.constant(
                features[:, k]) for k in range(0, features.shape[1])}
            return tf.data.Dataset.from_tensors((features, tf.constant(trnscores)))

        feature_columns = [tf.feature_column.numeric_column(
            str(x)) for x in range(0, X_train1.shape[1])]

        test_config = tf.estimator.RunConfig(tf_random_seed=45*rndnumb, keep_checkpoint_max=1,
                                             save_checkpoints_steps=save_checkpoints_steps, save_checkpoints_secs=None)

        if applyOnly or loadModelFlag:
            try:
                print('loading from ' + model_dir)
                if os.path.exists(model_dir):
                    ws = tf.estimator.WarmStartSettings(
                        ckpt_to_initialize_from=model_dir)
                    model = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                                      label_dimension=trnscores.shape[1],
                                                      hidden_units=hidden_units,
                                                      warm_start_from=ws,
                                                      model_dir=model_dir,
                                                      config=test_config,
                                                      optimizer=lambda: tf.keras.optimizers.SGD(
                                                          learning_rate=learning_rate, momentum=momentum)
                                                      )
                print('Model successfully loaded {} checkpoint {}'.format(
                    model_dir, model.latest_checkpoint()))
                if os.path.exists(csvfile + '.csv'):
                    dfvaltmp = pd.read_csv(csvfile + '.csv')
                    startsidx = dfvaltmp.shape[0] * freq_loss_print
                    endsidx = startsidx + EPOCHS
                    lossper10epoch = []
                    lossper10epoch.append(dfvaltmp['my_average_loss'].mean())
                    print('mean: {}'.format(
                        dfvaltmp['my_average_loss'].mean()))
                    dfvaltmp = []
            except Exception as e:
                print(e)
                print('Error loading model ' + model_dir)
                return
        else:
            delfiles(model_dir+"/eval")
            delfiles(model_dir)
            clearfiles([csvfile + '.csv'])

            model = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                              hidden_units=hidden_units,
                                              label_dimension=trnscores.shape[1],
                                              model_dir=model_dir,
                                              dropout=dropout,
                                              config=test_config,
                                              optimizer=lambda: tf.keras.optimizers.SGD(
                                                  learning_rate=learning_rate, momentum=momentum)
                                              )
            startsidx = 0
            endsidx = EPOCHS
            lossper10epoch = []

        if not applyOnly:
            dfvalues = []
            print('EPOCH range:  {} --> {}'.format(startsidx+1, endsidx+1))
            for epoch in range(startsidx+1, endsidx+1):
                model.train(input_fn=lambda: training_input_fn_with_dataaug(
                    X_train1, trnscores, BATCH_SIZE, EPOCHS), steps=STEPS_PER_EPOCH)
                # print(model.get_variable_names())
                if epoch % freq_loss_print == 0:
                    preds = model.predict(input_fn=training_input_fn2)
                    predictions = list(preds)
                    final_pred = []
                    for pred in predictions:
                        final_pred.append(get_key(pred, 'predictions'))
                    final_pred = np.asarray(final_pred)
                    final_pred = np.squeeze(final_pred)
                    final_pred = scaleFinalData('', '', '', final_pred)
                    loss = np.square(np.subtract(trnscores, final_pred)).mean()
                    lossper10epoch.append(loss)
                    avgloss = sum(lossper10epoch) / len(lossper10epoch)
                    evalds = model.evaluate(input_fn=training_input_fn2)
                    print('Epoch %i: %f LOSS %f AverageLoss myloss %f myavgloss %f' % (
                        epoch, evalds["loss"], evalds["average_loss"], loss, avgloss))
                    dfvalues.append(
                        [epoch, evalds["loss"], evalds["average_loss"], loss, avgloss])
                    # print('Epoch %i: myloss %f myavgloss %f' % (epoch, loss, avgloss ))

                if (epoch * STEPS_PER_EPOCH) % save_checkpoints_steps == 0:
                    dataset = pd.DataFrame(dfvalues, columns=[
                                           'epoch', 'loss', 'average_loss', 'my_loss', 'my_average_loss'])
                    dataset['learning_rate'] = learning_rate
                    if os.path.exists(csvfile + '.csv'):
                        dfval = pd.read_csv(csvfile + '.csv')
                        dataset = dfval.append(dataset, ignore_index=True)
                    dataset.to_csv(csvfile + '.csv', index=False)
                    deltfeventoutfiles(model_dir)
                    delfiles(temp_model_dir.format(
                        dataset.shape[0] * freq_loss_print * STEPS_PER_EPOCH))
                    nocopyexception = copymodelfiles(model_dir, temp_model_dir.format(
                        dataset.shape[0] * freq_loss_print * STEPS_PER_EPOCH), dataset.shape[0] * freq_loss_print * STEPS_PER_EPOCH)
                    if nocopyexception:
                        delfiles(temp_model_dir.format(
                            dataset.shape[0] * freq_loss_print * STEPS_PER_EPOCH))
                        nocopyexception = copymodelfiles(model_dir, temp_model_dir.format(
                            dataset.shape[0] * freq_loss_print * STEPS_PER_EPOCH), dfvalues.shape[0] * freq_loss_print * STEPS_PER_EPOCH)
                        print('copied {}'.format(nocopyexception))
                    dfvalues, dfval, dataset = [], [], []

            dfval = pd.read_csv(csvfile + '.csv')
            copymodelfiles(model_dir, "{}-{}".format(csvfile, learning_rate),
                           dfval.shape[0] * freq_loss_print * STEPS_PER_EPOCH)

        def predictResults(px):
            predictions = list(preds)
            final_pred = []
            for pred in predictions:
                final_pred.append(get_key(pred, 'predictions'))
            final_pred = np.asarray(final_pred)
            final_pred = np.squeeze(final_pred)
            return final_pred

        preds = model.predict(input_fn=training_input_fn2)
        final_pred = predictResults(preds)
        with open(finaloutfile + 'trn.pckl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(final_pred, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(final_pred.mean(axis=0))
        final_pred = scaleFinalData('', '', '', final_pred)
        loss = np.square(np.subtract(trnscores, final_pred))
        print(loss.mean(axis=0))
        print(loss.mean())
        X_train1 = []

        _, X_test1, _, _ = readFromPickleFile(readfilename)
        X_test1 = np.reshape(X_test1,  (X_test1.shape[0], height, width))
        X_test1 = np.trunc(np.clip((X_test1 * 255), 0, 255))
        print(X_test1.shape)
        preds = model.predict(input_fn=test_input_fn)
        final_pred = predictResults(preds)

        with open(finaloutfile + '.pckl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(final_pred, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(final_pred.mean(axis=0))
        X_test1 = []

    delfiles(model_dir+"/eval")

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    print(datetime.now()-start)

    final_pred = []
    X_train1, X_train1, trnscores = [], [], []


def callRegression(tFarray, targetprefixes, trnpath, trnfilename, tstpath, tstfilename, outputpath, mymodelspath, myparams, rndnumb, loadModelFlag=False, unZipFolder=False, onlyApply=True):

    readfilename = outputpath + targetprefixes[0]
    finaloutfilename = outputpath + targetprefixes[1]
    outputmodel_dir = outputpath + targetprefixes[2]
    inputmodel_dir = mymodelspath + targetprefixes[2]
    exportmodel_dir = outputpath + targetprefixes[2] + 'model'

    for idx, typeFeat in enumerate(tFarray):
        if typeFeat:
            readfile = "{}-{}.pckl".format(readfilename, typeFeat)
            finaloutfile = "{}-{}-{}".format(finaloutfilename,
                                             typeFeat, myparams[typeFeat]['learning_rate'])

            if unZipFolder:
                unZipfiles([inputmodel_dir + '-' + typeFeat],
                           [outputmodel_dir + '-' + typeFeat])

            doRegression(typeFeat, trnpath, trnfilename, readfile, finaloutfile, outputmodel_dir + '-' + typeFeat,
                         exportmodel_dir + '-' + typeFeat, myparams, rndnumb, loadModelFlag, onlyApply)

            if unZipFolder:
                delfiles(outputmodel_dir + '-' + typeFeat)
                # clearfiles([outputmodelfiledir + '-' + typeFeat + '.tar.gz'])

