import tensorflow as tf
import sys
from datetime import datetime
from modules.applyThreshholds import doThreshholdToBoneFiles
from modules.noiseremoverfunctions import Init_Save_NoiseRemoverAutoEncoder_Models
from modules.dimreducerfunctions import Init_Save_DimReductionAutoEncoder_Models
from modules.dnnregressorfunctions import callRegression


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print(sys.version)
    tf.get_logger().setLevel('WARNING')
    tf.autograph.set_verbosity(2)

    print(tf.__version__)
    print('**'*25)
    # !pip install -U -q PyDrive

    trnpath = '/train/'
    trnfilename = 'training.csv'
    tstpath = '/test/'
    tstfilename = 'template.csv'
    outputpath = '/output/'
    outputfilename = 'predictions.csv'
    mymodelspath = '/mymodels/'


    print('paths - {} : {} : {} : {}'.format(trnpath, tstpath, outputpath, mymodelspath))
    files = []
    print('checking tarballs in ' + mymodelspath)
    tFarray = ['RH', 'RF', 'LH', 'LF']

    learnRateNR = np.array([0.001, 0.001, 0.001, 0.001, 0.001])
    l2regNR = np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
    batch_sizeNR = 184
    learnRateDR = np.array([0.0001])
    l2regDR = np.array([1e-05])
    batch_sizeDR = 184
    width, height = 128, 128

    start = datetime.now()
    dt_string = start.strftime("%d/%m/%Y %H:%M:%S")
    print("Calculation date and time =", dt_string)
    NRflag, boneApproxFlag2, DRflag = True, True, True

    learnRateNR = np.array([0.001, 0.001, 0.001, 0.001, 0.001])
    l2regNR = np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
    batch_sizeNR = 184
    num_epochsNR = np.array([500, 500, 500, 500])
    ckptlistNR = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    myparamsNR = {}
    for idx, typeFeat in enumerate(tFarray):
        if typeFeat:
            targetprefixes = [['trnmodelNR-{}-{}-{}-{}'.format(learnRateNR[ii], l2regNR[ii], batch_sizeNR, typeFeat), 'trndataNR-{}-{}-{}-{}'.format(learnRateNR[ii], l2regNR[ii], batch_sizeNR, typeFeat)]
                              for ii in range(learnRateNR.shape[0])]

            myparamsNR[typeFeat] = {'learnRate': learnRateNR, 'l2reg': l2regNR, 'batch_size': batch_sizeNR, 'num_epochs': num_epochsNR[idx],
                                    'ckptlist': ckptlistNR, 'targetprefixes': targetprefixes, 'width': width, 'height': height,
                                    'unZipFolder': False, 'onlyApply': True
                                    }
    if NRflag:
        Init_Save_NoiseRemoverAutoEncoder_Models(
            tFarray, trnpath, trnfilename, tstpath, tstfilename, outputpath, mymodelspath, myparamsNR, my_drive)

    finaloutfilenameNR = 'trndata2-NR'
    if boneApproxFlag2:
        doThreshholdToBoneFiles(tFarray, trnpath, trnfilename, tstpath,
                                    tstfilename, outputpath, finaloutfilenameNR, myparamsNR)

    learnRateDR = np.array([0.0001])
    l2regDR = np.array([1e-05])
    batch_sizeDR = 184
    num_epochsDR = np.array([4000, 4000, 4000, 4000])
    num_epochsDR = np.array([1000, 1000, 1000, 1000])
    ckptlistDR = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    # num_epochsDR = np.array( [500, 100, 100, 100] )
    # ckptlistDR = [100, 500, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    myparamsDR = {}
    # finaloutfilename + fileprefix + typeFeat +
    # fileprefix = '-hist-eq-'
    for idx, typeFeat in enumerate(tFarray):
        if typeFeat:
            targetprefixes = [['trnmodelDR-{}-{}-{}-{}'.format(learnRateDR[ii], l2regDR[ii], batch_sizeDR, typeFeat), 'trndataDR-{}-{}-{}-{}'.format(learnRateDR[ii], l2regDR[ii], batch_sizeDR, typeFeat)]
                              for ii in range(learnRateDR.shape[0])]

            myparamsDR[typeFeat] = {'learnRate': learnRateDR, 'l2reg': l2regDR, 'batch_size': batch_sizeDR, 'num_epochs': num_epochsDR[idx],
                                    'ckptlist': ckptlistDR, 'targetprefixes': targetprefixes, 'width': width, 'height': height,
                                    'unZipFolder': False, 'onlyApply': False, 'finalfileNR': finaloutfilenameNR + '-' + typeFeat + '-hist-eq'
                                    }
    if DRflag:
        Init_Save_DimReductionAutoEncoder_Models(
            tFarray, trnpath, trnfilename, tstpath, tstfilename, outputpath, mymodelspath, myparamsDR, my_drive)

    finaloutfilenameDR = 'trndataDR-{}-{}-{}'.format(
        learnRateDR[0], l2regDR[0], batch_sizeDR)
    targetprefixes = [finaloutfilenameDR,
                      'result-{}'.format(batch_sizeDR), 'dnnreg-{}'.format(batch_sizeDR)]

    idx = 0
    steps_per_epochdnn = np.array([-1, -1, -1, -1])
    batch_sizednn = np.array([500, 500, 500, 500, 500])
    num_epochsdnn = np.array([2000, 2000, 2000, 2000])
    hidden_unitsdnn = np.array([[1024, 1024, 1024, 1024, 1024],
                                [1024, 1024, 1024, 1024, 1024],
                                [1024, 1024, 1024, 1024, 1024],
                                [1024, 1024, 1024, 1024, 1024]])
    save_checkpoints_stepsdnn = np.array([1000, 1000, 1000, 1000])

    learning_ratednn = np.array([0.0, 0.0, 0.0, 0.0])
    # ROTATE_ANGLEdnn = np.array( [45, 45, 45, 45] )
    ROTATE_ANGLEdnn = np.array([135.0, 135.0, 135.0, 135.0])
    SHIFT_VALUEdnn = np.array([5.0, 5.0, 5.0, 5.0])
    momentumdnn = np.array([0.9, 0.9, 0.9, 0.9])
    dropoutdnn = np.array([0.0, 0.0, 0.0, 0.0])


    # save_checkpoints_stepsdnn = np.array( [100, 100, 100, 100] )
    # num_epochsdnn = np.array([50, 50, 50, 50])

    myparamsDNN = {}
    for idx, typeFeat in enumerate(tFarray):
        if typeFeat:
            myparamsDNN[typeFeat] = {'num_epochs': num_epochsdnn[idx], 'batch_size': batch_sizednn[idx], 'steps_per_epoch': steps_per_epochdnn[idx], 'hidden_units': hidden_unitsdnn[idx, :],
                                     'learning_rate': learning_ratednn[idx], 'save_checkpoints_steps': save_checkpoints_stepsdnn[idx], 'ROTATE_ANGLE': ROTATE_ANGLEdnn[idx],
                                     'SHIFT_VALUE': SHIFT_VALUEdnn[idx], 'momentum': momentumdnn[idx], 'dropout': dropoutdnn[idx],
                                     'freq_loss_print': 10, 'freq_drv_trash_clear': 400, 'temp_model_dir': outputpath + 'TMP-{}'}

    learning_rate = [1e-1, 1e-3, 1e-7, 1e-11, 1e-15,
                     1e-19, 1e-23, 1e-27, 1e-31, 1e-35, 1e-39]
    idxlrnrate = 7

    rndnumb = [12345, 67890, 54321, 99876]
    idxrnd = 0
    print('rndnumb-{}'.format(rndnumb[idxrnd]))
    if idxlrnrate == 0 and idxrnd == 0:
        loadModelFlag, unZipFolder, onlyApply = False, False, False
    else:
        loadModelFlag, unZipFolder, onlyApply = True, False, False

    [myparamsDNN[typeFeat].update({'learning_rate': learning_rate[idxlrnrate]})
     for typeFeat in tFarray if typeFeat]
    callRegression(tFarray, targetprefixes, trnpath, trnfilename, tstpath, tstfilename, outputpath, mymodelspath, myparamsDNN,
           my_drive, rndnumb=rndnumb[idxrnd], loadModelFlag=loadModelFlag, unZipFolder=unZipFolder, onlyApply=onlyApply)

    # createPredictionsCSV(trnpath, trnfilename, tstpath, tstfilename, outputpath, outputfilename)
    # createPredictionsCSVTrain(trnpath, trnfilename, tstpath, tstfilename, outputpath, outputfilename)
    edtime = datetime.now()-start
    print("Completed Time - Full ", edtime)

    # trndata-0.0001-1e-05-184DRLF


if __name__ == "__main__":
    main()