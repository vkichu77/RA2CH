
import numpy as np
import math
import pandas as pd
import os
import pickle
from datetime import datetime
import sys
import random
import tarfile
import shutil

def getDataFromCSV(path, filename, typeFeat):
    df = pd.read_csv(path + filename)
    # print(df)
    colRH = ['RH_mcp_E__ip', 'RH_pip_E__2', 'RH_pip_E__3', 'RH_pip_E__4', 'RH_pip_E__5', 'RH_mcp_E__1', 'RH_mcp_E__2', 'RH_mcp_E__3', 'RH_mcp_E__4', 'RH_mcp_E__5', 'RH_wrist_E__mc1', 'RH_wrist_E__mul', 'RH_wrist_E__nav', 'RH_wrist_E__lunate', 'RH_wrist_E__radius', 'RH_wrist_E__ulna',
             'RH_pip_J__2', 'RH_pip_J__3', 'RH_pip_J__4', 'RH_pip_J__5', 'RH_mcp_J__1', 'RH_mcp_J__2', 'RH_mcp_J__3', 'RH_mcp_J__4', 'RH_mcp_J__5', 'RH_wrist_J__cmc3', 'RH_wrist_J__cmc4', 'RH_wrist_J__cmc5', 'RH_wrist_J__mna', 'RH_wrist_J__capnlun', 'RH_wrist_J__radcar']
    colLH = ['LH_mcp_E__ip', 'LH_pip_E__2', 'LH_pip_E__3', 'LH_pip_E__4', 'LH_pip_E__5', 'LH_mcp_E__1', 'LH_mcp_E__2', 'LH_mcp_E__3', 'LH_mcp_E__4', 'LH_mcp_E__5', 'LH_wrist_E__mc1', 'LH_wrist_E__mul', 'LH_wrist_E__nav', 'LH_wrist_E__lunate', 'LH_wrist_E__radius', 'LH_wrist_E__ulna',
             'LH_pip_J__2', 'LH_pip_J__3', 'LH_pip_J__4', 'LH_pip_J__5', 'LH_mcp_J__1', 'LH_mcp_J__2', 'LH_mcp_J__3', 'LH_mcp_J__4', 'LH_mcp_J__5', 'LH_wrist_J__cmc3', 'LH_wrist_J__cmc4', 'LH_wrist_J__cmc5', 'LH_wrist_J__mna', 'LH_wrist_J__capnlun', 'LH_wrist_J__radcar']
    colLF = ['LF_mtp_E__ip', 'LF_mtp_E__1', 'LF_mtp_E__2', 'LF_mtp_E__3', 'LF_mtp_E__4', 'LF_mtp_E__5',
             'LF_mtp_J__ip', 'LF_mtp_J__1', 'LF_mtp_J__2', 'LF_mtp_J__3', 'LF_mtp_J__4', 'LF_mtp_J__5']
    colRF = ['RF_mtp_E__ip', 'RF_mtp_E__1', 'RF_mtp_E__2', 'RF_mtp_E__3', 'RF_mtp_E__4', 'RF_mtp_E__5',
             'RF_mtp_J__ip', 'RF_mtp_J__1', 'RF_mtp_J__2', 'RF_mtp_J__3', 'RF_mtp_J__4', 'RF_mtp_J__5']
    colOAnames = ['Overall_Tol', 'Overall_erosion', 'Overall_narrowing']
    if typeFeat in 'ALL':
        patIDs = []
    else:
        patIDs = [path + s + '-' + typeFeat + '.jpg' for s in df['Patient_ID']]
    if typeFeat in 'RF':
        colscores = df[colRF].to_numpy()
        colnames = colRF
    elif typeFeat in 'RH':
        colscores = df[colRH].to_numpy()
        colnames = colRH
    elif typeFeat in 'LH':
        colscores = df[colLH].to_numpy()
        colnames = colLH
    elif typeFeat in 'LF':
        colscores = df[colLF].to_numpy()
        colnames = colLF
    elif typeFeat in 'ALL':
        colscores = df[colOAnames].to_numpy()
        colnames = colOAnames
    print('{} {} {} {} {} {}'.format(len(patIDs), len(colRH), len(colLH), len(
        colLF), len(colRF), len(colRH)+len(colLH)+len(colLF)+len(colRF)))
    return patIDs, colscores, colnames


def getImageData(patIDs, width, height):
    imgdims = (width, height)
    idx = 0
    imgdata = np.zeros((len(patIDs), imgdims[1], imgdims[0]), np.float32)
    for imagepath in patIDs:
        imgdata[idx, :, :] = cv2.resize(cv2.imread(str(imagepath), 0), imgdims)
        idx = idx + 1

    imgdata = np.divide(imgdata, 255)
    return imgdata


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def saveToPickleFile(outputfilename, trn_data, tst_data, trn_loss, tst_loss):
    print('saving to ' + outputfilename)
    with open(outputfilename, 'wb') as f:
        pickle.dump([trn_data, tst_data, trn_loss, tst_loss],
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    print(outputfilename + ' saved')


def readFromPickleFile(readfilename):
    with open(readfilename, 'rb') as f:
        trn_data, tst_data, trn_loss, tst_loss = pickle.load(f)
    return trn_data, tst_data, trn_loss, tst_loss

def get_key(my_dict, val):
    for value, key in my_dict.items():
        if val == value:
            return key
    return "key doesn't exist"


def getInputData(typeFeat, tfilepath, tfilename, width, height):
    X_data, X_scores, X_colnames = [], [], []
    X_patIDs, X_scores, X_colnames = getDataFromCSV(
        tfilepath, tfilename, typeFeat)
    X_data = getImageData(X_patIDs, width, height)
    print("{}-{}".format(typeFeat, X_data.shape))
    return X_data, X_scores, X_colnames

def getPredData(typeFeat, tstpath, tstfilename, finaloutfilename):
    _, _, colnames = getDataFromCSV(tstpath, tstfilename, typeFeat)
    with open(finaloutfilename, 'rb') as f:  # Python 3: open(..., 'wb')
        final_pred = pickle.load(f)
    return final_pred, colnames


def setPredData(df, colnames, final_pred):
    # print(colnames)
    # print(final_pred.shape)
    for ii in range(0, final_pred.shape[0]):
        df.loc[ii, colnames] = final_pred[ii, :]
    return df


def scaleFinalData(typeFeat, trnpath, trnfilename, final_pred):
    # _, trnscores, _ = getDataFromCSV(trnpath, trnfilename, typeFeat)
    # preprocessor = prep.StandardScaler().fit(trnscores)
    # final_pred = preprocessor.inverse_transform(final_pred)
    final_pred[np.nonzero(final_pred < 0.5)] = 0
    final_pred = np.around(final_pred, decimals=0)
    final_pred = final_pred.astype(int)
    return final_pred

def createPredictionsCSV(trnpath, trnfilename, tstpath, tstfilename, outputpath, outputfilename):

    finaloutfilename = outputpath + 'result-184'
    finaloutfilenameRH = "{}-RH-1e-15.pckl".format(finaloutfilename)
    finaloutfilenameRF = "{}-RF-1e-15.pckl".format(finaloutfilename)
    finaloutfilenameLH = "{}-LH-1e-15.pckl".format(finaloutfilename)
    finaloutfilenameLF = "{}-LF-1e-15.pckl".format(finaloutfilename)

    final_predRH, colnamesRH = getPredData(
        'RH', tstpath, tstfilename, finaloutfilenameRH)
    final_predRF, colnamesRF = getPredData(
        'RF', tstpath, tstfilename, finaloutfilenameRF)
    final_predLH, colnamesLH = getPredData(
        'LH', tstpath, tstfilename, finaloutfilenameLH)
    final_predLF, colnamesLF = getPredData(
        'LF', tstpath, tstfilename, finaloutfilenameLF)

    # final_predRH = scaleFinalData('RH', trnpath, trnfilename, final_predRH )
    # final_predRF = scaleFinalData('RF', trnpath, trnfilename, final_predRF )
    # final_predLH = scaleFinalData('LH', trnpath, trnfilename, final_predLH )
    # final_predLF = scaleFinalData('LF', trnpath, trnfilename, final_predLF )

    _, _, colnamesOA = getDataFromCSV(trnpath, trnfilename, 'ALL')

    df = pd.read_csv(tstpath + tstfilename)
    df = setPredData(df, colnamesRH, final_predRH)
    df = setPredData(df, colnamesRF, final_predRF)
    df = setPredData(df, colnamesLH, final_predLH)
    df = setPredData(df, colnamesLF, final_predLF)

    tmp = np.zeros((final_predLF.shape[0], 3))
    tmp[:, 1] = np.sum(df.values[:, 4:47], axis=1)
    tmp[:, 2] = np.sum(df.values[:, 48:], axis=1)
    tmp[:, 0] = tmp[:, 1]+tmp[:, 2]
    tmp = tmp.astype(int)
    for ii in range(0, final_predLF.shape[0]):
        df.loc[ii, colnamesOA] = tmp[ii, :]

    tmp = list(df.columns)
    df[tmp[1:]] = df[tmp[1:]].round(0).astype(int)
    df.to_csv(outputpath + outputfilename, index=False)
    print(outputpath + outputfilename + ' saved')
    print(df)


def createPredictionsCSVTrain(trnpath, trnfilename, tstpath, tstfilename, outputpath, outputfilename):

    finaloutfilename = outputpath + 'result-184'
    finaloutfilenameRH = "{}-RH-1e-15trn.pckl".format(finaloutfilename)
    finaloutfilenameRF = "{}-RF-1e-15trn.pckl".format(finaloutfilename)
    finaloutfilenameLH = "{}-LH-1e-15trn.pckl".format(finaloutfilename)
    finaloutfilenameLF = "{}-LF-1e-15trn.pckl".format(finaloutfilename)

    final_predRH, colnamesRH = getPredData(
        'RH', tstpath, tstfilename, finaloutfilenameRH)
    final_predRF, colnamesRF = getPredData(
        'RF', tstpath, tstfilename, finaloutfilenameRF)
    final_predLH, colnamesLH = getPredData(
        'LH', tstpath, tstfilename, finaloutfilenameLH)
    final_predLF, colnamesLF = getPredData(
        'LF', tstpath, tstfilename, finaloutfilenameLF)

    # final_predRH = scaleFinalData('RH', trnpath, trnfilename, final_predRH )
    # final_predRF = scaleFinalData('RF', trnpath, trnfilename, final_predRF )
    # final_predLH = scaleFinalData('LH', trnpath, trnfilename, final_predLH )
    # final_predLF = scaleFinalData('LF', trnpath, trnfilename, final_predLF )

    _, _, colnamesOA = getDataFromCSV(trnpath, trnfilename, 'ALL')

    df = pd.read_csv(trnpath + trnfilename)
    df = setPredData(df, colnamesRH, final_predRH)
    df = setPredData(df, colnamesRF, final_predRF)
    df = setPredData(df, colnamesLH, final_predLH)
    df = setPredData(df, colnamesLF, final_predLF)

    tmp = np.zeros((final_predLF.shape[0], 3))
    tmp[:, 1] = np.sum(df.values[:, 4:47], axis=1)
    tmp[:, 2] = np.sum(df.values[:, 48:], axis=1)
    tmp[:, 0] = tmp[:, 1]+tmp[:, 2]
    tmp = tmp.astype(int)
    for ii in range(0, final_predLF.shape[0]):
        df.loc[ii, colnamesOA] = tmp[ii, :]

    tmp = list(df.columns)
    df[tmp[1:]] = df[tmp[1:]].round(0).astype(int)
    df.to_csv(outputpath + 'trn' + outputfilename, index=False)
    print(outputpath + 'trn' + outputfilename + ' saved')
    print(df)


def clearfiles(filenames):
    try:
        for fn in filenames:
            # print("Deleting {}".format(fn))
            if os.path.exists(fn):
                os.remove(fn)
    except:
        print('Error while deleting file ' + fn)


def delfiles(model_dir):
    try:
        print('deleting - {}'.format(model_dir))
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
    except:
        print('Error while deleting directory ' + model_dir)
    # import glob
    # files = glob.glob(model_dir + "/*")
    # for f in files:
    #     os.remove(f)


def deltfeventoutfiles(from_dir):
    for fname in os.listdir(from_dir):
        if fname.startswith("events.out.tfevent"):
            try:
                os.remove(os.path.join(from_dir, fname))
            except Exception as e:
                print(e)


def create7zFiles(targetzips, outputpath):
    import subprocess
    for tfn in targetzips:
        print('creating ' + outputpath + tfn + '.7z')
        inpfile = '7z a -t7z -m0=lzma -mx=9 -mfb=64 -md=32m -ms=on "{}{}.7z" "{}{}"'.format(
            outputpath, tfn, outputpath, tfn)
        byte_result = subprocess.check_output(inpfile, shell=True)
        str_result = byte_result.decode('utf-8')
        line_result = [l.split()[-1]
                       for l in str_result.rsplit("\n\n", 1)[-1].splitlines()[2:-2]]
        print(line_result)
        print('finished ' + outputpath + tfn + '.7z')


def createZipFiles(targetzips, outputpath):
    for tfn in targetzips:
        print(tfn)
        if os.path.exists(outputpath + tfn) and not os.path.exists(outputpath + tfn + '.tar.gz'):
            print('creating ' + outputpath + tfn + '.tar.gz')
            shutil.make_archive(outputpath + tfn, 'gztar',
                                outputpath + tfn + '/')
            print('finished ' + outputpath + tfn + '.tar.gz')


def unZipfiles(inputtars, targetzips):
    try:
        for ii in range(len(inputtars)):
            if os.path.exists(inputtars[ii] + '.tar.gz'):
                print('extracting ' + inputtars[ii] + '.tar.gz')
                shutil.unpack_archive(
                    inputtars[ii] + '.tar.gz', targetzips[ii])
                print('finished ' + inputtars[ii] +
                      '.tar.gz to ' + targetzips[ii])
            else:
                print(inputtars[ii] + '.tar.gz not found : fn unZipfiles')
    except Exception as e:
        print(e.message)


def checkfor(args, rv=0):
    """Make sure that a program necessary for using this script is
    available.

    Arguments:
    args  -- string or list of strings of commands. A single string may
             not contain spaces.
    rv    -- expected return value from evoking the command.
    """
    if isinstance(args, str):
        if ' ' in args:
            raise ValueError('no spaces in single command allowed')
        args = [args]
    try:
        with open(os.devnull, 'w') as bb:
            rc = subprocess.call(args, stdout=bb, stderr=bb)
        if rc != rv:
            raise OSError
    except OSError as oops:
        outs = "Required program '{}' not found: {}."
        print(outs.format(args[0], oops.strerror))
        sys.exit(1)


def copymodelfiles(fromDirectory, toDirectory, stepsno):
    print('copying ' + fromDirectory + ' to ' + toDirectory)
    if not os.path.exists(toDirectory):
        os.makedirs(toDirectory)
    nocopyexception = False
    try:
        fn = 'checkpoint'
        fn1 = os.path.join(fromDirectory, fn)
        fn2 = os.path.join(toDirectory, fn)
        shutil.copyfile(fn1, fn2)
        print("1. File copied " + fn2)

        fn = 'graph.pbtxt'
        fn1 = os.path.join(fromDirectory, fn)
        fn2 = os.path.join(toDirectory, fn)
        shutil.copyfile(fn1, fn2)
        print("2. File copied " + fn2)

        fn = 'model.ckpt-{}.data-00000-of-00001'.format(stepsno)
        fn1 = os.path.join(fromDirectory, fn)
        fn2 = os.path.join(toDirectory, fn)
        shutil.copyfile(fn1, fn2)
        print("3. File copied " + fn2)

        fn = 'model.ckpt-{}.index'.format(stepsno)
        fn1 = os.path.join(fromDirectory, fn)
        fn2 = os.path.join(toDirectory, fn)
        shutil.copyfile(fn1, fn2)
        print("4. File copied " + fn2)

        fn = 'model.ckpt-{}.meta'.format(stepsno)
        fn1 = os.path.join(fromDirectory, fn)
        fn2 = os.path.join(toDirectory, fn)
        shutil.copyfile(fn1, fn2)
        print("5. File copied " + fn2)

    except shutil.SameFileError:
        print("Source and destination represents the same file." + fn1 + ' ' + fn2)
        nocopyexception = True

    # If destination is a directory.
    except IsADirectoryError:
        print("Destination is a directory." + fn1 + ' ' + fn2)
        nocopyexception = True

    # If there is any permission issue
    except PermissionError:
        print("Permission denied." + fn1 + ' ' + fn2)
        nocopyexception = True

    # For other errors
    except:
        print("Error occurred while copying file." + fn1 + ' ' + fn2)
        nocopyexception = True

    return nocopyexception
