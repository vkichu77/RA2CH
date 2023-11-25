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
from utils import getDataFromCSV, getInputData, saveToPickleFile, unZipfiles, delfiles
from modelclasses import saveBoneApproximationToFiles

def doThreshholdToBoneFiles(self, tFarray, trnpath, trnfilename, tstpath, tstfilename, outputpath, 
                                    finaloutfilename, myparams):
    
    sBZ = saveBoneApproximationToFiles()
    start=datetime.now()          
    finaloutfilename = outputpath + finaloutfilename    
    print(finaloutfilename)
    for typeFeat in tFarray:
        if typeFeat:
            X_patIDs, _, _ = getDataFromCSV(trnpath, trnfilename, typeFeat)  
            for ii in range(len(X_patIDs)):
                _, X_patIDs[ii] = os.path.split( X_patIDs[ii] )

            trn_data, _, _ = getInputData( typeFeat, trnpath, trnfilename, myparams[typeFeat]['width'], myparams[typeFeat]['height'] )    
            tst_data, _, _ = getInputData(typeFeat, tstpath, tstfilename, myparams[typeFeat]['width'], myparams[typeFeat]['height'] )
            trn_data, tst_data = np.clip(trn_data * 255, 0, 255), np.clip(tst_data * 255, 0, 255)                      
            trn_data, tst_data = np.reshape( trn_data, (trn_data.shape[0], trn_data.shape[1], trn_data.shape[2], 1) ), \
                                 np.reshape( tst_data, (tst_data.shape[0], tst_data.shape[1], tst_data.shape[2], 1) )

            thresh1 = [20, 30, 40, 50, 60, 70, 80, 100, 110, 120, 130] #0, 7, 7, 11
            # thresh1 = [20, 30, 40, 50, 60, 70, 80, 110, 120, 130] #0, 7, 7, 10
            threshsize1 = 10                                   
            minthreshsize1 = 15
            minthreshsize2 = 10
            fileprefix = '-hist-eq'
            saveFlag, applyHist = True, True
            if applyHist:
                trn_data, tst_data = sBZ.applyMultiThreshold(X_patIDs, myparams, typeFeat, finaloutfilename, trn_data, tst_data, thresh1, threshsize1, minthreshsize1, minthreshsize2, None, saveFlag, applyHist)           
                saveToPickleFile(finaloutfilename + '-' + typeFeat + '-new.pckl', trn_data, tst_data, 0.0, 0.0 )
                # saveImgs4( typeFeat, trnpath, trnfilename, finaloutfilename + 'new' )
                trn_data, tst_data = np.trunc(np.clip(np.divide(trn_data, 255), 0, 1)), np.trunc(np.clip(np.divide(tst_data, 255), 0, 1))
                for ii in range( myparams[typeFeat]['learnRate'].shape[0] ):
                    inputmodelfiledir = mymodelspath + myparams[typeFeat]['targetprefixes'][ii][0]
                    outputmodelfiledir = outputpath + myparams[typeFeat]['targetprefixes'][ii][0]
                    outputfilename = outputpath + myparams[typeFeat]['targetprefixes'][ii][1] + fileprefix                   
                    print(outputfilename)
                    if not myparams[typeFeat]['onlyApply'] and myparams[typeFeat]['unZipFolder']:  
                        unZipfiles([inputmodelfiledir + 'model'], [outputmodelfiledir + 'model'])

                    applyNoiseRemoverAutoEncoderModels(trn_data, tst_data, outputfilename, outputmodelfiledir, 
                                                        myparams[typeFeat]['learnRate'][ii], myparams[typeFeat]['l2reg'][ii] )
                    
                    if not myparams[typeFeat]['onlyApply'] and myparams[typeFeat]['unZipFolder']:  
                        delfiles(outputmodelfiledir + 'model')
            
            trn_data, tst_data, _, _ = readFromPickleFile(finaloutfilename + '-' + typeFeat + '-new.pckl')    
            trn_data, tst_data = sBZ.applySingleThreshold(X_patIDs, myparams, typeFeat, finaloutfilename + fileprefix, trn_data, tst_data, fileprefix, True)                             
            saveToPickleFile(finaloutfilename + '-' + typeFeat + fileprefix + '.pckl', trn_data, tst_data, 0.0, 0.0 )

    print('Bone Approximated File HIST EQ Saved --{}'.format(datetime.now()-start))


