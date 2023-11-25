import cv2
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

class saveBoneApproximationToFiles:

    def findobjectsbycontours(self, img, min_size=600):
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.Canny(img, 50, 150)
        edges = cv2.dilate(edges, kernel, iterations=1)
        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            edges, connectivity, cv2.CV_32S)
        sizes = stats[1:, -1]
        num_labels = num_labels - 1
        # your answer image
        img2 = np.zeros(labels.shape)
        for i in range(0, num_labels):
            if sizes[i] >= min_size:
                img2[labels == i + 1] = 255

        contours, hierarchy = cv2.findContours(img2.astype(
            np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        retsize = len(contours)

        return img2, retsize

    def applyMultiThreshold(self, X_patIDs, myparams, typeFeat, finaloutfilename, trn_data, tst_data, 
                            thresh1, threshsize1, minthreshsize1, minthreshsize2, fileprefix, saveFlag, applyEqHist):

        trn_dataz, tst_dataz = np.zeros((trn_data.shape[0], trn_data.shape[1], trn_data.shape[2], 1)), \
            np.zeros((tst_data.shape[0], tst_data.shape[1], tst_data.shape[2], 1))

        dfvalues1 = np.zeros((trn_data.shape[0], len(thresh1)+1))
        dfvalues1tst = np.zeros((tst_data.shape[0], len(thresh1)+1))

        for idx, thresh in enumerate(thresh1):
            for ii in range(myparams[typeFeat]['learnRate'].shape[0]):
                outputfilename = outputpath + \
                    myparams[typeFeat]['targetprefixes'][ii][1]
                if fileprefix is None:
                    restrn_data, restst_data, _, _ = readFromPickleFile(
                        outputfilename + '.pckl')
                else:
                    restrn_data, restst_data, _, _ = readFromPickleFile(
                        outputfilename + fileprefix + '.pckl')
                restrn_data, restst_data = np.trunc(np.clip(
                    (restrn_data * 255), 0, 255)), np.trunc(np.clip((restst_data * 255), 0, 255))
                restrn_data, restst_data = np.where(restrn_data > thresh, restrn_data, 0), np.where(
                    restst_data > thresh, restst_data, 0)
                for jj in range(restrn_data.shape[0]):
                    tmp = np.squeeze(restrn_data[jj, :, :, :]).astype(np.uint8)
                    _, retsize1 = self.findobjectsbycontours(tmp.copy(), threshsize1)
                    dfvalues1[jj, idx] = dfvalues1[jj, idx] + retsize1
                for jj in range(restst_data.shape[0]):
                    tmp = np.squeeze(restst_data[jj, :, :, :]).astype(np.uint8)
                    _, retsize1 = self.findobjectsbycontours(tmp.copy(), threshsize1)
                    dfvalues1tst[jj, idx] = dfvalues1tst[jj, idx] + retsize1
                # restrn_data[jj, :, :, :] = np.reshape(imgmask2, (1, myparams[typeFeat]['height'], myparams[typeFeat]['width'], 1 ) )
            # trn_dataz, tst_dataz = trn_dataz + restrn_data, tst_dataz + restst_data

        st1, ed1, st2, ed2 = 0, 6, 6, 11
        st1, ed1, st2, ed2 = 0, 5, 5, 10
        st1, ed1, st2, ed2 = 0, 7, 7, 11
        # st1, ed1, st2, ed2 = 0, 7, 7, 10
        dfvalues1 = dfvalues1 / myparams[typeFeat]['learnRate'].shape[0]
        dfvalues1[:, -1] = dfvalues1[:, :-1].mean(axis=1)

        selthresh11arr05 = dfvalues1[:, st1:ed1]
        mnselthresh11arr05 = selthresh11arr05.mean(axis=1)

        selthresh11arr610 = dfvalues1[:, st2:ed2]
        mnselthresh11arr610 = selthresh11arr610.mean(axis=1)

        dfvalues1tst = dfvalues1tst / myparams[typeFeat]['learnRate'].shape[0]
        dfvalues1tst[:, -1] = dfvalues1tst[:, :-1].mean(axis=1)

        selthresh11arr05tst = dfvalues1tst[:, st1:ed1]
        mnselthresh11arr05tst = selthresh11arr05tst.mean(axis=1)

        selthresh11arr610tst = dfvalues1tst[:, st2:ed2]
        mnselthresh11arr610tst = selthresh11arr610tst.mean(axis=1)

        tmpselthresh1, mnselthresh1 = np.zeros(
            (trn_data.shape[0], 1)), np.zeros((trn_data.shape[0], 1))
        tmpselthresh1tst, mnselthresh1tst = np.zeros(
            (tst_data.shape[0], 1)), np.zeros((tst_data.shape[0], 1))

        for ii in range(myparams[typeFeat]['learnRate'].shape[0]):
            outputfilename = outputpath + \
                myparams[typeFeat]['targetprefixes'][ii][1]
            restrn_data, restst_data, _, _ = readFromPickleFile(
                outputfilename + '.pckl')
            restrn_data, restst_data = np.trunc(np.clip(
                (restrn_data * 255), 0, 255)), np.trunc(np.clip((restst_data * 255), 0, 255))
            for jj in range(restrn_data.shape[0]):
                # if dfvalues1[jj, -1] < 5:
                #     restrn_data[jj, :, :, :] = np.where(restrn_data[jj, :, :, :], 1, 0)
                # elif( dfvalues1[jj, -1]<11 ) and \
                #     ( ( np.abs( mnselthresh11arr05[jj] - mnselthresh11arr610[jj] ) <= 1 ) or  \
                #       ( np.abs( mnselthresh11arr05[jj] - dfvalues1[jj, -1] ) <= 1 ) or \
                #       ( np.abs( dfvalues1[jj, -1] - mnselthresh11arr610[jj] ) <= 1 ) ):
                #     restrn_data[jj, :, :, :] = np.where(restrn_data[jj, :, :, :], 1, 0)
                # else:
                if mnselthresh11arr05[jj] >= mnselthresh11arr610[jj]:
                    tmp = selthresh11arr05[jj, :].copy()
                    tmp[tmp < minthreshsize1] = -999
                    if all([x < 0 for x in tmp]):
                        tmpselthresh1[jj, 0] = 0
                    else:
                        tmp = np.abs(tmp - mnselthresh11arr05[jj])
                        # tmp = np.abs( tmp - dfvalues1[jj, -1] )
                        result = np.amin(np.where(tmp == np.amin(tmp)))
                        tmpselthresh1[jj, 0] = (thresh1[st1:ed1])[result]
                else:
                    tmp = selthresh11arr610[jj, :].copy()
                    tmp[tmp < minthreshsize2] = -999
                    if all([x < 0 for x in tmp]):
                        tmpselthresh1[jj, 0] = 0
                    else:
                        tmp = np.abs(tmp - mnselthresh11arr610[jj])
                        # tmp = np.abs( tmp - dfvalues1[jj, -1] )
                        result = np.amax(np.where(tmp == np.amin(tmp)))
                        tmpselthresh1[jj, 0] = (thresh1[st2:ed2])[result]
                restrn_data[jj, :, :, :] = np.where(
                    restrn_data[jj, :, :, :] > tmpselthresh1[jj, 0], 1., 0.)

            for jj in range(restst_data.shape[0]):
                if mnselthresh11arr05tst[jj] >= mnselthresh11arr610tst[jj]:
                    tmp = selthresh11arr05tst[jj, :].copy()
                    tmp[tmp < minthreshsize1] = -999
                    if all([x < 0 for x in tmp]):
                        tmpselthresh1tst[jj, 0] = 0
                    else:
                        tmp = np.abs(tmp - mnselthresh11arr05tst[jj])
                        # tmp = np.abs( tmp - dfvalues1tst[jj, -1] )
                        result = np.amin(np.where(tmp == np.amin(tmp)))
                        tmpselthresh1tst[jj, 0] = (thresh1[st1:ed1])[result]
                else:
                    tmp = selthresh11arr610tst[jj, :].copy()
                    tmp[tmp < minthreshsize2] = -999
                    if all([x < 0 for x in tmp]):
                        tmpselthresh1tst[jj, 0] = 0
                    else:
                        tmp = np.abs(tmp - mnselthresh11arr610tst[jj])
                        # tmp = np.abs( tmp - dfvalues1tst[jj, -1] )
                        result = np.amax(np.where(tmp == np.amin(tmp)))
                        tmpselthresh1tst[jj, 0] = (thresh1[st2:ed2])[result]
                restst_data[jj, :, :, :] = np.where(
                    restst_data[jj, :, :, :] > tmpselthresh1tst[jj, 0], 1., 0.)

            trn_dataz, tst_dataz = trn_dataz + restrn_data, tst_dataz + restst_data

        trn_dataz, tst_dataz = np.trunc(np.divide(trn_dataz, (myparams[typeFeat]['learnRate'].shape[0]))), np.trunc(
            np.divide(tst_dataz, (myparams[typeFeat]['learnRate'].shape[0])))
        trn_data, tst_data = np.where(
            trn_dataz, trn_data, 0), np.where(tst_dataz, tst_data, 0)
        if saveFlag:
            tabhead = {}
            for idx, thresh in enumerate(thresh1):
                tabhead[str(thresh)] = dfvalues1[:, idx]

            tabhead['Mean'] = dfvalues1[:, -1]
            tabhead['Mean05'] = mnselthresh11arr05
            tabhead['Mean610'] = mnselthresh11arr610
            tabhead['SelctThresh1'] = tmpselthresh1[:, 0]
            dataset = pd.DataFrame(tabhead)
            dataset['patID'] = X_patIDs
            # if os.path.exists( finaloutfilename + typeFeat + '-11.csv' ):
            #     dfval = pd.read_csv(finaloutfilename + typeFeat + '-11.csv')
            #     dataset = dfval.append(dataset, ignore_index=True)
            dataset.to_csv(finaloutfilename + typeFeat + '-11.csv', index=False)

        if applyEqHist:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            print('Applying CLAHE')
            applyAll = True
            for jj in range(trn_data.shape[0]):
                if tmpselthresh1[jj] == 0 or applyAll:
                    img = np.squeeze(trn_data[jj, :, :, :]).astype(np.uint8)
                    img[img < 20] = 0.0
                    # img = cv2.equalizeHist( img ).astype(np.float32)
                    img = clahe.apply(img)
                    trn_data[jj, :, :, :] = np.reshape(
                        img, (1, img.shape[0], img.shape[1], 1))

            for jj in range(tst_data.shape[0]):
                if tmpselthresh1tst[jj] == 0 or applyAll:
                    img = np.squeeze(tst_data[jj, :, :, :]).astype(np.uint8)
                    img[img < 20] = 0.0
                    # img = cv2.equalizeHist( img ).astype(np.float32)
                    img = clahe.apply(img)
                    tst_data[jj, :, :, :] = np.reshape(
                        img, (1, img.shape[0], img.shape[1], 1))
        # print( sum(tmpselthresh1[jj] == 0) )
        return trn_data, tst_data


    def applySingleThreshold(self, X_patIDs, myparams, typeFeat, finaloutfilename, trn_data, tst_data, fileprefix, saveFlag):

        trn_dataz, tst_dataz = np.zeros((trn_data.shape[0], trn_data.shape[1], trn_data.shape[2], 1), dtype=np.float32), \
            np.zeros((tst_data.shape[0], tst_data.shape[1],
                     tst_data.shape[2], 1), dtype=np.float32)
        trn_dataz1, tst_dataz1 = np.zeros((trn_data.shape[0], trn_data.shape[1], trn_data.shape[2], 1), dtype=np.float32), \
            np.zeros((tst_data.shape[0], tst_data.shape[1],
                     tst_data.shape[2], 1), dtype=np.float32)
        trn_dataz2, tst_dataz2 = np.zeros((trn_data.shape[0], trn_data.shape[1], trn_data.shape[2], 1), dtype=np.float32), \
            np.zeros((tst_data.shape[0], tst_data.shape[1],
                     tst_data.shape[2], 1), dtype=np.float32)

        thresh = 20
        for ii in range(myparams[typeFeat]['learnRate'].shape[0]):
            outputfilename = outputpath + \
                myparams[typeFeat]['targetprefixes'][ii][1]
            restrn_data, restst_data, _, _ = readFromPickleFile(
                outputfilename + fileprefix + '.pckl')
            restrn_data, restst_data = np.clip(
                (restrn_data * 255), 0, 255), np.clip((restst_data * 255), 0, 255)
            restrn_data, restst_data = np.where(restrn_data > thresh, restrn_data, 0), np.where(
                restst_data > thresh, restst_data, 0)
            trn_dataz, tst_dataz = trn_dataz + restrn_data, tst_dataz + restst_data

            restrn_data1, restst_data1, _, _ = readFromPickleFile(
                outputfilename + '.pckl')
            restrn_data1, restst_data1 = np.clip(
                (restrn_data1 * 255), 0, 255), np.clip((restst_data1 * 255), 0, 255)
            restrn_data1, restst_data1 = np.where(restrn_data1 > thresh, restrn_data1, 0), np.where(
                restst_data1 > thresh, restst_data1, 0)
            trn_dataz1, tst_dataz1 = trn_dataz1 + restrn_data1, tst_dataz1 + restst_data1

            trn_dataz2, tst_dataz2 = trn_dataz2 + restrn_data + \
                restrn_data1, tst_dataz2 + restst_data + restst_data1

        # trn_dataz2, tst_dataz2 = trn_dataz2 + trn_data, tst_dataz2 + tst_data

        # trn_data, tst_data = np.clip( trn_dataz2 / (myparams[typeFeat]['learnRate'].shape[0]*2) + 1, 0, 255 ) , np.clip( tst_dataz2 / (myparams[typeFeat]['learnRate'].shape[0]*2) + 1, 0, 255 )

        trn_dataz, tst_dataz = np.clip(np.divide(trn_dataz, (myparams[typeFeat]['learnRate'].shape[0])), 0, 255), np.clip(
            np.divide(tst_dataz, (myparams[typeFeat]['learnRate'].shape[0])), 0, 255)
        trn_dataz1, tst_dataz1 = np.clip(np.divide(trn_dataz1, (myparams[typeFeat]['learnRate'].shape[0])), 0, 255), np.clip(
            np.divide(tst_dataz1, (myparams[typeFeat]['learnRate'].shape[0])), 0, 255)

        # # trn_dataz, tst_dataz = np.clip( np.divide( trn_dataz, (myparams[typeFeat]['learnRate'].shape[0]) ), 0, 255 ), np.clip ( np.divide( tst_dataz, (myparams[typeFeat]['learnRate'].shape[0] ) ), 0, 255 )
        # trn_dataz1, tst_dataz1 = np.clip( np.divide( trn_dataz1, (myparams[typeFeat]['learnRate'].shape[0]) ), 0, 255 ), np.clip ( np.divide( tst_dataz1, (myparams[typeFeat]['learnRate'].shape[0] ) ), 0, 255 )
        # trn_data, tst_data = np.clip( np.divide( trn_data + trn_dataz, 2 ), 0, 255 ), np.clip ( np.divide( tst_dataz + tst_data, 2 ), 0, 255 )
        # trn_data, tst_data = np.clip( np.divide( trn_data + trn_dataz1, 2 ), 0, 255 ), np.clip ( np.divide( tst_dataz1 + tst_data, 2 ), 0, 255 )

        trn_data, tst_data = np.clip(np.divide(trn_data + trn_dataz + trn_dataz1, 3),
                                     0, 255), np.clip(np.divide(tst_dataz + tst_data + tst_dataz1, 3), 0, 255)
        # trn_data, tst_data = np.clip( np.divide( trn_data + trn_dataz1, 2 ), 0, 255 ), np.clip ( np.divide( tst_dataz1 + tst_data, 2 ), 0, 255 )

        return trn_data, tst_data

    


