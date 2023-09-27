import sys
import os
import re
import glob
import numpy as np
import xml.etree.ElementTree as ET
from PyQt4 import QtGui, QtCore

from pylab import *                           # standard stuff
from scipy.signal import correlate2d          # drift correction
from scipy.misc import imresize               # image resize
from scipy.ndimage.interpolation import shift # subpixel image shift
import tifffile  # save 8bit single-channel TIF files
#import cv2

print("### melcAnalyzer started - version 1.0 ###\n")

class OpenDir(QtGui.QMainWindow):
    def __init__(self):
        super(OpenDir, self).__init__()
        self.openDirectory()

    def openDirectory(self):
        print("+ Choose main directory...")
        openDirectoryDialog=QtGui.QFileDialog()
        oD=openDirectoryDialog.getExistingDirectory(self, "MELC - Choose main directory", "V:\FORSCHUNG\MELC\MELC-Lauf\MELC-Lauf/")  # selects folder
        self.mainDir = str(oD)

class ProjectOverview(QtGui.QWidget):
    def __init__(self, mainDir, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.mainDir = mainDir
        self.toBeProcessed = []

        self.setWindowTitle('MELC - Project Overview')
        
        ### CHECK FOR UNPROCESSED FIELD-OF-VIEWS
        subDirs = next(os.walk(self.mainDir))[1]

        mask = [bool(re.search(r'\d{12}_\d{1}', dir)) for dir in subDirs] # create mask for filtering proper sub-dirs
        properSubDirs = [dir for dir, m in zip(subDirs, mask) if m] # masking without Numpy

        mask = [bool(re.search(r'\d{5}', dir)) for dir in subDirs] # create mask for filtering result dirs
        resultIDs = [dir for dir, m in zip(subDirs, mask) if m]

        projects = list(set([dir[:-2] for dir in properSubDirs]))

        # list structure for field-of-views
        self.fieldsOfView = []
        self.runIDs = []
        for p in projects:
            l = []
            for d in properSubDirs:
                if d[:-2] == p:
                    l.append(d)
            self.fieldsOfView.append(l)
            self.runIDs.append([])

        # check which field of views have already been processed
        self.isProcessed = []
        for i, p in enumerate(self.fieldsOfView):
            l = []
            for f in p:
                # Import XML with experiment protocol
                iniFiles = glob.glob(mainDir+ '/' + f + '/inifile/*.xml')
                assert len(iniFiles) <= 1, "ERROR: Found multiple protocol files (in /inifile)."
                assert len(iniFiles) >= 1, "ERROR: Found no protocol files (in /inifile)."
                protFile = iniFiles[0]

                xmlTree = ET.parse(protFile)
                xmlRoot = xmlTree.getroot()
            
                # run ID (stored as string)
                runID =  [m.attrib for m in xmlRoot.findall('{http://www.meltec.de/2004/xschema}run')][0]['{http://www.w3.org/1999/xlink}href'].split(':')[-1]
                self.runIDs[i].append(runID)
                
                if runID in resultIDs:
                    l.append(True)
                else:
                    l.append(False)
            self.isProcessed.append(l)
        ### END
        
        ### CREATE CHECKBOXES CORRESPONDING TO PROJECTS
        self.boxes = []
        for n in range(len(projects)):                   
            if np.all(self.isProcessed[n]):
                self.boxes.append(QtGui.QCheckBox('Project: ' + projects[n][6:8] + '.' + projects[n][4:6] + '.' + projects[n][:4] + '     ' + 
                                                  str(len(self.fieldsOfView[n])) + ' field(s) of view     Already processed.\n   '+str(self.runIDs[n])))
                self.boxes[n].setCheckState(QtCore.Qt.Unchecked)
            else:
                self.boxes.append(QtGui.QCheckBox('Project: ' + projects[n][6:8] + '.' + projects[n][4:6] + '.' + projects[n][:4] + '     ' + 
                                                  str(len(self.fieldsOfView[n])) + ' field(s) of view     ' + str(len(self.isProcessed[n])-np.sum(self.isProcessed[n])) + ' UNPROCESSED\n   '+str(self.runIDs[n])))
                self.boxes[n].setCheckState(QtCore.Qt.Checked)
            self.boxes[n].setCheckable(True)
        ### END
        
        start = QtGui.QPushButton("Start processing")
        start.clicked.connect(self.startProcessing)
        
        exit = QtGui.QPushButton("Exit")
        exit.clicked.connect(QtCore.QCoreApplication.instance().quit)

        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(start)
        hbox.addWidget(exit)

        vbox = QtGui.QVBoxLayout()
        vbox.addStretch(1)
        for box in self.boxes:
            vbox.addWidget(box)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

    def startProcessing(self):
        global pImRef, bImRef, old_bshift, counter
        self.close()  # close window after 'Start processing is pushed'
        checkedQ = [box.isChecked() for box in self.boxes]
        self.toBeProcessed = [z[0] for z in zip(self.fieldsOfView, checkedQ) if z[1]]  # reprocess partly processed projects...
        self.toBeProcessed = [item for sublist in self.toBeProcessed for item in sublist]  # flatten list
        
        if len(self.toBeProcessed) == 0:
            print("! NOTE: No projects chosen.")
        else:
            print('+ Started processing ' + str(np.sum(np.array(checkedQ))) + ' projects.')
        
        # loop over field-of-views that need to be processed
        for subDir in self.toBeProcessed:
            # directory containing raw data
            dirName = self.mainDir + '\\' + subDir

            assert len(dirName) > 0, "ERROR: Not a valid directory name."
            print("\n+ Data directory:")
            print("  " + dirName)

            # Import XML with experiment protocol
            iniFiles = glob.glob(dirName + '/inifile/*.xml')
            assert len(iniFiles) <= 1, "ERROR: Found multiple protocol files (in /inifile)."
            assert len(iniFiles) >= 1, "ERROR: Found no protocol files (in /inifile)."
            protFile = iniFiles[0]

            xmlTree = ET.parse(protFile)
            xmlRoot = xmlTree.getroot()

            # list of antibodies
            antibodies = [m.get('name') for m in xmlRoot.findall('{http://www.meltec.de/2004/xschema}incStep/'\
                                                                 '{http://www.meltec.de/2004/xschema}channelStep/'\
                                                                 '{http://www.meltec.de/2004/xschema}marker')]

            # list of channels
            filters = [m.get('name') for m in xmlRoot.findall('{http://www.meltec.de/2004/xschema}incStep/'\
                                                              '{http://www.meltec.de/2004/xschema}channelStep/'\
                                                              '{http://www.meltec.de/2004/xschema}fluorescenceFilter')]

            # list of exposure times (stored as string)
            exp_times = [m.text for m in xmlRoot.findall('{http://www.meltec.de/2004/xschema}incStep/'\
                                                         '{http://www.meltec.de/2004/xschema}channelStep/'\
                                                         '{http://www.meltec.de/2004/xschema}exposureTime')]
                                                              
            # run ID (stored as string)
            runID = [m.attrib for m in xmlRoot.findall('{http://www.meltec.de/2004/xschema}run')][0]['{http://www.w3.org/1999/xlink}href'].split(':')[-1]

            assert len(antibodies) == len(filters),  "ERROR: Could not read incubation steps properly."
            assert len(antibodies) == len(exp_times), "ERROR: Could not read incubation steps properly."
            assert len(runID) > 0, "ERROR: Could not read RunID."

            print("+ RunID: %s" % runID)

            # create output-directory, if necessary
            odirName = self.mainDir + '\\' + runID
            if not os.path.exists(odirName):
                os.makedirs(odirName)
                print("+ Created output directory (...\\" + runID + ")")
            else:
                print("! NOTE: Directory already exists. Overwriting existing results.")

            if not os.path.exists(odirName+'\\bleach'):
                os.makedirs(odirName+'\\bleach')
            if not os.path.exists(odirName+'\\phase'):
                os.makedirs(odirName+'\\phase')
            
            # drift correction
            def GetDriftCorrection(im1, im2):
                border = im1.shape[0]/4*1.90
                border2 = 4
                factor = 10.
                im1b = im1[border:-border,border:-border]
                im2b = im2[border+border2:-border-border2,border+border2:-border-border2]
                im2_5x = imresize(im2b, factor, 'cubic', 'F')
                im1_5x = imresize(im1b, factor, 'cubic', 'F')

                corr = correlate2d(im1_5x, im2_5x, 'valid')
                return (array(divmod(argmax(corr),corr.shape[0]))-array([border2*factor,border2*factor]))/factor

            # drift correction
            # def GetDriftCorrection(im1, im2):
            #     border = im1.shape[0]/4*1.90
            #     border2 = 10
            #     im1b = im1[border:-border, border:-border]
            #     im2b = im2[border+border2:-border-border2, border+border2:-border-border2]
            #
            #     # template matching for drift correction
            #     res = cv2.matchTemplate(im1b, im2b, cv2.TM_CCOEFF)
            #     res += np.amin(res)
            #     res = res**4.
            #     res /= np.sum(res)
            #
            #     idx = np.mgrid[0:res.shape[1], 0:res.shape[0]]
            #     shift = [np.sum(res*idx[1]) - (res.shape[1]/2-1), np.sum(res*idx[0]) - (res.shape[1]/2-1)]
            #
            #     print shift
            #     return shift

            pImRef = 0
            bImRef = 0

            # filter types and custom names
            filter_types = unique(filters).tolist()
            filter_otypes = {'XF111-2':'PE', 'XF116-2':'FITC'}

            if len(glob.glob(dirName + '/source/o_cal_b*_5000_*_000.png')) > 0:
                print('+ found calibration images with exposure time 5000.')
                brightfield_names = [glob.glob(dirName + '/source/o_cal_b*_5000_'+filt+'_000.png')[0] for filt in filter_types]
                brightfield_im = [imread(name)[15:-15, 15:-15] for name in brightfield_names]
                darkframe_names = [glob.glob(dirName + '/source/o_cal_d*_5000_'+filt+'_000.png') for filt in filter_types]
                darkframe_im = [mean([imread(name)[15:-15, 15:-15] for name in name_list], axis=0) for name_list in darkframe_names]
            else:
                print('! found no calibration images with exposure time 5000.')
                print('! trying to access calibration images with exposure time 2500.')
                brightfield_names = [glob.glob(dirName + '/source/o_cal_b*_2500_'+filt+'_000.png')[0] for filt in filter_types]
                brightfield_im = [imread(name)[15:-15, 15:-15] for name in brightfield_names]
                darkframe_names = [glob.glob(dirName + '/source/o_cal_d*_2500_'+filt+'_000.png') for filt in filter_types]
                darkframe_im = [mean([imread(name)[15:-15, 15:-15] for name in name_list], axis=0) for name_list in darkframe_names]
               
            corrected_brightfield_im = [(brightfield_im[i]-darkframe_im[i]) for i in xrange(len(filter_types))]
            mean_corrected_brightfield_im = [mean(brightfield_im[i]-darkframe_im[i]) for i in xrange(len(filter_types))]
            corrected_brightfield_im[0][corrected_brightfield_im[0] <= 0] = mean_corrected_brightfield_im[0]
            corrected_brightfield_im[1][corrected_brightfield_im[1] <= 0] = mean_corrected_brightfield_im[1]

            gain_im = [mean_corrected_brightfield_im[i]/corrected_brightfield_im[i] for i in xrange(len(filter_types))]
            gain_im[0][gain_im[0] < 0] = 0
            gain_im[0][gain_im[0] > 100] = 100
            gain_im[1][gain_im[1] < 0] = 0
            gain_im[1][gain_im[1] > 100] = 100

            print('+ Computed flat-field.')

            counter = 1  # put numbers on output files (do not count antibody "NONE")
            def ProcessAntibody(antibody_name, filter_name, exp_time):
                global pImRef, bImRef, old_bshift, counter
                # filter_oname assigns custom (output) filter name if known
                filter_oname = filter_otypes[filter_name] if filter_name in filter_otypes else filter_name
                
                filter_index = filter_types.index(filter_name)
                pFile = glob.glob(dirName + '/source/p_'+antibody_name+"_*_"+"*"+"_*.png")
                oFile = glob.glob(dirName + '/source/o_'+antibody_name+"_*_"+filter_name+"_*.png")

                pbFile = glob.glob(dirName + '/bleach/pb_'+antibody_name+"_*_"+"*"+"_*.png")
                bFile = glob.glob(dirName + '/bleach/b_'+antibody_name+"_*_"+filter_name+"_*.png")

                if antibody_name == "NONE":
                    bIm  = imread( bFile[0])[15:-15,15:-15]
                    bImRef = bIm
                    print("+ Processing filter switch.")
                    return
                else:
                    print('+ Processing antibody: '+ antibody_name +' with filter: '+ filter_oname + '.')

                pIm  = imread( pFile[0])[15:-15,15:-15]
                oIm  = imread( oFile[0])[15:-15,15:-15]
                pbIm = imread(pbFile[0])[15:-15,15:-15]
                bIm  = imread( bFile[0])[15:-15,15:-15]

                bshift = (0,0)
                
                # check whether first anti-body is being processed
                if type(pImRef) == type(0):
                    pImRef = pIm
                    # save phase contrast image
                    gray()
                    pImCorr = pIm.copy()
                    pImCorr = pImCorr-percentile(pImCorr,20.*0.135)
                    pImCorr[pImCorr<0] = 0
                    pImCorr = pImCorr/percentile(pImCorr,100-1.*0.135)
                    pImCorr[pImCorr>1] = 1
                    tifffile.imsave(odirName+"/"+runID+"_phase.tif", (pImCorr*255*255).astype(np.uint16))
                    print('+ Saved phase contrast image.')
                
                oshift = GetDriftCorrection(pImRef, pIm)
                bshift = GetDriftCorrection(pImRef, pbIm)
                    

                if type(bImRef) == type(0):
                    bImRef = bIm
                    old_bshift = np.array([0., 0.])
                
                #fluorescence = oIm-bImRef
                #fluorescence = shift( oIm*gain_im[filter_index], oshift)-shift( bImRef*gain_im[filter_index], old_bshift)
                fluorescence = (shift(oIm, oshift)-shift(bImRef, old_bshift))*gain_im[filter_index]
                bImRef = bIm
                gray()
                fluorescence[fluorescence<0] = 0
                fluorescence = fluorescence-percentile(fluorescence[20:-20,20:-20],0.135)
                fluorescence = fluorescence/percentile(fluorescence[20:-20,20:-20],100-0.135)
                fluorescence[fluorescence<0] = 0
                fluorescence[fluorescence>1] = 1
                tifffile.imsave(odirName+"/"+runID+"_"+antibody_name+"_"+exp_time+"_"+str(counter).zfill(3)+".tif", (fluorescence*255*255).astype(np.uint16))
                print('+ Saved fluorescence image.')

                bleach = bIm.copy()
                bleach[bleach<0] = 0
                bleach = bleach-percentile(bleach[20:-20,20:-20],0.135)
                bleach = bleach/percentile(bleach[20:-20,20:-20],100-0.135)
                bleach[bleach<0] = 0
                bleach[bleach>1] = 1
                tifffile.imsave(odirName+"/bleach/"+runID+"_"+antibody_name+"_"+exp_time+"_"+str(counter).zfill(3)+".tif", (bleach*255*255).astype(np.uint16))
                print('+ Saved corresponding bleach image.')

                phase = pIm.copy()
                phase[phase<0] = 0
                phase = phase-percentile(phase[20:-20,20:-20],0.135)
                phase = phase/percentile(phase[20:-20,20:-20],100-0.135)
                phase[phase<0] = 0
                phase[phase>1] = 1
                tifffile.imsave(odirName+"/phase/"+runID+"_"+antibody_name+"_"+exp_time+"_"+str(counter).zfill(3)+".tif", (phase*255*255).astype(np.uint16))
                print('+ Saved corresponding phase image.')
                
                old_bshift = bshift
                counter += 1


            for i in xrange(len(antibodies)):
                ProcessAntibody(antibodies[i], filters[i], exp_times[i])
            
            print("+ Finished RunID " + runID)
        
        print("\n### Finished. ###")

app = QtGui.QApplication(sys.argv)

# open-folder dialog
od = OpenDir()
mainDir = od.mainDir
assert len(mainDir) > 0, 'ERROR: No main directory chosen.'
print('+ Main directory: ' + mainDir)

# project overview window
po = ProjectOverview(mainDir)
po.show()

sys.exit(app.exec_())
