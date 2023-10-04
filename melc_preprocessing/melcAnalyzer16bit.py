import sys
import os
import re
import glob
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import json
import cv2
from tqdm import trange, tqdm
#os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import matplotlib.pyplot as plt
from pylab import *                           # standard stuff
from scipy.signal import correlate2d          # drift correction
from scipy.ndimage import shift # subpixel image shift
import tifffile  # save 8bit single-channel TIF files


class ProjectOverview():
    def __init__(self, mainDir):
        self.mainDir = mainDir
        self.toBeProcessed = []

        ### CHECK FOR UNPROCESSED FIELD-OF-VIEWS
        subDirs = os.listdir(self.mainDir)

        mask = [bool(re.search(r'\d{12}_\d{1}', dir)) for dir in subDirs]  # create a mask for filtering proper sub-dirs
        properSubDirs = [dir for dir, m in zip(subDirs, mask) if m]  # masking without Numpy
        mask = [bool(re.search(r'\d{5}', dir)) for dir in subDirs]  # create a mask for filtering result dirs
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
                ini_path = os.path.join(mainDir, f, "inifile", "*.xml")
                iniFiles = glob.glob(ini_path)
                assert len(iniFiles) <= 1, "ERROR: Found multiple protocol files (in /inifile)."
                assert len(iniFiles) >= 1, "ERROR: Found no protocol files (in /inifile)."
                protFile = iniFiles[0]

                xmlTree = ET.parse(protFile)
                xmlRoot = xmlTree.getroot()

                # run ID (stored as string)
                runID = [m.attrib for m in xmlRoot.findall('{http://www.meltec.de/2004/xschema}run')][0][
                    '{http://www.w3.org/1999/xlink}href'].split(':')[-1]
                self.runIDs[i].append(runID)

                if runID in resultIDs:
                    l.append(True)
                else:
                    l.append(False)
            self.isProcessed.append(l)    


    def import_xml_with_experimental_protocol(self, dirName):# Import XML with experiment protocol
        iniFiles = glob.glob(dirName + '/inifile/*.xml')
        assert len(iniFiles) <= 1, "ERROR: Found multiple protocol files (in /inifile)."
        assert len(iniFiles) >= 1, "ERROR: Found no protocol files (in /inifile)."
        protFile = iniFiles[0]

        xmlTree = ET.parse(protFile)
        xmlRoot = xmlTree.getroot()

        # list of antibodies
        antibodies = [m.get('name') for m in
                        xmlRoot.findall('{http://www.meltec.de/2004/xschema}incStep/' \
                                        '{http://www.meltec.de/2004/xschema}channelStep/' \
                                        '{http://www.meltec.de/2004/xschema}marker')]

        # list of channels
        filters = [m.get('name') for m in
                    xmlRoot.findall('{http://www.meltec.de/2004/xschema}incStep/' \
                                    '{http://www.meltec.de/2004/xschema}channelStep/' \
                                    '{http://www.meltec.de/2004/xschema}fluorescenceFilter')]

        # list of exposure times (stored as string)
        exp_times = [m.text for m in
                        xmlRoot.findall('{http://www.meltec.de/2004/xschema}incStep/' \
                                        '{http://www.meltec.de/2004/xschema}channelStep/' \
                                        '{http://www.meltec.de/2004/xschema}exposureTime')]

        # run ID (stored as string)
        runID = [m.attrib for m in xmlRoot.findall('{http://www.meltec.de/2004/xschema}run')][0][
            '{http://www.w3.org/1999/xlink}href'].split(':')[-1]

        assert len(antibodies) == len(filters), "ERROR: Could not read incubation steps properly."
        assert len(antibodies) == len(exp_times), "ERROR: Could not read incubation steps properly."
        assert len(runID) > 0, "ERROR: Could not read RunID."

        print("+ RunID: %s" % runID)
        return antibodies, filters, exp_times, runID

    # drift correction
    def GetDriftCorrection(self, im1, im2):
        border = int(im1.shape[0] / 4. * 1.90)
        border2 = 4
        factor = 10.

        im1b = im1[border:-border, border:-border]
        im2b = im2[border + border2:-border - border2, border + border2:-border - border2]
    

        im2_5x = np.array(Image.fromarray(im2b).resize((int(im2b.shape[1] * factor), int(im2b.shape[0] * factor)), Image.BICUBIC))
        im1_5x = np.array(Image.fromarray(im1b).resize((int(im1b.shape[1] * factor), int(im1b.shape[0] * factor)), Image.BICUBIC))
        # im2_5x = imresize(im2b, factor, 'cubic', 'F')
        # im1_5x = imresize(im1b, factor, 'cubic', 'F')
        
        corr = correlate2d(im1_5x, im2_5x, 'valid')
        return (np.array(divmod(np.argmax(corr), corr.shape[0])) - np.array(
            [border2 * factor, border2 * factor])) / factor
        
        
    def ProcessAntibody(self, antibody_name, filter_name, exp_time, dirName, odirName, runID, subDir):
   

        # filter_oname assigns custom (output) filter name if known
        filter_oname = self.filter_otypes[filter_name] if filter_name in self.filter_otypes else filter_name

        filter_index = self.filter_types.index(filter_name)
        pFile = glob.glob(
            dirName + '/source/p_' + antibody_name + "_*_" + "*" + "_*.png")
        oFile = glob.glob(
            dirName + '/source/o_' + antibody_name + "_*" + filter_name + "_*.png")

        pbFile = glob.glob(
            dirName + '/bleach/pb_' + antibody_name + "_*_" + "*" + "_*.png")
        bFile = glob.glob(
            dirName + '/bleach/b_' + antibody_name + "_*" + filter_name + "_*.png")

        if antibody_name == "NONE":
            bIm = plt.imread(bFile[0])[15:-15, 15:-15]
            self.bIMRef = bIm
            self.t.set_description(f"Filter switch", refresh=True)
            return
        else:
            self.t.set_description(f"Processing {antibody_name} with filter {filter_oname}", refresh=True)

        pIm = plt.imread(pFile[0])[15:-15, 15:-15]
        oIm = plt.imread(oFile[0])[15:-15, 15:-15]
        pbIm = plt.imread(pbFile[0])[15:-15, 15:-15]
        bIm = plt.imread(bFile[0])[15:-15, 15:-15]

        bshift = (0, 0)

        # check whether first anti-body is being processed
        if type(self.pIMRef) == type(0):
            self.pIMRef = pIm
            # save phase contrast image
            #gray()
            pImCorr = np.copy(pIm)
            pImCorr = pImCorr - np.percentile(pImCorr, 20. * 0.135)
            pImCorr[pImCorr < 0] = 0
            pImCorr = pImCorr / np.percentile(pImCorr, 100 - 1. * 0.135)
            pImCorr[pImCorr > 1] = 1
            #tifffile.imsave(odirName + "/" + runID + "_phase.tif", (pImCorr * 255 * 255).astype(np.uint16))
            tifffile.imwrite(odirName + "/" + runID + "_phase.tif", (pImCorr * 255 * 255).astype(np.uint16))
            self.t.set_description(f"Processing {antibody_name}: saved phase contrast image", refresh=True)


        oshift = self.GetDriftCorrection(self.pIMRef, pIm)
        bshift = self.GetDriftCorrection(self.pIMRef, pbIm)
        

        if type(self.bIMRef) == type(0):
            self.bIMRef = bIm
            self.old_bshift = np.array([0., 0.])

        # fluorescence = oIm-self.bIMRef
        # fluorescence = shift( oIm*gain_im[filter_index], oshift)-shift( self.bIMRef*gain_im[filter_index], self.old_bshift)
        fluorescence = (shift(oIm, oshift) - shift(self.bIMRef, self.old_bshift)) * self.gain_im[filter_index]
        self.bIMRef = bIm
        #gray()
        fluorescence[fluorescence < 0] = 0
        fluorescence = fluorescence - np.percentile(fluorescence[20:-20, 20:-20], 0.135)
        fluorescence = fluorescence / np.percentile(fluorescence[20:-20, 20:-20], 100 - 0.135)
        fluorescence[fluorescence < 0] = 0
        fluorescence[fluorescence > 1] = 1
        tifffile.imwrite(
            odirName + "/" + runID + "_" + antibody_name + "_" + exp_time + "_" + str(self.counter).zfill(3) + ".tif",
            (fluorescence * 255 * 255).astype(np.uint16))
        self.t.set_description(f"Processing {antibody_name}: saved phase fluorescence image", refresh=True)
    

        bleach = np.copy(bIm)
        bleach[bleach < 0] = 0
        bleach = bleach - np.percentile(bleach[20:-20, 20:-20], 0.135)
        bleach = bleach / np.percentile(bleach[20:-20, 20:-20], 100 - 0.135)
        bleach[bleach < 0] = 0
        bleach[bleach > 1] = 1
        tifffile.imwrite(
            odirName + "/bleach/" + runID + "_" + antibody_name + "_" + exp_time + "_" + str(self.counter).zfill(
                3) + ".tif", (bleach * 255 * 255).astype(np.uint16))
        self.t.set_description(f"Processing {antibody_name}: saved corresponding bleach image", refresh=True)


        phase = np.copy(pIm)
        phase[phase < 0] = 0
        phase = phase - np.percentile(phase[20:-20, 20:-20], 0.135)

        phase = phase / np.percentile(phase[20:-20, 20:-20], 100 - 0.135)
        phase[phase < 0] = 0
        phase[phase > 1] = 1
        tifffile.imwrite(
            odirName + "/phase/" + runID + "_" + antibody_name + "_" + exp_time + "_" + str(self.counter).zfill(
                3) + ".tif", (phase * 255 * 255).astype(np.uint16))
        self.t.set_description(f"Processing {antibody_name}: saved corresponding phase image", refresh=True)


        self.old_bshift = bshift
    
    
    def startProcessing(self):
        self.toBeProcessed = self.fieldsOfView
        self.toBeProcessed = [item for sublist in self.toBeProcessed for item in sublist]  # flatten list
        print(self.toBeProcessed)
        
        tr = tqdm(self.toBeProcessed, desc=f"Processing", leave=True, position=0)
        
        for subDir in tr:
            tr.set_description("Processing {subDir}", refresh=True)
            # directory containing raw data
            dirName = os.path.join(self.mainDir, subDir)

            assert len(dirName) > 0, "ERROR: Not a valid directory name."
            print("Data directory", dirName)

            antibodies, filters, exp_times, runID = self.import_xml_with_experimental_protocol(dirName)
            print(antibodies)
            # create output-directory, if necessary
            odirName = os.path.join(self.mainDir, runID)
            if not os.path.exists(odirName):
                os.makedirs(odirName)
                print("+ Created output directory", runID)
            else:
                print("! NOTE: Directory already exists. Overwriting existing results.")

            bleach_dir = os.path.join(odirName, "bleach")
            phase_dir = os.path.join(odirName, "phase")
            if not os.path.exists(bleach_dir):
                os.makedirs(bleach_dir)
            if not os.path.exists(phase_dir):
                os.makedirs(phase_dir)

       
            # filter types and custom names
            self.filter_types = np.unique(filters).tolist()
            self.filter_otypes = {'XF111-2': 'PE', 'XF116-2': 'FITC'}

            self.pIMRef = 0
            self.bIMRef = 0
        
            if len(glob.glob(dirName + '/source/o_cal_b*_5000_*_000.png')) > 0:
                print('+ found calibration images with exposure time 5000.')
                brightfield_names = [glob.glob(dirName + '/source/o_cal_b*_5000_' + filt + '_000.png')[0] for filt in
                                     self.filter_types]
                brightfield_im = [plt.imread(name)[15:-15, 15:-15] for name in brightfield_names]
                darkframe_names = [glob.glob(dirName + '/source/o_cal_d*_5000_' + filt + '_000.png') for filt in
                                  self.filter_types]
                darkframe_im = [np.mean([plt.imread(name)[15:-15, 15:-15] for name in name_list], axis=0) for name_list in
                                darkframe_names]
            else:
                print('! found no calibration images with exposure time 5000.')
                print('! trying to access calibration images with exposure time 2500.')
                brightfield_names = [glob.glob(dirName + '/source/o_cal_b*_2500_' + filt + '_000.png')[0] for filt in
                                     self.filter_types]
                brightfield_im = [plt.imread(name)[15:-15, 15:-15] for name in brightfield_names]
                darkframe_names = [glob.glob(dirName + '/source/o_cal_d*_2500_' + filt + '_000.png') for filt in
                                  self.filter_types]
                darkframe_im = [np.mean([plt.imread(name)[15:-15, 15:-15] for name in name_list], axis=0) for name_list in
                                darkframe_names]

            corrected_brightfield_im = [(brightfield_im[i] - darkframe_im[i]) for i in
                                        range(len(self.filter_types))]
            mean_corrected_brightfield_im = [np.mean(brightfield_im[i] - darkframe_im[i]) for i in
                                             range(len(self.filter_types))]
            corrected_brightfield_im[0][corrected_brightfield_im[0] <= 0] = mean_corrected_brightfield_im[0]
            corrected_brightfield_im[1][corrected_brightfield_im[1] <= 0] = mean_corrected_brightfield_im[1]

            self.gain_im = [mean_corrected_brightfield_im[i] / corrected_brightfield_im[i] for i in
                       range(len(self.filter_types))]
            self.gain_im[0][self.gain_im[0] < 0] = 0
            self.gain_im[0][self.gain_im[0] > 100] = 100
            self.gain_im[1][self.gain_im[1] < 0] = 0
            self.gain_im[1][self.gain_im[1] > 100] = 100

            print('+ Computed flat-field.')

            self.counter = 1  # put numbers on output files (do not count antibody "NONE")
            self.t = trange(len(antibodies), desc=f"Processing {subDir}", leave=False, position=1)
            for i in self.t:
                self.ProcessAntibody(antibodies[i], filters[i], exp_times[i], dirName, odirName, runID, subDir)
                self.counter += 1

            print("+ Finished RunID " + runID)

        print("### Finished. ###")



def main():
    f = open('melc_config.json')
    config = json.load(f)
    mainDir = config["als_test"]
    #mainDir = os.path.join(mainDir, "Eczema")
    assert len(mainDir) > 0, 'ERROR: No main directory chosen.'
    po = ProjectOverview(mainDir)
    po.startProcessing()

    ### END


if __name__ == "__main__":
    main()