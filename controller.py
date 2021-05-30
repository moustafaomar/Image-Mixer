import logging
import numpy as np
import cv2
logging.basicConfig(filename="logging.log", format='%(asctime)s %(message)s', filemode='w') 
logger=logging.getLogger() 
logger.setLevel(logging.DEBUG) 
class ImageData:
    def __init__(self, location=None):
        if location is not None:
            self.img_data = cv2.imread(location, cv2.IMREAD_GRAYSCALE)
        else:
            self.img_data = 0
        try:
            self.getImageData()
        except:
            print('Image not found')
            logger.debug('Image at path %s not found',location)
    def getImageData(self):
        self.img_fft = np.fft.fft2(self.img_data)
        self.magnitude = np.abs(self.img_fft)
        self.phase = np.angle(self.img_fft)
        self.real = self.img_fft.real
        self.imaginary = self.img_fft.imag
        self.image_attributes = [np.abs((self.img_fft)), self.phase, self.img_fft.real, self.img_fft.imag]
    def get_data(self):
        return self.image_attributes
    def mix(self, imageToBeMixed, mode1, mode2, slider_val1,slider_val2,val2,val1) -> np.ndarray:
        phase=0
        mag=1
        real=0
        imaginary=0
        map = [self,imageToBeMixed]
        if (mode2==0 and mode1==1) or mode2 == 3 or mode2==5 or mode2==4:
                (slider_val1,slider_val2) = (slider_val2,slider_val1) 
        else:
                (val2,val1) = (val1,val2)
        if (val1==val2):
            (val1,val2,slider_val2) = (1-val1,val1,10-slider_val2)
        if (mode1,mode2)==(1,0) or (mode1,mode2)==(0,1):
                mag=map[val2].get_data()[min(mode1,mode2)]*(slider_val1/10)+map[val1].get_data()[min(mode1,mode2)]*((10-slider_val1)/10)
                phase= map[val1].get_data()[max(mode1,mode2)]*(slider_val2/10)+map[val2].get_data()[max(mode1,mode2)]*((10-slider_val2)/10)
        elif (mode1,mode2)==(3,2) or (mode1,mode2)==(2,3):
                real=map[val2].get_data()[min(mode1,mode2)]*(slider_val1/10)+map[val1].get_data()[min(mode1,mode2)]*((10-slider_val1)/10)
                imaginary= 1j * (map[val1].get_data()[max(mode1,mode2)]*(slider_val2/10)+map[val2].get_data()[max(mode1,mode2)]*((10-slider_val2)/10))
        elif (mode1,mode2) == (4,1) or (mode1,mode2) == (1,4):
                phase=map[val1].get_data()[min(mode1,mode2)]*(slider_val2/10)+map[val2].get_data()[min(mode1,mode2)]*((10-slider_val2)/10)
        elif  (mode1,mode2)==(5,0) or (mode1,mode2)==(0,5):
                mag=map[val1].get_data()[min(mode1,mode2)]*(slider_val2/10)+map[val2].get_data()[min(mode1,mode2)]*((10-slider_val2)/10)
        if mode1 in [0,1,4,5] or mode2 in [0,1,4,5]:
            exp = np.exp(1j * phase)
        try:
            mix = mag * exp * np.ones(self.img_data.shape)
            logger.info("Mixing Magnitude and Phase")
        except:
            mix=real+imaginary
            logger.info("Mixing Real and Imaginary")
        return np.fft.ifft2(mix)