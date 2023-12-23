import cv2
import OpenEXR
import Imath
#needed lib: OpenEXR

def loadexr(path_exr): # path_exr: absolute path of one exr file.

    file_ = OpenEXR.InputFile(path_exr)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = file_.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    image1 = [Image.frombytes("F", size, file_.channel(c, pt)) for c in "G"]      
    d_image = np.array(image1[0].convert('L'), dtype = np.float32)
            
    #d_image = Image.open(d_tag).convert('L')#cv2.imread(d_tag, cv2.IMREAD_UNCHANGED)
    dep = np.array(d_image, dtype = np.float32)
    dep = dep.transpose(0,1) #depth array
    
    return dep
