import os
import cv2
import numpy as np
import shutil
from quant_checker.utils.utils import resize_and_pad

def make_raw_data(images_path: str, 
                  save_path: str,
                  bgr: bool = False,
                  image_preprocessing_function = resize_and_pad):
    """ Transform images to .raw data format.
    
    Parameters
    ----------
    images_path: str
        Path to where images are stored.
    save_path: str
        Path to where .raw data should be stored.
    image_preprocessing_function: Python function
    """

    files_raw = []
    img_save_path = os.path.join(save_path,"data")
    os.mkdir(img_save_path)
    for image_file_name in os.listdir(images_path):

        name, _ = os.path.splitext(image_file_name)

        # read image
        image = cv2.imread(os.path.join(images_path, image_file_name))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # preprocess to match model input
        image = np.array(image_preprocessing_function(image))

        image = image.astype(np.uint8) # for int8 quantized models
        
        # image = image.astype(np.uint16) # for int16 unquantized models
        # image = image.astype(np.float16) # for fp16 unquantized models
        img_raw = image.tobytes()
        # save as raw
        fn_raw = os.path.join(img_save_path, f"{name}.raw")

        with open(fn_raw, "wb") as f:
            f.write(img_raw)
        
        files_raw.append(os.path.join("test_raw_quant/data", f"{name}.raw"))
        # image.tofile(fn_raw)

    # save raw names txt file 
    with open(os.path.join(save_path,"inputs_raw.txt"), "w") as fp:
        fp.write("\n".join(files_raw))


if __name__ == "__main__":

    images_pth = "data/test/data"  # TODO: adjust
    save_pth = "data/test_raw/"  # TODO: adjust
    if os.path.exists(save_pth):
        shutil.rmtree(save_pth)
    os.mkdir(save_pth)

    make_raw_data(images_pth, save_pth, resize_and_pad)