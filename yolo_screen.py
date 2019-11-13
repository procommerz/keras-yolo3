import sys
import argparse
import numpy as np
from yolo import YOLO, timer
from functools import reduce
from PIL import Image
import pdb
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str, default="checkpoints/trained_weights_final.h5",
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str, default="model_data/yolo_anchors.txt",
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str, default="model_data/training-usbase.names",
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./training-usbase/thyroid_inna_right_v2.mp4',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

class NebrusScreenDemo(object):
    def __init__(self, yolo):
        self.yolo = yolo
        pass

    def run(self):
        pass

    def detect_img(self, yolo):
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
        yolo.close_session()

    def detect_video(self, yolo, video_path, output_path=""):
        import cv2
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != "" else False
        if isOutput:
            print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            return_value, frame = vid.read()
            image = Image.fromarray(frame)
            image = yolo.detect_image(image)
            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        yolo.close_session()

# if FLAGS.image:
#     """
#     Image detection mode, disregard any remaining command line arguments
#     """
#     print("Image detection mode")
#     if "input" in FLAGS:
#         print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
#     detect_img(YOLO(**vars(FLAGS)))
# elif "input" in FLAGS:
#     detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
# else:
#     print("Must specify at least video_input_path.  See usage with --help.")

app = None

if __name__ == '__main__':
    app = NebrusScreenDemo()
    app.run()