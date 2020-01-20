import cv2 as cv
import numpy as np
from keras import backend as K
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from PIL import Image
from pdb import set_trace
from timeit import default_timer as timer
import colorsys

class ObjectTrackingTest(object):
    def __init__(self):
        # If the recording is from an iPhone, image cropping will be different
        self.crop_style = "ipad_hd"
        self.bounding_box_model_image_size = (416, 416)
        self.current_detected_objects = dict()
        self.track_classes = [] # bounding boxes for these classes will be tracked with advanced CV features

    # Crops the frame to hide the UI
    # of the recorded app
    def crop_frame(self, frame):
        height = frame.shape[0]
        width = frame.shape[1]

        if self.crop_style == "ipad_sd":
            # Hide the 'B' left side indicator
            cv.circle(frame, (410,310), 8, (0,0,0), thickness=50, lineType=-1)
            return frame[280:height-300, 0:width-100]
        else:
            # Hide the 'B' left side indicator
            cv.circle(frame, (530,310), 10, (0,0,0), thickness=50, lineType=-1)
            return frame[280:height-300, 0:width-100]

    def generate_colors(self):
        hsv_tuples = [(x / len(self.box_detection_class_names), 1., 1.)
                              for x in range(len(self.box_detection_class_names))]
        self.bounding_box_model_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.bounding_box_model_colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.bounding_box_model_colors))

    def yolo_eval(self, yolo_outputs,
                  anchors,
                  num_classes,
                  image_shape,
                  max_boxes=20,
                  score_threshold=.6,
                  iou_threshold=.5):
        """Evaluate YOLO model on given input and return filtered boxes."""
        num_layers = len(yolo_outputs)
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
        input_shape = K.shape(yolo_outputs[0])[1:3] * 32
        boxes = []
        box_scores = []
        for l in range(num_layers):
            _boxes, _box_scores = self.yolo_boxes_and_scores(yolo_outputs[l],
                anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = K.concatenate(boxes, axis=0)
        box_scores = K.concatenate(box_scores, axis=0)

        mask = box_scores >= score_threshold
        max_boxes_tensor = K.constant(max_boxes, dtype='int32')
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(num_classes):
            # TODO: use keras backend instead of tf.
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
            class_boxes = K.gather(class_boxes, nms_index)
            class_box_scores = K.gather(class_box_scores, nms_index)
            classes = K.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = K.concatenate(boxes_, axis=0)
        scores_ = K.concatenate(scores_, axis=0)
        classes_ = K.concatenate(classes_, axis=0)

        return boxes_, scores_, classes_

    def yolo_head(self, feats, anchors, num_classes, input_shape, calc_loss=False):
        """Convert final layer features to bounding box parameters."""
        num_anchors = len(anchors)
        # Reshape to batch, height, width, num_anchors, box_params.
        anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

        grid_shape = K.shape(feats)[1:3] # height, width
        grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
            [1, grid_shape[1], 1, 1])
        grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
            [grid_shape[0], 1, 1, 1])
        grid = K.concatenate([grid_x, grid_y])
        grid = K.cast(grid, K.dtype(feats))

        feats = K.reshape(
            feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

        # Adjust preditions to each spatial grid point and anchor size.
        box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
        box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
        box_confidence = K.sigmoid(feats[..., 4:5])
        box_class_probs = K.sigmoid(feats[..., 5:])

        if calc_loss == True:
            return grid, feats, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs

    def yolo_boxes_and_scores(self, feats, anchors, num_classes, input_shape, image_shape):
        '''Process Conv layer output'''
        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(feats,
            anchors, num_classes, input_shape)
        boxes = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = K.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = K.reshape(box_scores, [-1, num_classes])
        return boxes, box_scores


    def detect_image(self, image):
        if self.bounding_box_model_image_size != (None, None):
            assert self.bounding_box_model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.bounding_box_model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = self.letterbox_image(image, tuple(reversed(self.bounding_box_model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = self.letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        # print([image.size, image.size])

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.box_detection_boxes, self.box_detection_scores, self.box_detection_classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        return (out_boxes, out_scores, out_classes)

    def letterbox_image(self, image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        '''Get corrected boxes'''
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = K.cast(input_shape, K.dtype(box_yx))
        image_shape = K.cast(image_shape, K.dtype(box_yx))
        new_shape = K.round(image_shape * K.min(input_shape/image_shape))
        offset = (input_shape-new_shape)/2./input_shape
        scale = input_shape/new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes =  K.concatenate([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ])

        # Scale boxes back to original image shape.
        boxes *= K.concatenate([image_shape, image_shape])
        return boxes

    def draw_and_grab_frames_bounding_boxes(self, frame, image, out_boxes, out_scores, out_classes):
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        self.current_detected_objects.clear()

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.box_detection_class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            # Draw the bounding box
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))

            # Extract the frame with the object
            if predicted_class not in self.current_detected_objects:
                self.current_detected_objects[predicted_class] = list()

            self.current_detected_objects[predicted_class].append(frame[top:bottom,left:right])

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.bounding_box_model_colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.bounding_box_model_colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)

            del draw

        # Debug output of collected current_detected_objects
        # for k in self.current_detected_objects:
        #     print("Detected %d boxes for class %s" % (len(self.current_detected_objects[k]), k))

        return image