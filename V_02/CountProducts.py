# import library
import cv2
from Modules import PublicModules as libs
import numpy as np

'''
    You can define variable final to here
    - Default: weight_name, config_name, classess_names
    important: please change it if not working correct
'''
# ---------- Name Dump JSON -------------------
URL_LOAD_VIDEO = 'URL_LOAD_VIDEO'
WEIGHT_NAME = 'WEIGHT_NAME'
CLASSES_NAMES = 'CLASSES_NAMES'
CONF_NAME = 'CONF_NAME'
CORLOR = 'CORLOR'
LEFT_REGION = 'LEFT_REGION'
RIGHT_REGION = 'RIGHT_REGION'
TOP_REGION = 'TOP_REGION'
BOTTOM_REGION = 'BOTTOM_REGION'
COPYRIGHT = 'COPYRIGHT'
COPYRIGHT_COLOR = 'COPYRIGHT_COLOR'
COPYRIGHT_FONT_SIZE = 'COPYRIGHT_FONT_SIZE'
# ---------------------------------------------

'''
    class detection logo using yolo_v3
    author: Viet-Saclo
    - Read weight set config and detect object in classess.
'''
class CountProducts:

    # Initial Contractor
    def __init__(self, CONFIG):
        # Define color to put text
        self.COLORS = CONFIG[CORLOR]
        self.URL_LOAD_VIDEO = CONFIG[URL_LOAD_VIDEO]
        self.BOX_WIDTH = -1
        self.flow_distance = -1
        self.COPYRIGHT = CONFIG[COPYRIGHT]
        self.COPYRIGHT_COLOR = CONFIG[COPYRIGHT_COLOR]
        self.COPYRIGHT_FONT_SIZE = CONFIG[COPYRIGHT_FONT_SIZE]

        # count pros
        self.currentPredict = -1
        self.countPros = [0, 0, 0]

        self.__fun_set_weight_conf_classes(
            weight_name= CONFIG[WEIGHT_NAME],
            conf_name= CONFIG[CONF_NAME],
            classes_names= CONFIG[CLASSES_NAMES]
        )
        self.__fun_set_LEFT_RIGHT_TOP_BOTTOM_REGION(
            left_region= CONFIG[LEFT_REGION],
            right_region= CONFIG[RIGHT_REGION],
            top_region= CONFIG[TOP_REGION],
            bottom_region= CONFIG[BOTTOM_REGION]
        )

        self.__fun_initial_CountProduct()

    # Set it if you want
    def __fun_set_weight_conf_classes(self, weight_name, conf_name, classes_names):
        self.weight_name = weight_name
        self.conf_name = conf_name
        self.classes_names = classes_names
    
    # Set it if you want
    def __fun_set_LEFT_RIGHT_TOP_BOTTOM_REGION(self, left_region, right_region, top_region, bottom_region):
        self.left_region = left_region
        self.right_region = right_region
        self.top_region = top_region
        self.bottom_region = bottom_region

    def fun_get_weight_conf_classes(self, ):
        return [
            self.weight_name,
            self.conf_name,
            self.classes_names
        ]

    '''
        using it to load classess names
        and load weight
        and define color to put text
    '''
    def __fun_initial_CountProduct(self, ):
        # Load classes names
        self.classes = None
        with open(self.classes_names, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Load weight name
        self.net = cv2.dnn.readNet(self.weight_name, self.conf_name)


    def __get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1]
                         for i in net.getUnconnectedOutLayers()]
        return output_layers

    '''
        functon draw into image a rectangle after detected
    '''
    def __draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        if type(class_id) is str:
            label = class_id
        else:
            label = str(self.classes[class_id])
        color = self.COLORS
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 1)
        cv2.putText(img, label, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    '''
    Hàm nhận diện đối tượng
    @param: sourceImage là nguồn hình ảnh, có thể là một đường dẫn hình hoặc một hình
        đã được đọc lên bằng OpenCV
    @param: classesName là phân lớp cần lưu trữ, classesName= 0 tức là người.
        chi tiết từng classes xem tại File yolov3.txt

    @return: một danh sách các hình ảnh con, là các hình ảnh người đã được nhận dạng.
    '''
    def __fun_DetectObject(self, sourceImage, classesName=0, isShowDetectionFull: bool = False):
        image = None
        width = None
        height = None
        scale = 0.00392
        if type(sourceImage) is str:
            try:
                image = cv2.imread(sourceImage)
            except:
                print('Path sourceImage non valid!')
                return
        else:
            image = sourceImage

        try:
            width = image.shape[1]
            height = image.shape[0]
        except:
            print('sourceIamge non valid!')
            return

        blob = cv2.dnn.blobFromImage(
            image, scale, (416, 416), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)

        outs = self.net.forward(self.__get_output_layers(self.net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, conf_threshold, nms_threshold)

        index = 0
        imgOriganal = image.copy()
        imgsGet = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.__draw_prediction(image, class_ids[i], confidences[i], round(
                x), round(y), round(x + w), round(y + h))
            if class_ids[i] == classesName or True:
                y = int(y)
                yh = int(y + h)
                x = int(x)
                xw = int(x + w)
                img = imgOriganal[y:yh, x:xw]
                imgsGet.append([img, [y, yh, x, xw]])
                self.currentPredict = class_ids[i]
            index += 1

        if isShowDetectionFull:
            cv2.imshow('ff', image)
            cv2.waitKey()
        return image, imgsGet

    '''
        Detect videos are very slower
        Select number of frame you want to skip
        Defaut I skip 5 frame for each detection
    '''
    def __fun_skip_frame(self, cap, count: int = 5):
        while count > -1:
            cap.read()
            count -= 1

    # def fun_drawRegion(self, image):
    #     # right region
    #     image[self.top_region:self.bottom_region, self.right_region:self.right_region + THIN_REGION] = CORLOR

    #     # bottom region
    #     image[self.bottom_region:self.bottom_region + THIN_REGION, self.left_region:self.right_region] = CORLOR

    #     # left region
    #     image[self.top_region:self.bottom_region, self.left_region:self.left_region + THIN_REGION] = CORLOR

    #     # top region
    #     image[self.top_region:self.top_region + THIN_REGION, self.left_region:self.right_region] = CORLOR

    '''
        Detect logo with a video
        @param: reduce_size: float: select 1 if you want keep original size, 0.5 if you want haft part size, 0.2, 0.7, ...
    '''
    def __fun_detect_logo_test_video(self, url: any = 0, reduce_size: float = 1, skip_frame: int= -1, frame_show_name: str= 'Logo_Detection', fps: int= 1):
        cap = cv2.VideoCapture(url)
        isContinue, frame = cap.read()
        while isContinue:
            # Reduce Size Image
            image = libs.fun_reduceSizeImage(frame, reduce_size)

            # Draw Region
            # self.fun_drawRegion(image)
            # cv2.rectangle(frame, (self.left_region, self.top_region), (self.right_region, self.bottom_region), color, 2)

            # Detect Logo
            image, _ = self.fun_DetectObject(image)

            # show
            cv2.imshow(frame_show_name, image)

            # wait
            if cv2.waitKey(fps) & 0xFF == ord('q'):
                break

            # Skip frame ?
            self.fun_skip_frame(cap, skip_frame)

            # next frame
            isContinue, frame = cap.read()

    # def fun_isInside_and_isValid(self, imageGet: list):
    #     isCount = True
    #     isPass = False

    #     # truong hop khong detect duoc products
    #     if len(imageGet) == 0:
    #         return isCount, isPass

    #     # remove outsize box
    #     for product in imageGet:
    #         pro = product[1]
    #         # x, y so sanh
    #         if pro[0] >= self.top_region and pro[0] <= self.bottom_region and pro[2] >= self.left_region and pro[2] <= self.right_region:
    #             continue
    #         imageGet.remove(product)

    #     # truong hop da remove sach products
    #     if len(imageGet) == 0:
    #         return isCount, isPass
    #     else:
    #         isCount = True

    #     ''' 
    #         Neu detect ra nhieu hon 1 product tren chuoi day chuyen
    #         B1: thuc hien chon ra 1 product nam trong (box) co (x, y) la lon nhat 
    #         B2: product can vach MARGIN_END
    #     '''
    #     # Buoc 1 tim max (x, y)
    #     product_max = imageGet[0][1]
    #     for i in range(1, len(imageGet)):
    #         pro = imageGet[i][1]
    #         # y, yh, x, xw (pro[3] ~ xw)
    #         if pro[3] >= product_max[3] + MARGIN_END:
    #             product_max = pro
        
    #     # Buoc 2 product can vach MARGIN_END ?
    #     if product_max[3] >= self.right_region - MARGIN_END:
    #         self.last_pass = self.right_region - MARGIN_END
    #         isPass = True
        
    #     # Final Check ??

    #     return isCount, isPass

    def __fun_update_BOX_WIDTH(self, pro):
        if self.BOX_WIDTH == -1:
            self.BOX_WIDTH = pro[3] - pro[2]
        else:
            width = pro[3] - pro[2]
            self.BOX_WIDTH = (self.BOX_WIDTH + width) / 2

    def __fun_flow_product(self, imageGet: list):
        # remove outsize box
        imageGet_tmp = []
        for product in imageGet:
            pro = product[1]
            # x, y so sanh
            if pro[0] >= self.top_region and pro[0] <= self.bottom_region and pro[2] >= self.left_region and pro[2] <= self.right_region:
                self.__fun_update_BOX_WIDTH(pro)
                imageGet_tmp.append(product.copy())
        
        imageGet = imageGet_tmp

        # truong hop da remove sach products
        if len(imageGet) == 0:
            return False

        if self.flow_distance == -1:
            pro = imageGet[0][1]
            self.flow_distance = pro[2]
            return True
        
        if len(imageGet) == 1:
            pro = imageGet[0][1]
            if self.flow_distance - pro[2] >= self.BOX_WIDTH:
                self.flow_distance = pro[2]
                return True
            if pro[2] - self.flow_distance <= self.BOX_WIDTH:
                self.flow_distance = pro[2]
        
        if len(imageGet) > 1:
            # tim distance nho nhat va cap nhat flow_distance
            product_min = imageGet[0][1]
            for i in range(1, len(imageGet)):
                pro = imageGet[i][1]
                # y, yh, x, xw (pro[3] ~ xw)
                if pro[2] < product_min[2]:
                    product_min = pro
            
            print('x-min: ', product_min[2])
            print('flow_distance: ', self.flow_distance)
            print('BOX_WIDTH: ', self.BOX_WIDTH)
            print('------------------')
            if self.flow_distance - product_min[2] >= self.BOX_WIDTH:
                self.flow_distance = product_min[2]
                return True
            self.flow_distance = product_min[2]
        return False

    '''
        Detect logo with a video
        @param: reduce_size: float: select 1 if you want keep original size, 0.5 if you want haft part size, 0.2, 0.7, ...
    '''
    def fun_startVideoAndCountObject(self, reduce_size: float = 1, skip_frame: int= -1, frame_show_name: str= 'Logo_Detection', fps: int= 1, pathSave: str= None):
        cap = cv2.VideoCapture(self.URL_LOAD_VIDEO)
        isContinue, frame = cap.read()
        if not isContinue:
            return
        wr = None
        image = libs.fun_reduceSizeImage(frame, reduce_size)
        h, w, _ = image.shape
        if pathSave is not None:
            wr = cv2.VideoWriter(pathSave, cv2.VideoWriter_fourcc(*'MJPG'), 30, (w, h))
        while isContinue:
            # Reduce Size Image
            image = libs.fun_reduceSizeImage(frame, reduce_size)

            # Draw Region
            # self.fun_drawRegion(image)
            cv2.rectangle(image, (self.left_region, self.top_region), (self.right_region, self.bottom_region), self.COLORS, 2)

            # Detect Logo
            image, imageGet = self.__fun_DetectObject(image)

            # is Inside Box
            isNext = self.__fun_flow_product(imageGet)
            if isNext and self.currentPredict != -1:
                self.countPros[self.currentPredict] += 1

            # show
            # image[25:65, :] *= 0
            cv2.putText(image, 'Product Count: ' + str(self.countPros[0]), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)

            cv2.putText(image, 'Product Count: ' + str(self.countPros[1]), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [13, 213, 228], 1)

            cv2.putText(image, 'Product Count: ' + str(self.countPros[2]), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [183, 183, 183], 1)
                    
            cv2.putText(image, self.COPYRIGHT, (10, h - 10), cv2.FONT_HERSHEY_COMPLEX, self.COPYRIGHT_FONT_SIZE, self.COPYRIGHT_COLOR, 1)
            cv2.imshow(frame_show_name, image)
            if pathSave is not None:
                wr.write(image)

            # wait
            if cv2.waitKey(fps) & 0xFF == ord('q'):
                break

            # Skip frame ?
            self.__fun_skip_frame(cap, skip_frame)

            # next frame
            isContinue, frame = cap.read()
