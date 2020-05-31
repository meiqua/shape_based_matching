import cv2
import numpy
import shape_based_matching_py

prefix = "/home/meiqua/shape_based_matching/test/"

def angle_test(mode, use_rot):
    detector = shape_based_matching_py.Detector(128, [4, 8])
    if mode != 'test':
        img = cv2.imread(prefix+"case1/train.png")
        # print(img.shape)

        # order of ny is row col
        img = img[110:380, 130:400]
        mask = numpy.ones((img.shape[0], img.shape[1]), numpy.uint8)
        mask *= 255

        padding = 100
        padded_img = numpy.zeros((img.shape[0]+2*padding, 
            img.shape[1]+2*padding, img.shape[2]), numpy.uint8)
        padded_mask = numpy.zeros((padded_img.shape[0], padded_img.shape[1]), numpy.uint8)

        padded_img[padding:padded_img.shape[0]-padding, padding:padded_img.shape[1]-padding, :] = \
            img[:, :, :]
        padded_mask[padding:padded_img.shape[0]-padding, padding:padded_img.shape[1]-padding] = \
            mask[:, :]
        # cv2.imshow("padded_img", padded_img)
        # cv2.imshow("padded_mask", padded_mask)
        # cv2.waitKey()

        shapes = shape_based_matching_py.shapeInfo_producer(padded_img, padded_mask)
        shapes.angle_range = [0, 360]
        shapes.angle_step = 1
        shapes.scale_range = [1]
        shapes.produce_infos()
        
        infos_have_templ = []
        class_id = "test"
        is_first = True
        first_id = 0
        first_angle = 0

        for info in shapes.infos:
            to_show = shapes.src_of(info)

            templ_id = 0
            if is_first:
                templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info))
                first_id = templ_id
                first_angle = info.angle

                if use_rot:
                    is_first = False
            else:
                templ_id = detector.addTemplate_rotate(class_id, first_id,
                                                       info.angle-first_angle,
                    shape_based_matching_py.CV_Point2f(padded_img.shape[1]/2.0, padded_img.shape[0]/2.0))
            templ = detector.getTemplates(class_id, templ_id)
            for feat in templ[0].features:
                to_show = cv2.circle(to_show, (feat.x+templ[0].tl_x, feat.y+templ[0].tl_y), 3, (0, 0, 255), -1)
            cv2.imshow("show templ", to_show)
            cv2.waitKey(1)
            if templ_id != -1:
                infos_have_templ.append(info)

        detector.writeClasses(prefix+"case1/%s_templ.yaml")
        shapes.save_infos(infos_have_templ, prefix + "case1/test_info.yaml")
    else:
        ids = []
        ids.append('test')
        detector.readClasses(ids, prefix+"case1/%s_templ.yaml")

        producer = shape_based_matching_py.shapeInfo_producer()
        infos = producer.load_infos(prefix + "case1/test_info.yaml")
        test_img = cv2.imread(prefix+"case1/test.png")
        padding = 250
        padded_img = numpy.zeros((test_img.shape[0]+2*padding, 
            test_img.shape[1]+2*padding, test_img.shape[2]), numpy.uint8)
        padded_img[padding:padded_img.shape[0]-padding, padding:padded_img.shape[1]-padding, :] = \
            test_img[:, :, :]

        stride = 16
        img_rows = int(padded_img.shape[0] / stride) * stride
        img_cols = int(padded_img.shape[1] / stride) * stride
        img = numpy.zeros((img_rows, img_cols, padded_img.shape[2]), numpy.uint8)
        img[:, :, :] = padded_img[0:img_rows, 0:img_cols, :]
        matches = detector.match(img, 90, ids)
        top5 = 1
        if top5 > len(matches):
            top5 = 1
        for i in range(top5):
            match = matches[i]
            templ = detector.getTemplates("test", match.template_id)
            # r_scaled = 270/2.0*infos[match.template_id].scale
            # train_img_half_width = 270/2.0 + 100
            # train_img_half_height = 270/2.0 + 100
            # x =  match.x - templ[0].tl_x + train_img_half_width
            # y =  match.y - templ[0].tl_y + train_img_half_height
            for feat in templ[0].features:
                img = cv2.circle(img, (feat.x+match.x, feat.y+match.y), 3, (0, 0, 255), -1)

            # cv2 have no RotatedRect constructor?
            print('match.template_id: {}'.format(match.template_id))
            print('match.similarity: {}'.format(match.similarity))
        cv2.imshow("img", img)
        cv2.waitKey(0)

if __name__ == "__main__":
    angle_test('test', True)