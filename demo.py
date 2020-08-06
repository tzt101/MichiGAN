import os
import sys
import cv2
from ui.ui4 import Ui_Form
from ui.mouse_event import GraphicsScene
from ui_util.config import Config
#
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

from collections import OrderedDict

import data
from options.demo_options import DemoOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util.util import tensor2im, tensor2label
from util import html
from data.base_dataset import demo_inference_dataLoad
from PIL import Image
import torch
import time
import math
import numpy as np
from scipy.misc import imresize
from ui_util import cal_orient_stroke

color_list = [QColor(0, 0, 0), QColor(255, 255, 255), QColor(76, 153, 0), QColor(204, 204, 0), QColor(51, 51, 255), QColor(204, 0, 204), QColor(0, 255, 255), QColor(51, 255, 255), QColor(102, 51, 0), QColor(255, 0, 0), QColor(102, 204, 0), QColor(255, 255, 0), QColor(0, 0, 153), QColor(0, 0, 204), QColor(255, 51, 153), QColor(0, 204, 204), QColor(0, 51, 0), QColor(255, 153, 51), QColor(0, 204, 0)]

def dliate_erode(img, kernel):
    er_k = kernel
    di_k = kernel
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(er_k, er_k))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(di_k, di_k))
    img_f = cv2.dilate(img, dilate_kernel)
    img_f = cv2.erode(img_f, erode_kernel)

    return img_f

class Ex(QWidget, Ui_Form):
    def __init__(self, model, opt):
        super(Ex, self).__init__()
        self.setupUi(self)
        self.show()
        self.model = model
        self.opt = opt
        self.img_size = 512
        self.root_dir = opt.demo_data_dir
        self.save_dir = opt.results_dir

        self.output_img = None

        self.mat_img = None

        self.mode = 0
        self.size = 6
        self.mask = None
        self.mask_m = None
        self.tag_img = None
        self.recon_tag_img = None
        self.ref_img = None
        self.ref_mask_path = None
        self.orient = None
        self.orient_m = None
        self.orient_mask = None
        self.orient_image = None
        self.mask_hole = None
        self.mask_stroke = None
        self.orient_stroke = None
        self.save_datas = {}

        self.mouse_clicked = False
        self.scene = GraphicsScene(self.mode, self.size)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.orient_scene = GraphicsScene(self.mode, self.size)
        self.graphicsView_2.setScene(self.orient_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.ref_scene = QGraphicsScene()
        self.graphicsView_5.setScene(self.ref_scene)
        self.graphicsView_5.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_5.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_5.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.result_scene = QGraphicsScene()
        self.graphicsView_3.setScene(self.result_scene)
        self.graphicsView_3.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.tag_scene = QGraphicsScene()
        self.graphicsView_4.setScene(self.tag_scene)
        self.graphicsView_4.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_4.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_4.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.dlg = QColorDialog(self.graphicsView)
        self.color = None

    def open_ref(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", self.opt.demo_data_dir)
        if fileName:
            image_name = fileName.split('/')[-1]
            image = QPixmap(fileName)
            mat_img = Image.open(fileName)
            self.ref_img = mat_img.copy()
            self.ref_mask_path = os.path.join(self.root_dir,'labels',image_name[:-4]+'.png')

            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return
            image = image.scaled(self.graphicsView_5.size(), Qt.IgnoreAspectRatio)

            if len(self.ref_scene.items()) > 0:
                self.ref_scene.removeItem(self.ref_scene.items()[-1])
            self.ref_scene.addPixmap(image)
            # if len(self.result_scene.items()) > 0:
            #     self.result_scene.removeItem(self.result_scene.items()[-1])
            # self.result_scene.addPixmap(image)

    def open_tag(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", self.opt.demo_data_dir)
        if fileName:
            image_name = fileName.split('/')[-1]
            image = QPixmap(fileName)
            mat_img = Image.open(fileName)

            recon_dir = os.path.join(self.opt.demo_data_dir,'images_recon', image_name)
            if os.path.exists(recon_dir):
                recon_img = Image.open(recon_dir)
                self.recon_tag_img = recon_img.copy()
            else:
                self.recon_tag_img = None

            self.tag_img = mat_img.copy()
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return
            image1 = image.scaled(self.graphicsView_4.size(), Qt.IgnoreAspectRatio)

            if len(self.tag_scene.items()) > 0:
                self.tag_scene.removeItem(self.tag_scene.items()[-1])
            self.tag_scene.addPixmap(image1)
            if len(self.result_scene.items()) > 0:
                self.result_scene.removeItem(self.result_scene.items()[-1])
            image2 = image.scaled(self.graphicsView_3.size(), Qt.IgnoreAspectRatio)
            self.result_scene.addPixmap(image2)

            # process mask and orient by default
            mat_mask = cv2.imread(os.path.join(self.root_dir,'labels', image_name[:-4]+'.png'))

            self.mask = mat_mask.copy() # original mask
            self.mask_m = mat_mask # edited mask
            mat_mask = mat_mask.copy()
            mask= QImage(mat_mask, self.img_size, self.img_size, QImage.Format_RGB888)

            if mask.isNull():
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return

            for i in range(self.img_size):
                for j in range(self.img_size):
                    r, g, b, a = mask.pixelColor(i, j).getRgb()
                    mask.setPixel(i, j, color_list[r].rgb())

            pixmap = QPixmap()
            pixmap.convertFromImage(mask)
            self.mask_show = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            self.scene.reset()
            if len(self.scene.items()) > 0:
                self.scene.reset_items()
            self.scene.addPixmap(self.mask_show)

            # for orient
            mat_img = cv2.imread(os.path.join(self.root_dir,'orients',image_name[:-4]+'_orient_dense.png'), cv2.IMREAD_GRAYSCALE)
            orient_mask = cv2.imread(os.path.join(self.root_dir, 'labels',image_name[:-4]+'.png'), cv2.IMREAD_GRAYSCALE)
            self.orient_image = Image.open(os.path.join(self.root_dir, 'images',image_name[:-4]+'.jpg'))

            self.orient = mat_img.copy()
            self.orient_m = mat_img
            mat_img = mat_img.copy()
            self.orient_mask = orient_mask.copy()
            orient = mat_img / 255.0 * math.pi
            H, W = orient.shape
            orient_rgb = np.zeros((H, W, 3))
            orient_rgb[..., 1] = (np.sin(2 * orient) + 1) / 2
            orient_rgb[..., 0] = (np.cos(2 * orient) + 1) / 2
            orient_rgb[..., 2] = 0.5
            orient_rgb *= orient_mask[..., np.newaxis]
            orient_rgb = np.uint8(orient_rgb * 255.0)
            image = QImage(orient_rgb, self.img_size, self.img_size,self.img_size*3, QImage.Format_RGB888)

            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return

            pixmap = QPixmap()
            pixmap.convertFromImage(image)
            self.orient_show = pixmap.scaled(self.graphicsView_2.size(), Qt.IgnoreAspectRatio)
            self.orient_scene.reset()
            if len(self.orient_scene.items()) > 0:
                self.orient_scene.reset_items()
            self.orient_scene.addPixmap(self.orient_show)


    def open_orient(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",self.opt.demo_data_dir)
        if fileName:
            image_name = fileName.split('/')[-1]
            mat_img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            orient_mask = cv2.imread(os.path.join(self.root_dir, 'labels',image_name[:-17]+'.png'), cv2.IMREAD_GRAYSCALE)
            self.orient_image = Image.open(os.path.join(self.root_dir, 'images', image_name[:-17] + '.jpg'))
            # mat_img = imresize(mat_img, (self.img_size, self.img_size), interp='nearest')

            # mat_img = Image.open(fileName)
            # mat_img = np.array(mat_img.resize((self.size,self.size)))

            self.orient = mat_img.copy()
            self.orient_m = mat_img
            mat_img = mat_img.copy()
            # transfor to RGB
            # orient_mask = mat_img.copy()
            # orient_mask[orient_mask > 0] = 1
            # orient_mask = dliate_erode(orient_mask, 10)
            self.orient_mask = orient_mask.copy()
            orient = mat_img / 255.0 * math.pi
            H, W = orient.shape
            orient_rgb = np.zeros((H, W, 3))
            orient_rgb[..., 1] = (np.sin(2 * orient) + 1) / 2
            orient_rgb[..., 0] = (np.cos(2 * orient) + 1) / 2
            orient_rgb[..., 2] = 0.5
            orient_rgb *= orient_mask[..., np.newaxis]
            orient_rgb = np.uint8(orient_rgb * 255.0)
            # orient_save = Image.fromarray(np.uint8(orient_rgb)).convert('RGB')
            # orient_save.save('./inference_samples/original_orient.png')
            image = QImage(orient_rgb, self.img_size, self.img_size,self.img_size*3, QImage.Format_RGB888)

            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return

            pixmap = QPixmap()
            pixmap.convertFromImage(image)
            self.orient_show = pixmap.scaled(self.graphicsView_2.size(), Qt.IgnoreAspectRatio)
            self.orient_scene.reset()
            if len(self.orient_scene.items()) > 0:
                self.orient_scene.reset_items()
            self.orient_scene.addPixmap(self.orient_show)

    def open_mask(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",self.opt.demo_data_dir)
        if fileName:
            mat_img = cv2.imread(fileName)
            # mat_img = imresize(mat_img, (self.img_size, self.img_size), interp='nearest')

            # mat_img = Image.open(fileName)
            # mat_img = np.array(mat_img.resize((self.size,self.size)))

            self.mask = mat_img.copy() # original mask
            self.mask_m = mat_img # edited mask
            mat_img = mat_img.copy()
            image = QImage(mat_img, self.img_size, self.img_size, QImage.Format_RGB888)

            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return

            for i in range(self.img_size):
                for j in range(self.img_size):
                    r, g, b, a = image.pixelColor(i, j).getRgb()
                    image.setPixel(i, j, color_list[r].rgb())

            pixmap = QPixmap()
            pixmap.convertFromImage(image)
            self.mask_show = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
            self.scene.reset()
            if len(self.scene.items()) > 0:
                self.scene.reset_items()
            self.scene.addPixmap(self.mask_show)

    def bg_mode(self):
        self.scene.mode = 0

    def hair_mode(self):
        self.scene.mode = 1

    def increase(self):
        if self.scene.size < 15:
            self.scene.size += 1

    def decrease(self):
        if self.scene.size > 1:
            self.scene.size -= 1


    def edit(self):
        # get the edited mask
        self.mask_m = self.mask.copy()
        for i in range(2):
            self.mask_m = self.make_mask(self.mask_m, self.scene.mask_points[i], self.scene.size_points[i], i)

        # get the edited orient
        orient_new = self.mask_m.copy()
        orient_new = self.make_mask(orient_new, self.scene.mask_points[2], self.scene.size_points[2], 2)
        vis_stroke = orient_new.copy()
        orient_new[orient_new == 1] = 0
        orient_new[orient_new == 2] = 1
        mask_stroke = orient_new.copy()[:,:,0]
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50, 50))
        mask_hole = cv2.dilate(np.uint8(orient_new), dilate_kernel)[:,:,0]
        cal_stroke_orient = cal_orient_stroke.orient()
        orient_stroke = cal_stroke_orient.stroke_to_orient(mask_stroke)

        # process the tag image
        ranges = np.unique(self.mask - self.mask_m)
        if not self.clickButtion1.isChecked() and self.recon_tag_img is not None and 1 in ranges:
            tag_image = self.recon_tag_img.copy()
        else:
            tag_image = self.tag_img.copy()


        if self.clickButtion1.isChecked():
            # reference mask
            print('select Reference Mask')
            if self.clickButtion3.isChecked():
                # reference orient
                print('select Reference Orientation')
                self.model.opt.inpaint_mode = 'ref'
                data = demo_inference_dataLoad(self.opt, self.ref_mask_path, self.mask[:, :, 0], self.orient_mask.copy(), self.orient, self.ref_img, tag_image)
            else:
                print('select Edited Orientation')
                self.model.opt.inpaint_mode = 'stroke'
                data = demo_inference_dataLoad(self.opt, self.ref_mask_path, self.mask[:, :, 0],
                                               self.orient_mask.copy(), self.orient, self.ref_img, tag_image,orient_stroke, mask_stroke, mask_hole)
        else:
            # Edited mask
            print('select Edited Mask')
            if self.clickButtion3.isChecked():
                # reference orient
                print('select Reference Orientation')
                self.model.opt.inpaint_mode = 'ref'
                data = demo_inference_dataLoad(self.opt, self.ref_mask_path, self.mask_m[:,:,0], self.orient_mask.copy(), self.orient, self.ref_img, tag_image)
            else:
                print('select Edited Orientation')
                self.model.opt.inpaint_mode = 'stroke'
                data = demo_inference_dataLoad(self.opt, self.ref_mask_path, self.mask_m[:, :, 0],
                                               self.orient_mask.copy(), self.orient, self.ref_img, tag_image,orient_stroke, mask_stroke, mask_hole)

        start_t = time.time()
        generated, new_orient_rgb = self.model(data, mode='demo_inference')
        end_t = time.time()
        print('inference time : {}'.format(end_t - start_t))

        # # save_image((generated.data[0] + 1) / 2,'./results/1.jpg')
        # result = tensor2im(generated[0])
        # fake_image = Image.fromarray(result)
        # fake_image.save('./inference_samples/inpaint_fake_image.jpg')
        if self.opt.add_feat_zeros:
            th = self.opt.add_th
            tmp = generated[:,:,int(th/2):int(th/2)+self.opt.crop_size, int(th/2):int(th/2)+self.opt.crop_size]
            generated = tmp
        result = generated.permute(0, 2, 3, 1)
        result = result.cpu().numpy()
        result = (result + 1) * 127.5
        result = np.asarray(result[0,:,:,:], dtype=np.uint8)
        # update the self.save_datas
        self.save_datas['result'] = Image.fromarray(result.copy())
        self.save_datas['ref_img'] = self.ref_img.copy()
        self.save_datas['tag_img'] = self.tag_img.copy()
        self.save_datas['ori_img'] = self.orient_image.copy()
        vis_stroke[vis_stroke==1] = 255
        vis_stroke[vis_stroke == 2] = 127
        # save
        # stroke_save = Image.fromarray(np.uint8(vis_stroke))
        # stroke_save.save('inference_samples/stroke_mask.png')

        self.save_datas['stroke'] = Image.fromarray(np.uint8(vis_stroke))
        self.save_datas['mask'] = Image.fromarray(np.uint8(self.mask_m[:, :, 0].copy()*255))

        qim = QImage(result.data, result.shape[1], result.shape[0], result.shape[0] * 3, QImage.Format_RGB888)
        pixmap = QPixmap()
        pixmap.convertFromImage(qim)
        image = pixmap.scaled(self.graphicsView_3.size(), Qt.IgnoreAspectRatio)

        if len(self.result_scene.items()) > 0:
            self.result_scene.removeItem(self.result_scene.items()[-1])
            self.result_scene.addPixmap(image)

        # for orient
        H, W, C = new_orient_rgb.shape
        image = QImage(new_orient_rgb, H, W, W * 3, QImage.Format_RGB888)

        pixmap = QPixmap()
        pixmap.convertFromImage(image)
        image = pixmap.scaled(self.graphicsView_2.size(), Qt.IgnoreAspectRatio)

        if len(self.orient_scene.items()) > 0:
            self.orient_scene.removeItem(self.orient_scene.items()[-1])
        self.orient_scene.addPixmap(image)

    def save(self):
        # for save all results
        print('save..')

        fileName, _ = QFileDialog.getSaveFileName(self, "Save File",
                                                  self.save_dir)
        sum = Image.new(self.save_datas['result'].mode, (5 * self.opt.crop_size, self.opt.crop_size))
        sum.paste(self.save_datas['stroke'], box=(3*self.opt.crop_size, 0))
        # sum.paste(self.save_datas['mask'], box=(self.opt.crop_size, 0))
        sum.paste(self.save_datas['tag_img'], box=(0, 0))
        sum.paste(self.save_datas['ref_img'], box=(self.opt.crop_size,0))
        sum.paste(self.save_datas['ori_img'], box=(2*self.opt.crop_size,0))
        sum.paste(self.save_datas['result'], box=(4*self.opt.crop_size, 0))
        sum.save(fileName+'.jpg')


    def make_mask(self, mask, pts, sizes, color):
        if len(pts) > 0:
            for idx, pt in enumerate(pts):
                cv2.line(mask, pt['prev'], pt['curr'], (color, color, color), sizes[idx])
        return mask

    def save_img(self):
        if type(self.output_img):
            fileName, _ = QFileDialog.getSaveFileName(self, "Save File",
                                                      QDir.currentPath())
            cv2.imwrite(fileName + '.jpg', self.output_img)

    def undo(self):
        self.scene.undo()

    def clear(self):
        self.mask_m = self.mask.copy()

        self.scene.reset_items()
        self.scene.reset()
        if type(self.mask_show):
            self.scene.addPixmap(self.mask_show)

    def save_orient_edit(self, input, name):
        if np.max(input) > 1:
            img = Image.fromarray(np.uint8(input))
        else:
            img = Image.fromarray(np.uint8(input*255))
        img.save('./inference_samples/'+name)

    def orient_edit(self):
        # get the new mask
        for i in range(2):
            self.mask_m = self.make_mask(self.mask_m, self.scene.mask_points[i], self.scene.size_points[i], i)
        # get the edited orient
        orient_new = self.mask_m
        orient_new = self.make_mask(orient_new, self.scene.mask_points[2], self.scene.size_points[2], 2)
        orient_new[orient_new == 1] = 0
        orient_new[orient_new == 2] = 1
        # self.save_orient_edit(orient_new[...,0],'edited_orient.jpg')
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20, 20))
        orient_new = cv2.dilate(np.uint8(orient_new), dilate_kernel)
        print(np.unique(orient_new))
        # self.save_orient_edit(orient_new[...,0], 'Gauss_edited_orient.jpg')

        # image = QImage(orient_new, self.img_size, self.img_size, QImage.Format_RGB888)
        #
        # for i in range(self.img_size):
        #     for j in range(self.img_size):
        #         r, g, b, a = image.pixelColor(i, j).getRgb()
        #         image.setPixel(i, j, color_list[r].rgb())
        #
        # pixmap = QPixmap()
        # pixmap.convertFromImage(image)
        # mask_show = pixmap.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)
        # self.orient_scene.reset_items()
        # self.orient_scene.reset()
        # if type(mask_show):
        #     self.orient_scene.addPixmap(mask_show)

    def orient_mode(self):
        # self.orient_scene.mode = 1
        self.scene.mode = 2
        # self.orient_scene.size = 5

    def erase_mode(self):
        self.orient_scene.mode = 0
        self.orient_scene.size = 14

    def orient_increase(self):
        if self.scene.size < 15:
            self.scene.size += 1

    def orient_decrease(self):
        if self.scene.size > 1:
            self.scene.size -= 1

    def selectM(self):
        if self.clickButtion1.isChecked():
            print('select Reference Mask')
        elif self.clickButtion2.isChecked():
            print('select Edited Mask')

    def selectO(self):
        if self.clickButtion3.isChecked():
            print('select Reference Orient')
        elif self.clickButtion4.isChecked():
            print('select Edited Orient')

if __name__ == '__main__':
    opt = DemoOptions().parse()
    model = Pix2PixModel(opt)
    model.eval()
    app = QApplication(sys.argv)
    ex = Ex(model, opt)
    sys.exit(app.exec_())
