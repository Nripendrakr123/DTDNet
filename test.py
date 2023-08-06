import torch
import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt
from utils import func_utils
from torch import nn
from torchvision import transforms
import torch
from PIL import Image


class CustomConvNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()
        self.num_classes = num_classes
        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.gap = self.global_avg_pool(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, self.num_classes)
        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))


class Model_infrance:
    def __init__(self, model_path="model.pth"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        num_classes = 8
        self.class_names = ['Implant', 'ToothBud', 'Tooth', 'ImpTooth', 'RCT+CRW', 'I+CRW', 'Restoration', 'RCT']
        self.colors = [(30, 156, 69), (240, 219, 45), (240, 57, 27), (73, 250, 255), (67, 132, 245), (233, 83, 245),
                       (245, 15, 68), (219, 116, 222)]  # np.random.uniform(0, 255, size=(len(self.class_names), 3))
        self.model = CustomConvNet(num_classes=num_classes).to(self.device)
        self.model_PATH = model_path
        self.model.load_state_dict(torch.load(self.model_PATH, map_location=self.device))
        self.model.eval()
        self.transforms = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    def get_output(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transforms(image).unsqueeze(0)
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
        _, predicted = torch.max(outputs.data, 1)
        idx = predicted.data.cpu().numpy()[0]
        label_name = self.class_names[idx]
        return label_name, self.colors[idx]


def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


class TestModule(object):
    def __init__(self, dataset, num_classes, model, decoder):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.num_classes = num_classes
        self.model = model
        self.decoder = decoder
        self.classi_model = Model_infrance()
        self.out_dir = "./heat_map/"
        try:
            os.makedirs(self.out_dir)
        except Exception as e:
            print(e)

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        model.load_state_dict(state_dict_, strict=True)
        return model

    def map_mask_to_image(self, mask, img, color=None):
        if color is None:
            color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)

    def get_croped_image(self, pts, img):
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        x, y = x - 1, y - 1
        w, h = w, h
        croped = img[y:y + h, x:x + w].copy()
        return croped

    def imshow_heatmap(self, pr_dec, images, cnt):
        wh = pr_dec['wh']
        hm = pr_dec['hm']
        cls_theta = pr_dec['cls_theta']
        wh_w = wh[0, 0, :, :].data.cpu().numpy()
        wh_h = wh[0, 1, :, :].data.cpu().numpy()
        hm = hm[0, 0, :, :].data.cpu().numpy()
        cls_theta = cls_theta[0, 0, :, :].data.cpu().numpy()
        hm = cv2.resize(hm, (images.shape[1], images.shape[0]))
        plt.imshow(hm)
        plt.savefig(os.path.join(self.out_dir, "heat_" + str(cnt) + ".png"))

    def test(self, args, down_ratio):
        save_path = 'weights_' + args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=down_ratio)
        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)

        total_time = []
        output_image_path = "./image_output/"
        model_crop_output = "./crop_images/"
        if not os.path.exists(output_image_path):
            os.makedirs(output_image_path)

        if not os.path.exists(model_crop_output):
            os.makedirs(model_crop_output)

        n_crop = 0
        for cnt, data_dict in enumerate(data_loader):
            image = data_dict['image'][0].to(self.device)
            img_id = data_dict['img_id'][0]
            print('processing {}/{} image ...'.format(cnt, len(data_loader)))
            begin_time = time.time()
            with torch.no_grad():
                pr_decs = self.model(image)

            ori_image = dsets.load_image(cnt)
            ori_image_copy = ori_image.copy()
            height, width, _ = ori_image.shape
            self.imshow_heatmap(pr_decs, ori_image, cnt)
            torch.cuda.synchronize(self.device)
            decoded_pts = []
            decoded_scores = []
            predictions = self.decoder.ctdet_decode(pr_decs)
            pts0, scores0 = func_utils.decode_prediction(predictions, dsets, args, img_id, down_ratio)
            decoded_pts.append(pts0)
            decoded_scores.append(scores0)
            # nms
            results = {cat: [] for cat in dsets.category}
            for cat in dsets.category:
                if cat == 'background':
                    continue
                pts_cat = []
                scores_cat = []
                for pts0, scores0 in zip(decoded_pts, decoded_scores):
                    pts_cat.extend(pts0[cat])
                    scores_cat.extend(scores0[cat])
                pts_cat = np.asarray(pts_cat, np.float32)
                scores_cat = np.asarray(scores_cat, np.float32)
                if pts_cat.shape[0]:
                    nms_results = func_utils.non_maximum_suppression(pts_cat, scores_cat)
                    results[cat].extend(nms_results)

            end_time = time.time()
            total_time.append(end_time - begin_time)
            # nms
            for cat in dsets.category:
                if cat == 'background':
                    continue
                result = results[cat]
                for pred in result:
                    score = pred[-1]
                    tl = np.asarray([pred[0], pred[1]], np.float32)
                    tr = np.asarray([pred[2], pred[3]], np.float32)
                    br = np.asarray([pred[4], pred[5]], np.float32)
                    bl = np.asarray([pred[6], pred[7]], np.float32)

                    tt = (np.asarray(tl, np.float32) + np.asarray(tr, np.float32)) / 2
                    rr = (np.asarray(tr, np.float32) + np.asarray(br, np.float32)) / 2
                    bb = (np.asarray(bl, np.float32) + np.asarray(br, np.float32)) / 2
                    ll = (np.asarray(tl, np.float32) + np.asarray(bl, np.float32)) / 2

                    box = np.asarray([tl, tr, br, bl], np.float32)
                    crop_image = self.get_croped_image(box, ori_image_copy)
                    if crop_image.shape[0] <= 0 or crop_image.shape[1] <= 0:
                        continue
                    cv2.imwrite(os.path.join(model_crop_output, "image_" + str(n_crop) + ".png"), crop_image)
                    n_crop += 1
                    classes_name, colors = self.classi_model.get_output(crop_image)

          

                    ori_image = cv2.drawContours(ori_image, [np.int0(box)], 0, colors, 4, 1)
                    cv2.putText(ori_image, '{:.2f} {}'.format(score, classes_name), (int(box[1][0]), int(box[1][1])),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1, 1)
        total_time = total_time[1:]
        print('avg time is {}'.format(np.mean(total_time)))
        print('FPS is {}'.format(1. / np.mean(total_time)))
