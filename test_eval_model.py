import argparse
import test
import eval
from datasets.dataset_dota import DOTA
from models import DTDNet
from utils import decoder
import os


def parse_args():
    parser = argparse.ArgumentParser(description='DTDNet Model')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of batch size')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers')
    parser.add_argument('--init_lr', type=float, default=1.25e-5, help='Initial learning rate')
    parser.add_argument('--input_h', type=int, default=900, help='Resized image height')
    parser.add_argument('--input_w', type=int, default=900, help='Resized image width')
    parser.add_argument('--K', type=int, default=500, help='Maximum of objects')
    parser.add_argument('--conf_thresh', type=float, default=0.07, help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--ngpus', type=int, default=1, help='Number of gpus, ngpus>1 for multigpu')
    parser.add_argument('--resume_train', type=str, default='', help='Weights resumed in training')
    parser.add_argument('--resume', type=str, default='/home/nripendra/weights_dota_without_atten/model_150.pth', help='Weights resumed in testing and evaluation')
    parser.add_argument('--dataset', type=str, default='dota', help='Name of dataset')
    parser.add_argument('--data_dir', type=str, default='/home/nripendra/Multiple_treatment&Diagnosis/Experiment/DATA/Eval_matrix_data/', help='Data directory')
    parser.add_argument('--phase', type=str, default='test', help='Phase choice= {test, eval}')
    parser.add_argument('--wh_channels', type=int, default=8, help='Number of channels for the vectors (4x2)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = {'dota': DOTA}
    num_classes = {'dota': 1}
    heads = {'hm': num_classes[args.dataset],
             'wh': 10,
             'reg': 2,
             'cls_theta': 1
             }
    down_ratio = 4
    model = DTDNet.Teeth_detection(heads=heads,
                              pretrained=True,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              head_conv=256)

    decoder = decoder.DecDecoder(K=args.K,
                                 conf_thresh=args.conf_thresh,
                                 num_classes=num_classes[args.dataset])

    if args.phase == 'test':
        ctrbox_obj = test.TestModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
        ctrbox_obj.test(args, down_ratio=down_ratio)
    else:
        ctrbox_obj = eval.EvalModule(dataset=dataset, num_classes=num_classes, model=model, decoder=decoder)
        ctrbox_obj.evaluation(args, down_ratio=down_ratio)
