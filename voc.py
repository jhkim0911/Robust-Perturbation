import time, random
import warnings
import torch.backends.cudnn as cudnn
import statistics

from os import listdir
from os.path import isfile, join
from eval.pointing_game import *
from eval.in_del import *
from loader.attribution_map import *
from loader.network import *
from loader.data_loader import *
from glob import glob
from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")

class VOC(object):
    def __init__(self, args):
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epoch = args.epoch
        self.resume = args.resume
        self.device = args.device
        self.print_freq = args.print_freq

        self.log_dir = args.log_dir
        self.xai_dir = args.xai_dir + '/' + self.model_name
        self.xai_data = args.xai_data
        self.result_dir = args.result_dir

        self.data_dir = args.data_dir
        self.dataset_dir = self.data_dir
        self.class_path = './voc.names'

        self.selectedFont = ImageFont.truetype(os.path.join('usr/share/fonts/', 'NanumGothic.ttf'), size=15)
        self.class_num = 20
        self.desired_size = 1150
        self.img_size = 300

        # visualization parameters
        self.num_cam = 20
        self.xai_batch = 1
        self.box_width = 3
        self.v_seed = 777

        print("########## Information ##########")
        print("# model   :", self.model_name)
        print("# dataset :", self.dataset)
        print("# b-size  :", self.batch_size)
        print("# t_epoch :", self.epoch)

    # We train model with multi-gpu, for single gpu modify DataParallel option or remove it.
    def build_model(self):
        if self.model_name == 'GoogLeNet':
            self.backbone_net = GoogLenet(self.class_num).to(self.device)

        elif self.model_name == 'ResNet':
            self.backbone_net = ResNet(self.class_num).to(self.device)

        elif self.model_name == 'VGG':
            self.backbone_net = VGG(self.class_num).to(self.device)

        self.backbone_net.apply(weights_init).cuda()
        self.backbone_net = torch.nn.DataParallel(self.backbone_net).cuda()

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.m = torch.nn.Sigmoid()

        self.optimizer = torch.optim.SGD(self.backbone_net.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-6, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 12, eta_min=0, last_epoch=-1)

    def train(self):
        log_dir = self.log_dir + '/' + self.model_dir
        check_dir(log_dir)

        self.writer = SummaryWriter(log_dir=log_dir)
        check_dir(self.result_dir + '/' + self.model_dir)

        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.model_dir, '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_epoch = int(model_list[-1].split('_')[-2])
                counter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.model_dir), start_epoch, counter)

                print("\n [*] Load SUCCESS")
                print(" [*] Training start from epoch %d / counter %d !\n" % (start_epoch, counter))

        else:
            print("\n [*] Load FAIL")
            print(" [*] Training start from Scratch !\n")
            counter = 0
            start_epoch = 1
            start_batch_id = 0

        train_dataloader = get_voc_data_loader(self.dataset_dir, self.batch_size, split='train')
        val_dataloader = get_voc_data_loader(self.dataset_dir, self.batch_size, split='val')

        if self.resume:
            start_batch_id = counter - len(train_dataloader) * (start_epoch - 1)

        start_time = time.time()
        self.best_val_map = 0.0

        # Each epoch has a training and validation phase
        for epoch in range(start_epoch, self.epoch + 1):
            self.scheduler.step(epoch)
            self.backbone_net.train()

            running_loss = 0.0
            running_ap = 0.0

            for i, data in enumerate(train_dataloader, start=start_batch_id):
                input_img, target = data

                # print(data)
                target = target.float()
                input_img, target = input_img.to(self.device), target.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                logits = self.backbone_net(input_img)
                loss = self.criterion(logits, target)

                running_loss += loss  # sum up batch loss
                running_ap += get_ap_score(torch.Tensor.cpu(target).detach().numpy(),
                                           torch.Tensor.cpu(self.m(logits)).detach().numpy())

                loss.backward()
                self.optimizer.step()

                counter += 1
                self.writer.add_scalar('Train/loss', loss, counter)
                self.writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], counter)

            num_samples = float(len(train_dataloader.dataset))

            tr_map_ = running_ap / num_samples
            tr_loss_ = running_loss / num_samples
            print(" Epoch: [%3d] [%4d/%4d] time: %4.2f, loss: %.4f, avg_precision: %.4f" % (epoch, i, len(train_dataloader), time.time() - start_time, tr_loss_, tr_map_))

            val_loss, val_map = self.acc_check_for_train(self.backbone_net, val_dataloader)

            if val_map > self.best_val_map:
                self.best_val_map = val_map
                self.save(os.path.join(self.result_dir, self.model_dir), epoch, counter)
                print(" [*] epoch %4d / counter %8d is saved, val loss %.4f, val precision %.4f" % (epoch, counter, val_loss, val_map))

    @property
    def model_dir(self):
        return "{}_{}".format(self.model_name, self.dataset)

    def save(self, dir, epoch, counter):
        params = {}
        params['backbone_net'] = self.backbone_net.module.state_dict()
        params['scheduler'] = self.scheduler.state_dict()
        params['optimizer'] = self.optimizer.state_dict()

        torch.save(params, os.path.join(dir, self.dataset + '_params_%07d_%07d.pt' % (epoch, counter)))

    def load(self, dir, epoch, counter):
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d_%07d.pt' % (epoch, counter)))
        self.backbone_net.module.load_state_dict(params['backbone_net'])
        self.scheduler.load_state_dict(params['scheduler'])
        self.optimizer.load_state_dict(params['optimizer'])

    def acc_check_for_train(self, model, val_dataloader):
        model.eval()
        running_loss = 0.0
        running_ap = 0.0

        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                input_img, target = data

                target = target.float()
                input_img, target = input_img.to(self.device), target.to(self.device)

                logits = self.backbone_net(input_img)
                loss = self.criterion(logits, target)
                running_loss += loss  # sum up batch loss
                running_ap += get_ap_score(torch.Tensor.cpu(target).detach().numpy(),
                                           torch.Tensor.cpu(self.m(logits)).detach().numpy())

            num_samples = float(len(val_dataloader.dataset))
            val_loss_ = running_loss.item() / num_samples
            val_map_ = running_ap / num_samples

        return val_loss_, val_map_

    def test(self):
        model = self.backbone_net.to(self.device)
        model_list = glob(os.path.join(self.result_dir, self.model_dir, '*.pt'))

        if not len(model_list) == 0:
            model_list.sort()
            start_epoch = int(model_list[-1].split('_')[-2])
            counter = int(model_list[-1].split('_')[-1].split('.')[0])

            self.load(os.path.join(self.result_dir, self.model_dir), start_epoch, counter)

            print("\n [*] Load SUCCESS")
            print(" [*] Params from epoch %d / counter %d LOADED !\n" % (start_epoch, counter))

        else:
            print("\n [*] Load FAIL")
            print(" [*] Train Model FIRST !\n")

        class_name = ['aeroplane', 'bicycle', 'bird', 'boat',
                      'bottle', 'bus', 'car', 'cat', 'chair',
                      'cow', 'diningtable', 'dog', 'horse',
                      'motorbike', 'person', 'pottedplant',
                      'sheep', 'sofa', 'train', 'tvmonitor']

        cudnn.benchmark = True
        model.eval()

        test_dataloader = get_voc_data_loader(self.dataset_dir, self.batch_size, split='test')

        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()

        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                input_img, target = data

                target = target.float().cuda()
                outGT = torch.cat((outGT, target), 0)
                bs, n_crops, c, h, w = input_img.size()
                varInput = Variable(input_img.view(-1, c, h, w).cuda(), volatile=True)
                out = model(varInput)
                out = self.m(out)
                outMean = out.view(bs, n_crops, -1).mean(1)

                outPRED = torch.cat((outPRED, outMean.data), 0)

            aurocIndividual = computeAUROC(outGT, outPRED, self.class_num)
            aurocMean = np.array(aurocIndividual).mean()

            for i in range(0, len(aurocIndividual)):
                print(' [%d]' % (i + 1), class_name[i], ' ', aurocIndividual[i])

            print('\n', ' [*] AUROC mean ', aurocMean)

    def robust_perturbation(self):
        assign_random_seed(self.v_seed)
        rp_dir = self.xai_dir + '/rp'
        check_dir(rp_dir)

        model = self.backbone_net.to(self.device)
        model_list = glob(os.path.join(self.result_dir, self.model_dir, '*.pt'))

        if not len(model_list) == 0:
            model_list.sort()
            start_epoch = int(model_list[-1].split('_')[-2])
            counter = int(model_list[-1].split('_')[-1].split('.')[0])

            self.load(os.path.join(self.result_dir, self.model_dir), start_epoch, counter)

            print("\n [*] Load SUCCESS")
            print(" [*] Params from epoch %d / counter %d LOADED !\n" % (start_epoch, counter))

        else:
            raise Exception(" [*] Load FAIL! Train Model FIRST !\n")

        model.eval()
        cudnn.benchmark = True

        test_dataloader = get_xai_data_loader(self.xai_data, self.class_path, self.xai_batch)

        for i_idx, data in enumerate(test_dataloader):
            attr_map = robust(data, model, self.device, self.class_num, rp_dir, i_idx, 150, visual=True)

    def energy(self):
        assign_random_seed(self.v_seed)
        torch.backends.cudnn.deterministic = True

        h5path = self.xai_dir + '/rp/h5'

        onlyfiles = [f for f in listdir(h5path) if isfile(join(h5path, f))]

        total_energy = 0

        cudnn.benchmark = True

        counter = 0
        for h5file in enumerate(onlyfiles):
            h5f = h5py.File(h5path + '/' + h5file[1], 'r')

            attr_map = h5f['attr_map'].value
            ori_img = cv2.imread(self.xai_data + '/JPEGImages/' + h5f['directory'].value)
            h, w, c = ori_img.shape

            attr_map = resize(attr_map, (h, w), anti_aliasing=True)

            attr_min = attr_map.min()
            attr_max = attr_map.max()

            attr_map = (attr_map - attr_min) / (attr_max - attr_min)
            temp_energy = energy_pointing_game(h5f['bbox'].value, attr_map, self.device, self.class_num).run()

            if math.isnan(temp_energy) == False:
                total_energy += temp_energy
            counter += 1

        mean_energy = total_energy / counter
        print(" [*] mean energy: ", mean_energy)

        return mean_energy

    def indel_game(self):
        assign_random_seed(self.v_seed)
        torch.backends.cudnn.deterministic = True

        model = self.backbone_net
        model_list = glob(os.path.join(self.result_dir, self.model_dir, '*.pt'))

        if not len(model_list) == 0:
            model_list.sort()
            start_epoch = int(model_list[-1].split('_')[-2])
            counter = int(model_list[-1].split('_')[-1].split('.')[0])

            self.load(os.path.join(self.result_dir, self.model_dir), start_epoch, counter)

            print("\n [*] Load SUCCESS")
            print(" [*] Params from epoch %d / counter %d LOADED !\n" % (start_epoch, counter))

        else:
            raise Exception(" [*] Load FAIL! Train Model FIRST !\n")

        model.eval()

        cudnn.benchmark = True
        h5path = self.xai_dir + '/rp/h5'

        # image size for insertion & deletion evaluation
        img_size = 300

        onlyfiles = [f for f in sorted(listdir(h5path)) if isfile(join(h5path, f))]
        total_ins = []
        total_del = []

        for m_idx in range(len(onlyfiles)):
            h5f = h5py.File(h5path + '/' + onlyfiles[m_idx], 'r')
            input_img = get_indel_data_loader(self.xai_data + '/JPEGImages/' + h5f['directory'].value)
            attr_map = h5f['attr_map'].value
            target_c = h5f['class'].value

            blur = lambda input_img: nn.functional.conv2d(input_img, gkern(11,5).to(self.device), padding=11//2)

            print(" [%4d/%4d] Insertion Game is processing..." % (m_idx + 1, len(onlyfiles)))
            ins_score = InsertionDeletion(input_img, target_c, attr_map, model, self.device, img_size, mode='ins', step=100,
                                         substrate_fn=blur).run()
            print(" [%4d/%4d] Deletion Game is processing..." % (m_idx + 1, len(onlyfiles)))
            del_score = InsertionDeletion(input_img, target_c, attr_map, model, self.device, img_size, mode='del', step=300,
                                         substrate_fn=torch.zeros_like).run()

            total_ins.append(auc(ins_score))
            total_del.append(auc(del_score))

        insertion = np.array(total_ins).mean()
        insertion_v = np.array(total_ins).std()
        deletion = np.array(total_del).mean()
        deletion_v = np.array(total_del).std()

        print("insertion mean :", insertion)
        print("insertion std :", insertion_v)
        print("deletion mean :", deletion)
        print("deletion std :", deletion_v)

    def c_sen(self):
        assign_random_seed(self.v_seed)
        torch.backends.cudnn.deterministic = True

        h5path = self.xai_dir + '/rp/h5'
        onlyfiles = [f for f in listdir(h5path) if isfile(join(h5path, f))]
        cudnn.benchmark = True

        value1 = []
        value2 = []

        # load attribution data from h5 file.
        for h5file in enumerate(onlyfiles):
            h5f = h5py.File(h5path + '/' + h5file[1], 'r')

            t_class = h5f['class'].value
            pred = h5f['pred'].value
            prt_pred = h5f['prt'].value
            psv_pred = h5f['psv'].value

            pred = np.expand_dims(pred, axis=0)
            prt_pred = np.expand_dims(prt_pred, axis=0)
            psv_pred = np.expand_dims(psv_pred, axis=0)

            pred_t = pred[:, t_class]
            prt_t = prt_pred[:, t_class]
            psv_t = psv_pred[:,t_class]

            pred_nt = np.concatenate([pred[:, :t_class], pred[:, t_class + 1:]], axis=-1)
            prt_nt = np.concatenate([prt_pred[:, :t_class], prt_pred[:, t_class + 1:]], axis=-1)
            psv_nt = np.concatenate([psv_pred[:, :t_class], psv_pred[:, t_class + 1:]], axis=-1)

            # Please refer the equation of class distortion in the main paper.
            eq1 = np_kld(pred_t, prt_t)
            eq2 = np_kld(pred_nt, prt_nt)

            eq3 = np_kld(pred_t, psv_t)
            eq4 = np_kld(pred_nt, psv_nt)

            total_score = np.log(((eq1 + 1e-7) / (eq2 + 1e-7)) + 1)
            total_score_r = np.log(((eq3 + 1e-7) / (eq4 + 1e-7)) + 1)

            value1.append(total_score)
            value2.append(total_score_r)

        print(" [*] Class Distortion Score: %.4f, %.2f" %(statistics.mean(value1), statistics.stdev(value1)))
        print(" [*] Reverse Class Distortion Score: %.4f, %.2f" %(statistics.mean(value2), statistics.stdev(value2)))