import torchvision.datasets.voc as voc
from torchvision import transforms
from torch.utils.data import DataLoader
from util.ops import *
import torch
from util.Format import YOLO as cvtYOLO
from util.Format import VOC_d as cvtVOC

class PascalVOC_Dataset(voc.VOCDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
                (default: alphabetic indexing of VOC's 20 classes).
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, required): A function/transform that takes in the
                target and transforms it.
    """

    def __init__(self, root, year='2007', image_set='train', download=False, transform=None, target_transform=None):

        super().__init__(
             root,
             year=year,
             image_set=image_set,
             download=download,
             transform=transform,
             target_transform=target_transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        return super().__getitem__(index)

    def __len__(self):
        """
        Returns:
            size of the dataset
        """
        return len(self.images)

class PascalVOC_Dataset2(voc.VOCDetection):
    IMAGE_FOLDER = "JPEGImages"
    LABEL_FOLDER = "Annotations"
    IMG_EXTENSIONS = '.jpg'

    def __init__(self, root, train=True, transform=None, target_transform=None, resize=300, class_path='./voc.names'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize_factor = resize
        self.class_path = class_path

        with open(class_path) as f:
            self.classes = f.read().splitlines()

        self.data = self.cvtData()

    def cvtData(self):

        result = []
        voc = cvtVOC()

        yolo = cvtYOLO(os.path.abspath(self.class_path))
        flag, self.dict_data = voc.parse(os.path.join(self.root, self.LABEL_FOLDER))

        try:

            if flag:
                flag, data =yolo.generate(self.dict_data)

                keys = list(data.keys())
                keys = sorted(keys, key=lambda key: int(key.split("_")[-1]))

                for key in keys:
                    contents = list(filter(None, data[key].split("\n")))
                    target = []
                    for i in range(len(contents)):
                        tmp = contents[i]
                        tmp = tmp.split(" ")
                        for j in range(len(tmp)):
                            tmp[j] = float(tmp[j])
                        target.append(tmp)

                    result.append({os.path.join(self.root, self.IMAGE_FOLDER, "".join([key, self.IMG_EXTENSIONS])) : target})

                return result

        except Exception as e:
            raise RuntimeError("Error : {}".format(e))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        key = list(self.data[index].keys())[0]

        img = Image.open(key).convert('RGB')
        current_shape = img.size

        target = self.data[index][key]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            # Future works
            pass

        return img, target, current_shape, key

def get_voc_data_loader(data_dir, batch_size, split='train'):
    mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    transformations = transforms.Compose([transforms.Resize((300, 300)),
                                          transforms.RandomChoice([
                                              transforms.ColorJitter(brightness=(0.80, 1.20)),
                                              transforms.RandomGrayscale(p=0.25)]),
                                          transforms.RandomHorizontalFlip(p=0.25),
                                          transforms.RandomRotation(25),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std), ])

    transformations_valid = transforms.Compose([transforms.Resize(330),
                                               transforms.CenterCrop(300),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=mean, std=std),])

    transformations_test = transforms.Compose([transforms.Resize(330),
                                               transforms.FiveCrop(300),
                                               transforms.Lambda(lambda crops: torch.stack(
                                                   [transforms.ToTensor()(crop) for crop in crops])),
                                               transforms.Lambda(lambda crops: torch.stack(
                                                   [transforms.Normalize(mean=mean, std=std)(crop) for crop in crops])),])
    if split == 'train':
        dataset_train = PascalVOC_Dataset(data_dir,
                                          year='2007',
                                          image_set='train',
                                          download=False,
                                          transform=transformations,
                                          target_transform=encode_labels)
        loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=8, shuffle=True)

    elif split == 'val':
        dataset_valid = PascalVOC_Dataset(data_dir,
                                          year='2007',
                                          image_set='val',
                                          download=False,
                                          transform=transformations_valid,
                                          target_transform=encode_labels)
        loader = DataLoader(dataset_valid, batch_size=batch_size, num_workers=8)

    elif split == 'test':
        dataset_test = PascalVOC_Dataset(data_dir,
                                         year='2007',
                                         image_set='val',
                                         download=False,
                                         transform=transformations_test,
                                         target_transform=encode_labels)
        loader = DataLoader(dataset_test, batch_size=int(batch_size/5), num_workers=8, shuffle=False)

    return loader


def get_xai_data_loader(data_path, class_path, batch_size, shuffle=True):
    mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    #root, train = True, transform = None, target_transform = None, resize = 300, class_path = './voc.names'):

    transformations = transforms.Compose([transforms.Resize((300, 300)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std), ])

    datasetTest= PascalVOC_Dataset2(root=data_path, class_path=class_path, transform=transformations)

    loader = DataLoader(dataset=datasetTest, batch_size=batch_size, shuffle=shuffle)#, collate_fn=detection_collate)

    return loader

def get_indel_data_loader(image_name):
    mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    transformations = transforms.Compose([transforms.Resize((300, 300)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std), ])

    image = Image.open(image_name)
    image = transformations(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet

    return image.cuda()  #assumes that you're using GPU

def get_exp_data_loader(image_name):
    transformations = transforms.Compose([transforms.Resize((300, 300)),
                                          transforms.ToTensor(),])

    image = Image.open(image_name)
    image = transformations(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet

    return image.cuda()  #assumes that you're using GPU