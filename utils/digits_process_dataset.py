import torchvision
from torchvision import transforms
import numpy as np
import time
import torch.nn.parallel
import torch.optim
import torch.utils.data
import scipy
from scipy import misc
from scipy import io
import pickle
import PIL.Image as Image
from utils.ops import *

def _dataset_info(data_dir, txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    images = np.zeros((len(images_list), 222, 222, 3),dtype=float)
    labels = np.zeros(len(images_list), dtype=int)
    idx = 0
    for row in images_list:
        row = row.split(' ')
        img = np.array(Image.open(data_dir+"/"+row[0]).convert('RGB').resize((222,222))) / 255. # HWC
        images[idx] = img
        labels[idx] = int(row[1])
        idx += 1
    return images, labels

"""
Load datasets:
    Shape: Channel X H X W

"""
def load_cifar(data_dir, dname='cifar10', split='train'):
    if dname == 'cifar10':
        return get_CIFAR_dataset(data_dir, split)
    else:
        return get_CIFAR_C_dataset(data_dir, dname, split, level=5)

def get_CIFAR_dataset(data_dir, split, tf=None):
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=(split=='train'),
                                            download=True, transform=None)
    imgdata = trainset.data
    labels = trainset.targets
    print ('Loading CIFAR10 '+split+' dataset: '+str(len(labels)))

    img_tr = [
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0)
    ]
    tf = transforms.Compose(img_tr)

    imgs = []
    for imgd in imgdata:
        img = Image.fromarray(imgd,'RGB')
        imgd_aft = tf(img)
        imgs.append(imgd_aft)

    imgdata_aft = np.stack(imgs, axis=0)
    return imgdata_aft, labels


def get_CIFAR_C_dataset(data_dir, dname, split=None, level=1):    
    if dname == 'cifar10':
        return get_CIFAR_dataset('train')
    
    print ('Loading CIFAR-C {0} dataset at severity {1}'.format(dname, level))
    imgdata = np.load(data_dir+'/CIFAR-10-C/'+dname+'.npy')
    labels = np.load(data_dir+'/CIFAR-10-C/labels.npy')

    img_tr = [
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0)
    ]
    tf = transforms.Compose(img_tr)

    if level != 0:
        start = (level - 1) * 10000
        imgdata, labels = imgdata[start:start + 10000], labels[start:start + 10000].tolist()

    imgs = []
    for imgd in imgdata:
        imgd_aft = tf(imgd)
        imgs.append(imgd_aft)
    imgdata_aft = np.stack(imgs, axis=0)
    return imgdata_aft, labels
    

def load_pacs(data_dir, dname='photo', split='train'):
    
    images, labels = _dataset_info(data_dir, data_dir+"/PACS/kfold/%s_%s.txt" % (dname, split))
    print('PACS-%s %s dataset loaded, size: %d' % (dname, split, images.shape[0]))
    return images, labels

def load_svhn(data_dir, split='train'):
    print('Loading SVHN dataset.')
    image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'
    image_dir = os.path.join(data_dir, 'Digits/svhn', image_file)
    svhn = io.loadmat(image_dir)
    images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 255.
    labels = svhn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    return images, labels

def load_mnist(data_dir, split='train'):
    print('Loading MNIST dataset.')
    image_file = 'train.pkl' if split == 'train' else 'test.pkl'
    image_dir = os.path.join(data_dir, 'Digits/MNIST', image_file)
    with open(image_dir, 'rb') as f:
        mnist = pickle.load(f, encoding="bytes")
    images = mnist[b'X']
    labels = mnist[b'y']
    images = images / 255.
    images = np.stack((images, images, images), axis=3)  # grayscale to rgb

    images = np.squeeze(images[:10000])
    labels = labels[:10000]

    return images, labels

def load_mnist_m(data_dir, split='train'):
    print('Loading MNIST_M dataset.')

    image_dir = os.path.join(data_dir, 'Digits/mnist_m')

    if split == 'train':
        data_dir = os.path.join(image_dir, 'mnist_m_train')
        with open(os.path.join(image_dir, 'mnist_m_train_labels.txt')) as f:
            content = f.readlines()

    elif split == 'test':
        data_dir = os.path.join(image_dir, 'mnist_m_test')
        with open(os.path.join(image_dir, 'mnist_m_test_labels.txt')) as f:
            content = f.readlines()

    content = [c.split('\n')[0] for c in content]
    images_files = [c.split(' ')[0] for c in content]
    labels = np.array([int(c.split(' ')[1]) for c in content]).reshape(-1)
    images = np.zeros((len(labels), 32, 32, 3))
    for no_img, img in enumerate(images_files):
        img_dir = os.path.join(data_dir, img)
        im = misc.imread(img_dir)
        im = np.expand_dims(im, axis=0)
        images[no_img] = im
    images = images
    images = images / 255.
    return images, labels

def load_syn(data_dir, split='train'):
    print('Loading SYN dataset.')
    image_file = 'synth_train_32x32.mat' if split == 'train' else 'synth_test_32x32.mat'
    image_dir = os.path.join(data_dir, 'Digits/syn', image_file)
    syn = scipy.io.loadmat(image_dir)
    images = np.transpose(syn['X'], [3, 0, 1, 2])
    labels = syn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    images = images / 255.
    return images, labels

def load_usps(data_dir, split='train'):
    print('Loading USPS dataset.')
    image_file = 'usps_train_32x32.pkl' if split == 'train' else 'usps_test_32x32.pkl'
    # image_file = 'usps_32x32.pkl'
    image_dir = os.path.join(data_dir, 'Digits/usps', image_file)
    with open(image_dir, 'rb') as f:
        usps = pickle.load(f, encoding="bytes")
    images = usps['X']
    labels = usps['y']
    print('label range [{0}-{1}]'.format(np.min(labels), np.max(labels)))
    # labels -= 1
    # labels[labels == 255] = 9
    if np.max(images) == 255:
        images = images / 255.
    assert np.max(images) == 1
    images = np.squeeze(images)
    images = np.stack((images, images, images), axis=3)  # grayscale to rgb
    return images, labels

def load_test_data(data_dir, target):
    if target == 'svhn':
        target_test_images, target_test_labels = load_svhn(data_dir, split='test')
    elif target == 'mnist':
        target_test_images, target_test_labels = load_mnist(data_dir, split='test')
    elif target == 'syn':
        target_test_images, target_test_labels = load_syn(data_dir, split='test')
    elif target == 'usps':
        target_test_images, target_test_labels = load_usps(data_dir, split='test')
    elif target == 'mnist_m':
        target_test_images, target_test_labels = load_mnist_m(data_dir, split='test')
    elif target in ['fog','snow', 'frost',
                       'zoom_blur', 'defocus_blur', 'glass_blur',
                       'speckle_noise', 'shot_noise', 'impulse_noise',
                       'jpeg_compression', 'pixelate', 'spatter', 'elastic_transform','gaussian_blur','saturate','brightness','contrast','gaussian_noise','motion_blur'
                      ]:
        target_test_images, target_test_labels = load_cifar(data_dir, target, split='test')
    else:
        target_test_images, target_test_labels = load_pacs(data_dir, target, split='test')
    return target_test_images, target_test_labels

def asarray_and_reshape(imgs, labels, img_shape):
    imgs = np.asarray(imgs)
    labels = np.asarray(labels)
    # B X C X H X W
    imgs = np.reshape(imgs, (-1, img_shape[3],img_shape[1], img_shape[2]))
    labels = np.reshape(labels, (-1,))
    return imgs, labels

def construct_datasets(data_dir, batch_size, kwargs, sd='mnist'):

    def data2loader(imgs, labels):
        assert len(imgs) == len(labels)
        y = torch.stack([torch.from_numpy(np.array(i)) for i in labels])
        if imgs.shape[3] == 3:
            imgs = np.transpose(imgs, (0, 3, 1, 2))  # pytorch CHW, tf HWC
        X = torch.stack([torch.from_numpy(imgs[i]) for i in range(len(labels))])
        X_dataset = torch.utils.data.TensorDataset(X, y)
        X_loader = torch.utils.data.DataLoader(X_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        return X_loader

    if sd == 'mnist':
        train_imgs, train_labels = load_mnist(data_dir, 'train')
        val_imgs, val_labels = load_mnist(data_dir, 'test')
    elif sd == 'cifar10':
        train_imgs, train_labels = load_cifar(data_dir, 'cifar10', 'train')
        val_imgs, val_labels = load_cifar(data_dir, 'cifar10','test')
    else:
        train_imgs, train_labels = load_pacs(data_dir, sd, 'train')
        val_imgs, val_labels = load_pacs(data_dir, sd, 'test')

    return data2loader(train_imgs, train_labels), data2loader(val_imgs, val_labels), train_imgs.shape

def validate(val_loader, model):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    params = list(model.parameters())
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True).long()
        input = input.cuda(non_blocking=True).float()
        with torch.no_grad():
            output, _ = model.functional(params, False, input)
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg

def evaluation(model, data_dir, batch_size, kwargs, sd='mnist'):

    def data2loader(imgs, labels):
        assert len(imgs) == len(labels)
        y = torch.stack([torch.from_numpy(np.array(i)) for i in labels])
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        X = torch.stack([torch.from_numpy(imgs[i]) for i in range(len(labels))])
        X_dataset = torch.utils.data.TensorDataset(X, y)
        X_loader = torch.utils.data.DataLoader(X_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
        return  X_loader

    model.eval()
    params = list(model.parameters())
    accs = []
    
    if sd == 'mnist':
        target_domains = ['svhn', 'mnist_m', 'syn', 'usps']
    elif sd == 'cifar10':
        target_domains = ['fog','snow', 'frost',
                       'zoom_blur', 'defocus_blur', 'glass_blur',
                       'speckle_noise', 'shot_noise', 'impulse_noise',
                       'jpeg_compression', 'pixelate', 'spatter', 'elastic_transform','gaussian_blur','saturate','brightness','contrast','gaussian_noise','motion_blur'
                      ]
    else:
        target_domains = [item for item in ['photo', 'art_painting', 'cartoon', 'sketch'] if item != sd ]
    
    for td in target_domains:
        print(td)
        target_test_images, target_test_labels = load_test_data(data_dir, td)
        test_loader = data2loader(target_test_images, target_test_labels)

        batch_time = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        for i, (input, target) in enumerate(test_loader):
            target = target.cuda(non_blocking=True).long()
            input = input.cuda(non_blocking=True).float()
            with torch.no_grad():
                output, _ = model.functional(params, False, input)
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            top1.update(prec1.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        accs.append(top1.avg)
        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    avg_acc = np.mean(accs)
    accs.append(avg_acc)
    print('avg acc', avg_acc)
