import json
from datetime import datetime
import torch
from unet.unet_transfer import UNet16, UNetResNet


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# def cuda(x):
#     # return x.cuda(async=True) if torch.cuda.is_available() else x
#     return x.cuda(non_blocking=True) if torch.cuda.is_available() else x
def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()

def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.
    Args:
        image_height:
        image_width:
    Returns:
        True if both height and width divisible by 32 and False otherwise.
    """
    return image_height % 32 == 0 and image_width % 32 == 0

def create_model(device, type ='vgg16'):
    assert type == 'vgg16' or type == 'resnet101'
    if type == 'vgg16':
        model = UNet16(pretrained=True)
    elif type == 'resnet101':
        model = UNetResNet(pretrained=True, encoder_depth=101, num_classes=1)
    else:
        assert False
    model.eval()
    return model.to(device)

def load_unet_vgg16(model_path):
    model = UNet16(pretrained=True)
    checkpoint = torch.load(model_path,  map_location=torch.device('cpu'))
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['check_point'])
    else:
        raise Exception('undefind model format')

    # model.cuda()
    model.cpu()
    model.eval()

    return model

def load_unet_resnet_101(model_path):
    model = UNetResNet(pretrained=True, encoder_depth=101, num_classes=1)
    checkpoint = torch.load(model_path,  map_location=torch.device('cpu'))
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['check_point'])
    else:
        raise Exception('undefind model format')

    # model.cuda()
    model.cpu()
    model.eval()

    return model

def load_unet_resnet_34(model_path):
    model = UNetResNet(pretrained=True, encoder_depth=34, num_classes=1)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['check_point'])
    else:
        raise Exception('undefind model format')

    # model.cuda()
    model.cpu()
    model.eval()

    return model
