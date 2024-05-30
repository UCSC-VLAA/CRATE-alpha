import numpy as np
import torch
import sys
import torch
from torchvision import transforms
import torchvision
from imagenet_templates import  *
import torch.nn.functional as  F
from model import clipa_model


transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485 , 0.456 , 0.406], std=[0.229, 0.224, 0.225 ])
        ])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenize = clipa_model.get_crate_clipa()


pth_path ="[The path of checkpoint file]"

model_weights = torch.load(pth_path)

model.load_state_dict(model_weights)
model = model.to(device) 
model = model.eval()

image_dataset = torchvision.datasets.ImageNet("[The directory of the ImageNet-1K validation set]", split='val', transform=transform)


def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        # for classname in tqdm(classnames):
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= (class_embeddings.norm(dim=-1, keepdim=True) +  1e-8)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
     
zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)

loader = torch.utils.data.DataLoader(image_dataset, batch_size=64, num_workers=6)


with torch.no_grad():
    top1, top5, n = 0., 0., 0.
    for i, (images, target) in enumerate(loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # predict
        image_features = model.encode_image(images)
        image_features /= (image_features.norm(dim=-1, keepdim=True) +  1e-8 )
        logits = 100. * image_features @ zeroshot_weights

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)
        print(f'finished {i} batches')
     

top1 = (top1 / n) * 100
top5 = (top5 / n) * 100 

print(f"Top-1 accuracy: {top1:.5f}")
print(f"Top-5 accuracy: {top5:.5f}")
     
