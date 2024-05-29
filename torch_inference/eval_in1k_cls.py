import torch
from torchvision import transforms
import torchvision
from model import crate_alpha


transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485 , 0.456 , 0.406], std=[0.229, 0.224, 0.225 ])
        ])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = crate_alpha.CRATE_base()
# model = crate_alpha.CRATE_large()

pth_path ="[The path of checkpoint file]"
model_weights = torch.load(pth_path)
model.load_state_dict(model_weights)

model = model.to(device)
model = model.eval()


image_dataset = torchvision.datasets.ImageNet("[The directory of the ImageNet-1K validation set]", split='val', transform=transform)


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
     

loader = torch.utils.data.DataLoader(image_dataset, batch_size=512, num_workers=8)


with torch.no_grad():
    top1, top5, n = 0., 0., 0.
    for i, (images, target) in enumerate(loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # predict
        logits = model(images)

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
     
