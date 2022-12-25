import torch
import torchvision
import torchvision.transforms as transfroms
from conf import settings

# e.g. total training dataset: 700
# Batch size (mini batch): 100
# epoch는 전체 트레이닝 셋이 신경망을 통과한 횟수 의미합니다. 예를 들어, 1-epoch는 전체 트레이닝 셋이 하나의 신경망에 적용되어 순전파와 역전파를 통해 신경망을 한 번 통과했다는 것을 의미합니다.
# iteration은 1-epoch를 마치는데 필요한 미니배치 갯수를 의미합니다. 다른 말로, 1-epoch를 마치는데 필요한 파라미터 업데이트 횟수 이기도 합니다. 각 미니 배치 마다 파라미터 업데이터가 한번씩 진행되므로 iteration은 파라미터 업데이트 횟수이자 미니배치 갯수입니다. 예를 들어, 700개의 데이터를 100개씩 7개의 미니배치로 나누었을때, 1-epoch를 위해서는 7-iteration이 필요하며 7번의 파라미터 업데이트가 진행됩니다.

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

train_set = torchvision.datasets.FashionMNIST(root = './data/FashionMNIST', train = True, download = True, transform = transfroms.Compose([transfroms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST(root = './data/FashionMNIST',train = False,download = True,transform = transfroms.Compose([transfroms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=settings.BATCH)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=settings.BATCH)