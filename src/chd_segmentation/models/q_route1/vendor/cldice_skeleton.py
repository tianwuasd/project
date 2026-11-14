import torch
import torch.nn.functional as F

# 来源：jocpae/clDice，提交47d31a6cc4a8101b1ffe8052994821961e57af9f，MIT许可见LICENSE-clDice.txt。
class SoftSkeletonize(torch.nn.Module):
    def __init__(self, num_iter=40):
        super().__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img,(3,1),(1,1),(1,0))
            p2 = -F.max_pool2d(-img,(1,3),(1,1),(0,1))
            return torch.min(p1,p2)
        if len(img.shape) == 5:
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1,p2),p3)
        raise ValueError('soft skeleton只支持二维或三维批次')

    def soft_dilate(self, img):
        if len(img.shape) == 4:
            return F.max_pool2d(img,(3,3),(1,1),(1,1))
        if len(img.shape) == 5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))
        raise ValueError('soft skeleton只支持二维或三维批次')

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img-img1)
        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img-img1)
            skel = skel+F.relu(delta-skel*delta)
        return skel

    def forward(self, img):
        return self.soft_skel(img)

