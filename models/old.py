class FC3D_1_0(nn.Module):
    def __init__(self, args):
        super(FC3D_1_0, self).__init__()
        self.conv0 = nn.Conv2d(5,16,3,padding=1)
        self.pool0 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.bn0   = nn.BatchNorm2d(16)

        self.conv1 = nn.Conv2d(16,32,3,padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.bn2   = nn.BatchNorm2d(64)

        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.dbn2     = nn.BatchNorm2d(32)

        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.dbn1     = nn.BatchNorm2d(16)

        self.deconv0 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.dbn0     = nn.BatchNorm2d(1)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.pool0(out)
        skip0 = out

        out = F.relu(self.bn1(self.conv1(out)))
        out = self.pool1(out)
        skip1 = out

        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool2(out)

        out = F.relu(self.dbn2(self.deconv2(out)))

        out = F.relu(self.dbn1(self.deconv1(out+skip1)))

        img = F.sigmoid(self.dbn0(self.deconv0(out+skip0)))

        return(img)
    


