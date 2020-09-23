import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class AnomalyModel(BaseModel):
    """This class implements the AnomalyModel model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode anomaly' dataset.
    By default, it uses a '--netG resnet_6blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a vanilla GAN loss ('--gan_mode vanilla' by pix2pix),
    and a lsgan GAN loss ('--gan_mode lsgan' by cycle_gan).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    Note: lsgan needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        parser.add_argument("--real_label", type=float, default=1.0, help="label for a real image")
        parser.add_argument("--fake_label", type=float, default=0.0, help="label for a fake image")
        if is_train:
            parser.add_argument("--cycle_size_G", type=int, default=1, help="iters update G in cycle")
            parser.add_argument("--cycle_size_D", type=int, default=1, help="iters update D in cycle")
            parser.add_argument("--lambda_L1", type=float, default=10.0, help="weight for L1 loss")
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ["real_A", "fake_B", "real_B"]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ["G", "D"]
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if opt.netD.startswith(("resnet", "unet")):
            self.netD = networks.define_G(opt.output_nc, 1, opt.ndf, opt.netD, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                          opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            assert(opt.input_nc == opt.output_nc)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode, opt.real_label, opt.fake_label).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.cur_iter = 0
        self.cycle_size_G = opt.cycle_size_G
        self.cycle_size_D = opt.cycle_size_D
        self.cycle_size = opt.cycle_size_G + opt.cycle_size_D

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        # Real
        self.loss_D_real = self.criterionGAN(self.netD(self.real_B), True)
        # Fake
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_fake = self.criterionGAN(self.netD(fake_B.detach()), False)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # First, GAN loss D(G(A))
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        cur_iter = self.cur_iter % self.cycle_size
        self.forward()                                # compute fake images: G(A)
        # update D
        if self.cycle_size <= 2 or cur_iter < self.cycle_size_D:
            self.set_requires_grad(self.netD, True)   # enable backprop for D
            self.optimizer_D.zero_grad()              # set D's gradients to zero
            self.backward_D()                         # calculate gradients for D
            self.optimizer_D.step()                   # update D's weights
        # update G
        if self.cycle_size <= 2 or cur_iter >= self.cycle_size_D:
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()              # set G's gradients to zero
            self.backward_G()                         # calculate graidents for G
            self.optimizer_G.step()                   # udpate G's weights
        self.cur_iter += 1

    def test(self):
        with torch.no_grad():
            self.fake_B = self.netG(self.real_A)  # G(A)
            self.real_B = self.netD(self.real_A)  # D(A)
            self.compute_visuals()

    def compute_visuals(self):
        pass
