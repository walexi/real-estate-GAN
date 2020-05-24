import torch.optim as optim
from torchvision.utils import save_image
from datetime import datetime
from libs.compute import *
from libs.constant import *
from libs.model import *
from options import TestOpt



if __name__ == "__main__":

    start_time = datetime.now()

    opt = TestOpt().parse

    generator = Generator()
   # generator = nn.DataParallel(generator)
    generator.load_state_dict(torch.load(opt.path))

    generator.train()

    discriminator = Discriminator()
  #  discriminator = nn.DataParallel(discriminator)

    if torch.cuda.is_available():
        generator.cuda(device=device)
        discriminator.cuda(device=device)

    # Loading Training and Test Set Data
    _, _, _, testLoader = data_loader(opt)



    # TEST NETWORK
    batches_done = 0
    with torch.no_grad():
        psnrAvg = 0.0
        for j, (gt, data) in enumerate(testLoader, 0):
            input, dummy = data
            groundTruth, dummy = gt
            trainInput = Variable(input.type(Tensor_gpu))
            real_imgs = Variable(groundTruth.type(Tensor_gpu))
            output = generator(trainInput)
            loss = criterion(output, real_imgs)
            psnr = 10 * torch.log10(1 / loss)
            psnrAvg += psnr

            if batches_done >= 95:
                for k in range(0, output.data.shape[0]):
                    save_image(output.data[k],
                               "./models/test_images/1Way/test_%d_%d_%d.png" % (batches_done + 1, j + 1, k + 1),
                               nrow=1,
                               normalize=True)
                for k in range(0, real_imgs.data.shape[0]):
                    save_image(real_imgs.data[k],
                               "./models/gt_images/1Way/gt_%d_%d_%d.png" % (batches_done + 1, j + 1, k + 1),
                               nrow=1,
                               normalize=True)
                for k in range(0, trainInput.data.shape[0]):
                    save_image(trainInput.data[k],
                               "./models/input_images/1Way/input_%d_%d_%d.png" % (batches_done + 1, j + 1, k + 1), nrow=1,
                               normalize=True)

            batches_done += 5
            print("Loss loss: %f" % loss)
            print("PSNR Avg: %f" % (psnrAvg / (j + 1)))
            f = open("./models/psnr_Score.txt", "a+")
            f.write("PSNR Avg: %f" % (psnrAvg / (j + 1)))
        f = open("./models/psnr_Score.txt", "a+")
        f.write("Final PSNR Avg: %f" % (psnrAvg / len(testLoader)))
        print("Final PSNR Avg: %f" % (psnrAvg / len(testLoader)))
