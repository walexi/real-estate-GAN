import torch.optim as optim
from torchvision.utils import save_image
from datetime import datetime
from libs.compute import *
from libs.constant import *
from libs.model import *
from options import TrainOpt



if __name__ == "__main__":

    start_time = datetime.now()

    opt = TrainOpt().parse
    lr_gen = opt.lr_generator
    lr_disc = opt.lr_discriminator
    num_epochs = opt.n_epochs
    decay_rate = opt.decay_rate

    # Creating generator and discriminator
    generator = Generator()
   # generator = nn.DataParallel(generator)
    if opt.pretrained:
        generator.load_state_dict(torch.load(opt.model_path))

    generator.train()

    discriminator = Discriminator()
  #  discriminator = nn.DataParallel(discriminator)

    if torch.cuda.is_available():
        generator.cuda(device=device)
        discriminator.cuda(device=device)

    # Loading Training and Test Set Data
    trainLoader1, trainLoader2, trainLoader_cross, testLoader = data_loader(opt)

    # MSE Loss and Optimizer
    criterion = nn.MSELoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=lr_gen, betas=(BETA1, BETA2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_disc, betas=(BETA1, BETA2))


    # Training Network
    dataiter = iter(testLoader)
    gt_test, data_test = dataiter.next()
    input_test, dummy = data_test
    testInput = Variable(input_test.type(Tensor_gpu))
    batches_done = 0
    generator_loss = []
    discriminator_loss = []
    for epoch in range(num_epochs):

        for param_group in optimizer_d.param_groups:
            param_group['lr'] = adjustLearningRate(lr_disc, epoch_num=epoch, decay_rate=decay_rate)

        for i, (data, gt1) in enumerate(trainLoader_cross, 0):
            input, dummy = data
            groundTruth, dummy = gt1
            trainInput = Variable(input.type(Tensor_gpu))
            real_imgs = Variable(groundTruth.type(Tensor_gpu))

            # TRAIN DISCRIMINATOR
            optimizer_d.zero_grad()
            fake_imgs = generator(trainInput)

            # Real Images
            realValid = discriminator(real_imgs)
            # Fake Images
            fakeValid = discriminator(fake_imgs)

            gradientPenalty = computeGradientPenaltyFor1WayGAN(discriminator, real_imgs.data, fake_imgs.data)
            dLoss = discriminatorLoss(realValid, fakeValid, gradientPenalty)
            dLoss.backward(retain_graph=True)
            optimizer_d.step()

            if batches_done % 50 == 0:
                for param_group in optimizer_g.param_groups:
                    param_group['lr'] = adjustLearningRate(learning_rate, epoch_num=epoch, decay_rate=decay_rate)

                # TRAIN GENERATOR
                optimizer_g.zero_grad()
                gLoss = computeGeneratorLoss(trainInput, fake_imgs, discriminator, criterion)
                gLoss.backward(retain_graph=True)
                optimizer_g.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
                epoch + 1, num_epochs, i + 1, len(trainLoader_cross), dLoss.item(), gLoss.item()))

            f = open("./models/log_Train.txt", "a+")
            f.write("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n" % (
                epoch + 1, num_epochs, i + 1, len(trainLoader_cross), dLoss.item(), gLoss.item()))
            f.close()

            if batches_done % 10 == 0:
                for k in range(0, fake_imgs.data.shape[0]):
                    save_image(fake_imgs.data[k], "./models/train_images/1Way/1Way_Train_%d_%d_%d.png" % (epoch+1, batches_done+1, k+1),
                               nrow=1,
                               normalize=True)
                torch.save(generator.state_dict(),
                           './models/train_checkpoint/1Way/gan1_train_' + str(epoch+1) + '_' + str(i+1) + '.pth')
                torch.save(discriminator.state_dict(),
                           './models/train_checkpoint/1Way/discriminator_train_' + str(epoch+1) + '_' + str(i+1) + '.pth')
                fake_test_imgs = generator(testInput)
                for k in range(0, fake_test_imgs.data.shape[0]):
                    save_image(fake_test_imgs.data[k],
                               "./models/train_test_images/1Way/1Way_Train_Test_%d_%d_%d.png" % (epoch+1,batches_done+1, k+1),
                               nrow=1, normalize=True)

            batches_done += 1
            print("Done training discriminator on iteration: %d" % i)

    end_time = datetime.now()
    print(end_time - start_time)
