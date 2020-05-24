import torch.optim as optim
import torch
from torchvision.utils import save_image
from datetime import datetime
from libs.compute import *
from libs.constant import *
from libs.model import *
from options import train_opt as TrainOpt


def computeGradientPenalty(D, realSample, fakeSample):
    alpha = Tensor_gpu(np.random.random((realSample.shape)))
    interpolates = (alpha * realSample + ((1 - alpha) * fakeSample)).requires_grad_(True)
    dInterpolation = D(interpolates)
    fakeOutput = Variable(Tensor_gpu(realSample.shape[0],1,1,1).fill_(1.0), requires_grad=False)
    
    gradients = autograd.grad(
        outputs = dInterpolation,
        inputs = interpolates,
        grad_outputs = fakeOutput,
        create_graph = True,
        retain_graph = True,
        only_inputs = True)[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    maxVals = []
    normGradients = gradients.norm(2, dim=1)-1
    for i in range(len(normGradients)):
        if(normGradients[i] > 0):
            maxVals.append(Variable(normGradients[i].type(Tensor)).detach().numpy())
        else:
            maxVals.append(0)

    gradientPenalty = np.mean(maxVals)
    return gradientPenalty



def generatorAdversarialLoss( output_images):
    validity = discriminator(output_images)
    gen_adv_loss = torch.mean(validity)
    
    return gen_adv_loss


def computeGeneratorLoss(inputs, outputs_g1):
    # generator 1
    gen_adv_loss1 = generatorAdversarialLoss(outputs_g1)
    
    i_loss = criterion(inputs, outputs_g1)
    gen_loss = -gen_adv_loss1 + ALPHA*i_loss
    
    return gen_loss


def discriminatorLoss(d1Real, d1Fake, gradPenalty):
    return (torch.mean(d1Fake) - torch.mean(d1Real)) + (LAMBDA*gradPenalty)


if __name__ == "__main__":

    start_time = datetime.now()

    #initialize parser
    opt = TrainOpt().parse
    lr_gen = opt.lr_generator
    lr_disc = opt.lr_discriminator
    num_epochs = opt.n_epochs
    decay_rate = opt.decay_rate
    isPretrained = opt.pretrained


    generator = Generator()
    discriminator = Discriminator()

    if torch.cuda.is_available():
        device = torch.device('cuda')    # Default CUDA device
        generator.to(device)
        discriminator.to(device)

    if isPretrained:
        generator.load_state_dict(torch.load(opt.path))
        generator.eval()
    else:
        generator.train()
    
    Tensor_gpu = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    Tensor = torch.FloatTensor

    # load dataset
    trainLoader_cross, 

    # initialize loss and optim 

    criterion = nn.MSELoss()

    optimizer_g = optim.Adam(generator.parameters(), lr = lr_gen, betas=(BETA1,BETA2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr = lr_disc, betas=(BETA1,BETA2))

    batches_done = 0
    
    batches_done = 0

    for epoch in range(num_epochs*2):
        for i, (data, gt1) in enumerate(trainLoader_cross, 0):
            input, dummy = data
            groundTruth, dummy = gt1
            trainInput = Variable(input.type(Tensor_gpu))
            realImgs = Variable(groundTruth.type(Tensor_gpu))

            ### TRAIN DISCRIMINATOR
            optimizer_d.zero_grad()
            fakeImgs = generator(trainInput)

            # Real Images
            realValid = discriminator(realImgs)
            # Fake Images
            fakeValid = discriminator(fakeImgs)

            gradientPenalty = computeGradientPenalty(discriminator, realImgs.data, fakeImgs.data)
            dLoss = discriminatorLoss(realValid, fakeValid, gradientPenalty)
            dLoss.backward()
            optimizer_d.step()
            optimizer_g.zero_grad()

            ### TRAIN GENERATOR
            if batches_done % 50 == 0:
                print("Training Generator on Iteration: %d" % (i))
                # Generate a batch of images
                fake_imgs = generator(trainInput)
                residual_learning_output = fake_imgs + trainInput
                gLoss = computeGeneratorLoss(trainInput,residual_learning_output)

                gLoss.backward()
                optimizer_g.step()

                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, num_epochs , i, len(trainLoader_cross), dLoss.item(), gLoss.item()))
                f = open("log_Train.txt","a+")
                f.write("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n" % (epoch, num_epochs , i, len(trainLoader_cross), dLoss.item(), gLoss.item()))
                f.close()

                if batches_done % 200 == 0:
                    save_image(fake_imgs.data[:25], output_images + "/train%d.png" % batches_done, nrow=5, normalize=True)
                    torch.save(generator.state_dict(), output_gen + '/gan_train_'+ str(epoch) + '_' + str(i) + '.pth')
                    torch.save(discriminator.state_dict(), output_disc + '/discriminator_train_'+ str(epoch) + '_' + str(i) + '.pth')

            batches_done += 1
            print("Done training generator on iteration: %d" % (i))
    end_time = datetime.now()
    print(end_time - start_time)
