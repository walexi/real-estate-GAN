import torch.optim as optim
from torchvision.utils import save_image
from datetime import datetime
from libs.compute import *
from libs.constant import *
from libs.model import *
from options import TestOpt
from PIL import Image
from numpy import array


def processImg(opt, file_in_name):

    file_out_name_without_ext = os.path.splitext(file_in_name)[0]
    
    input_img = np.array( Image.open(file_in_name))

    h, w, c = input_img.shape
    rate = int(round(max(h, w) / data_image_size))
    if rate == 0:
        rate = 1


    generator = GeneratorWDilation(1)

    #generator = nn.DataParallel(generator)

    generator.load_state_dict(torch.load(opt.path, map_location=device))
    
    generator = GeneratorWDilationamp(generator,rate)
    generator = nn.DataParallel(generator)

    if torch.cuda.is_available():
        device = torch.device('cuda')    # Default CUDA device
        generator.to(device)

    generator.eval()

    padrf = rate * data_padrf_size
    patch = data_patch_size

    pad_h = 0 if h % patch == 0 else patch - (h % patch)
    pad_w = 0 if w % patch == 0 else patch - (w % patch)
    pad_h = pad_h + padrf if pad_h < padrf else pad_h
    pad_w = pad_w + padrf if pad_w < padrf else pad_w

    input_img = np.pad(input_img, [ (padrf, pad_h),(padrf, pad_w), (0, 0)], 'reflect')
   



    input_img=input_img.transpose((2,0,1))


    input_img =  input_img / 255



    y_list = []
    
    #process for each chunk
    for y in range(padrf, h+padrf, patch):
        x_list = []
        for x in range(padrf, w+padrf, patch):
            
            crop_img = input_img[None,:,y-padrf:y+padrf+patch,  x-padrf:x+padrf+patch]

            crop_img = Variable(torch.Tensor(crop_img).type(Tensor_gpu)) 
            
            enhance_test_img = generator(crop_img)

            enhance_test_img = enhance_test_img[:,:, padrf:-padrf, padrf:-padrf]
           
            x_list.append(enhance_test_img.detach().cpu())

        y_list.append(torch.cat(x_list, axis=3))
    
    enhance_test_img = torch.cat(y_list, axis=2)



    enhance_test_img = enhance_test_img[:,:,:h, :w]




    save_image( enhance_test_img, opt.output + file_out_name_without_ext + '.jpg')



if __name__ == "__main__":

    start_time = datetime.now()

    opt = TestOpt().parse

    generator = Generator()
   # generator = nn.DataParallel(generator)
    generator.load_state_dict(torch.load(opt.path))

    generator.eval()

    discriminator = Discriminator()
  #  discriminator = nn.DataParallel(discriminator)

    if torch.cuda.is_available():
        generator.cuda(device=device)
        discriminator.cuda(device=device)

    # Loading Training and Test Set Data
    _, _, _, testLoader = data_loader(opt)



    # TEST NETWORK
    batches_done = 0

    if(opt.unbounded):
      for file in os.dir(opt.path):
        processImg(opt, file)
    else:
      with torch.no_grad():
          psnrAvg = 0.0
          for i, (gt, data) in enumerate(testLoader, 0):
              input, dummy = data
              groundTruth, dummy = gt
              trainInput = Variable(input.type(Tensor_gpu))
              realImgs = Variable(groundTruth.type(Tensor_gpu))
              output = generator(trainInput)
              loss = criterion(output, realImgs)
              psnr = 10*torch.log10(1/loss)
              psnrAvg += psnr
              save_image(output.data[:25], opt.output + "/test%d.png" % i, nrow=5, normalize=True)
              # for j in range(output.shape[0]):
                  # imshowOutput(output[j,...], j)
              print("PSNR Avg: %f" % (psnrAvg / (i+1)))
          print("Final PSNR Avg: %f" % (psnrAvg / len(testLoader)))