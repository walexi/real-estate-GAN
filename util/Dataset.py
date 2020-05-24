from .ConcatDataset import ConcatDataset
import torch

class Dataset():

	def __init__(self, opt):
		self.isTrain = opt.isTrain
		self.bs = opt.batch_size

	def load_dataset(self):
		# Converting the images for PILImage to tensor, so they can be accepted as the input to the network
		print("Loading Dataset")
		bs = self.bs

		transform = transforms.Compose([transforms.Resize((SIZE,SIZE), interpolation=2),transforms.ToTensor()])

		trainset_1_gt =torchvision.datasets.ImageFolder(root='/content/dataset/Expert-C/Training1/', transform=transform, target_transform=None)    
		trainset_2_gt =torchvision.datasets.ImageFolder(root='/content/dataset/Expert-C/Training2/', transform=transform, target_transform=None)    
		testset_gt =torchvision.datasets.ImageFolder(root='/content/dataset/Expert-C/Testing/', transform=transform, target_transform=None)      
		trainset_1_inp =torchvision.datasets.ImageFolder(root='/content/dataset/input/Training1/', transform=transform, target_transform=None)    
		trainset_2_inp =torchvision.datasets.ImageFolder(root='/content/dataset/input/Training2/', transform=transform, target_transform=None)    
		testset_inp =torchvision.datasets.ImageFolder(root='/content/dataset/input/Testing/', transform=transform, target_transform=None)


		trainLoader1 = torch.utils.data.DataLoader(
             ConcatDataset(
                 trainset_1_gt,
                 trainset_1_inp
             ),
             batch_size=bs, shuffle=True,)

		trainLoader2 = torch.utils.data.DataLoader(
		             ConcatDataset(
		                 trainset_2_gt,
		                 trainset_2_inp
		             ),
		             batch_size=bs, shuffle=True,)

		trainLoader_cross = torch.utils.data.DataLoader(
		             ConcatDataset(
		                 trainset_2_inp,
		                 trainset_1_gt
		             ),
		             batch_size=bs, shuffle=True,)


		testLoader = torch.utils.data.DataLoader(
		             ConcatDataset(
		                 testset_gt,
		                 testset_inp
		             ),
		             batch_size=bs, shuffle=True,)
		print("Finished loading dataset")


		return trainLoader1, trainLoader2, trainLoader_cross, testLoader



