from .BaseOpt import BaseOpt



class TrainOpt(BaseOpt):



	def initialize(self, parser):
		parser = BaseOpt.initialize(self, parser)

		parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train the model')
		parser.add_argument('--lr_discriminator', type=float, default=0.0002, help='learning rate for the discriminator network')
		parser.add_argument('--lr_generator', type=float, default=0.0002, help='learning rate for the generator network')
		parser.add_argument('--pretrained', action='store_true', help='if specified, fine tune a pretrained model' )
		parser.add_argument('--model_path', type=str, default='./', help='path to the pretrained model')
		parser.add_argument('--decay_rate', type=float, default=0.95, help='decay rate')

		self.isTrain = True

		return parser
