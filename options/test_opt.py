from .BaseOpt import BaseOpt



class TestOpt(BaseOpt):



	def initialize(self, parser):
		parser = BaseOpt.initialize(self, parser)

		parser.add_argument('--num_test', type=int, default=100, help='number of images to generate')
		parser.add_argument('--output_dir', type=str, default='', help='path to output directory')
		parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ration of output images' )
		parser.add_argument('--path', type=str, default='./', help='path to the pretrained model')

		self.isTrain = False

		return parser
