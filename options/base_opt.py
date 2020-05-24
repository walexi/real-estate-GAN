import argparse
import os
import torch
from libs import model
from libs import compute


class BaseOpt():
	def __init__(self):
		self.initialized = True

	def initialize(self, parser):

		parser.add_argument('--dataroot', default='./', help='path to dataset folder')
		parser.add_argument('--checkpoints_dir', default='', help='path where saved models will be saved')

		parser.add_argument('--input_nc', default=3, help='number of input image channels, 3 for RGB, 1 for grayscale')
		parser.add_argument('--ouput_nc', default=3, help='number of output image channels, 3 for RGB, 1 for grayscale')
		parser.add_argument('--batch_size', default=1, help='number of batches')
		parser.add_argument('--verbose', action='store_true', help='if specified, print debugging information')

		self.initialized = True

		return parser
