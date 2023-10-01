import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
from IPython import embed

from .darts.visualize import plot

class Plotter():
    def __init__(self, args):
        self.args = args
    
    def plot(self, genotype, file_name, task=None):
        plot(   genotype, 
                file_name, 
                self.args,
                task)


if __name__ == '__main__':
    pass
