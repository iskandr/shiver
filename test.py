import numpy as np

from llvm import * 
from llvm.core import * 
import llvm.ee as ee 

import shiver

def test_zero_iters():
  shiver.parfor(None, niters=0)
