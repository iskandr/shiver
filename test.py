import shiver

def test_zero_iters():
  shiver.parfor(None, niters=0)
