import sys
from nose.tools import nottest


def run_local_functions(prefix, locals_dict = None):
  if locals_dict is None:
    last_frame = sys._getframe()
    locals_dict = last_frame.f_back.f_locals

  good = set([])
  # bad = set([])
  for k, test in locals_dict.iteritems():
    if k.startswith(prefix):
      print "Running %s..." % k
      try:
        test()
        print "\n --- %s passed\n" % k
        good.add(k)
      except:
        raise

  print "\n%d tests passed: %s\n" % (len(good), ", ".join(good))

@nottest
def run_local_tests(locals_dict = None):
  if locals_dict is None:
    last_frame = sys._getframe()
    locals_dict = last_frame.f_back.f_locals
  return run_local_functions("test_", locals_dict)