import importlib


# find the loss definition by name, for example TripletLoss (TripletLoss.py)
def find_loss_def(loss):
    module_name = 'loss.{}'.format(loss)
    module = importlib.import_module(module_name)
    return getattr(module, "Loss")
