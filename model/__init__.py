import importlib


# find the model definition by name, for example SimeseResNet (SiameseResNet.py)
def find_model_def(model):
    module_name = 'model.{}'.format(model)
    module = importlib.import_module(module_name)
    return getattr(module, "SiameseResNet")
