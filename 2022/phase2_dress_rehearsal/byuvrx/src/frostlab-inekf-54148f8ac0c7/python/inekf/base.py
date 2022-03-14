from . import _inekf
import inspect

########################### Measurement Model ##############################
# figure this one out
class _meta_InEKF(type):
    def __getitem__(cls,key):
        # make instance if it's not one
        if inspect.isclass(key):
            key = key()

        # Parse name
        group_name = key.__class__.__mro__[-3].__name__
        name = "InEKF_" + group_name.split('_',1)[1]

        return InEKF(getattr(_inekf, name), key)

# Wrapper class since we need to set process model manually 
# (isn't templated for custom classes)
class InEKF(metaclass=_meta_InEKF):
    def __init__(self, base, pModel):
        # save for later
        self.base = base
        self.pModel = pModel

    # This is secretly used as our init function
    def __call__(self, *args, **kwargs):
        # initialize base
        self.base = self.base(*args, **kwargs)
        # initialize process model
        self.base.pModel = self.pModel

        return self

    def Predict(self, *args, **kwargs):
        return self.base.Predict(*args, **kwargs)

    def Update(self, *args, **kwargs):
        return self.base.Update(*args, **kwargs)

    def addMeasureModel(self, *args, **kwargs):
        return self.base.addMeasureModel(*args, **kwargs)

    @property
    def state(self):
        return self.base.state

############################ Measurement Model ##############################
class _meta_Measure(type):
    def __getitem__(cls,key):
        # Parse name
        group_name = key.__name__
        name = "MeasureModel_" + group_name

        return getattr(_inekf, name)

class MeasureModel(metaclass=_meta_Measure):
    pass

############################ Generic Measure Model ##############################
class _meta_GenericMeasure(type):
    def __getitem__(cls,key):
        # Parse name
        group_name = key.__name__
        name = "GenericMeasureModel_" + group_name

        return getattr(_inekf, name)

class GenericMeasureModel(metaclass=_meta_GenericMeasure):
    pass

############################ Process Model ##############################
class _meta_Process(type):
    def __getitem__(cls,key):
        # if only one thing was passed to us
        if not isinstance(key, tuple):
            key = (key, key)

        # Parse name
        if isinstance(key, tuple) and len(key) == 2:
            name = "ProcessModel_" + key[0].__name__ + "_"
            if isinstance(key[1], str):
                name += key[1]
            elif isinstance(key[1], int):
                if key[1] == -1:
                    name += "Vec" + "D"
                else:
                    name += "Vec" + str(key[1])
            else:
                name += key[1].__name__


            return getattr(_inekf, name)

class ProcessModel(metaclass=_meta_Process):
    pass