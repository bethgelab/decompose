from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf

from decompose.distributions.distribution import DrawType, UpdateType
from decompose.distributions.distribution import Distribution


class Product(metaclass=ABCMeta):

    @abstractmethod
    def fromUnordered(self, d0: Distribution,
                      d1: Distribution) -> Distribution:
        ...

    def productParams(self, d0: Distribution,
                      d1: Distribution):
        name = self.name(d0, d1)
        drawType = self.drawType(d0, d1)
        updateType = self.updateType(d0, d1)
        params = {'name': name,
                  'drawType': drawType,
                  'updateType': updateType,
                  'persistent': False}
        return(params)

    def name(self, d0: Distribution, d1: Distribution) -> str:
        return(d0.name + "_" + d1.name)

    def homogenous(self, d0: Distribution, d1: Distribution) -> bool:
        return(np.logical_and(d0.homogenous, d1.homogenous))

    def drawType(self, d0: Distribution, d1: Distribution) -> DrawType:
        if d0.drawType == d1.drawType:
            return(d0.drawType)
        else:
            raise ValueError("DrawType does not match")

    def updateType(self, d0: Distribution,
                   d1: Distribution) -> UpdateType:
        if d0.updateType == d1.updateType:
            return(d0.updateType)
        else:
            raise ValueError("UpdateType does not match")
