class ProductDict(object):
    class __ProductDict(object):
        def __init__(self):
            self.data = {}

            # Normal times Normal
            from decompose.distributions.normal import Normal
            from decompose.distributions.normalNormal import NormalNormal
            self.data[frozenset((Normal, Normal))] = NormalNormal()

            # Normal times CenNormal
            from decompose.distributions.cenNormal import CenNormal
            self.data[frozenset((Normal, CenNormal))] = NormalNormal()

            # Normal times NnNormal
            from decompose.distributions.nnNormal import NnNormal
            from decompose.distributions.normalNnNormal import NormalNnNormal
            self.data[frozenset((Normal, NnNormal))] = NormalNnNormal()

            # Normal times CenNnNormal
            from decompose.distributions.cenNnNormal import CenNnNormal
            self.data[frozenset((Normal, CenNnNormal))] = NormalNnNormal()

            # Normal times Uniform
            from decompose.distributions.uniform import Uniform
            from decompose.distributions.normalUniform import NormalUniform
            self.data[frozenset((Normal, Uniform))] = NormalUniform()

            # Normal times NnUniform
            from decompose.distributions.nnUniform import NnUniform
            from decompose.distributions.normalNnUniform import NormalNnUniform
            self.data[frozenset((Normal, NnUniform))] = NormalNnUniform()

            # Normal times Exponential
            from decompose.distributions.exponential import Exponential
            from decompose.distributions.normalExponential import NormalExponential
            self.data[frozenset((Normal, Exponential))] = NormalExponential()

            from decompose.distributions.cenLaplace import CenLaplace
            from decompose.distributions.normalLaplace import NormalLaplace
            self.data[frozenset((Normal, CenLaplace))] = NormalLaplace()

            from decompose.distributions.laplace import Laplace
            self.data[frozenset((Normal, Laplace))] = NormalLaplace()

            # Normal times CenNnFullyElasticNet
            from decompose.distributions.cenNnFullyElasticNetCond import CenNnFullyElasticNetCond
            from decompose.distributions.normalCenNnFullyElasticNetCond import NormalCenNnFullyElasticNetCond
            self.data[frozenset((Normal, CenNnFullyElasticNetCond))] = NormalCenNnFullyElasticNetCond()


        def lookup(self, d0, d1):
            from decompose.distributions.distribution import Distribution
            if not isinstance(d0, Distribution):
                raise ValueError("Both factors must be distributions")
            if not isinstance(d1, Distribution):
                raise ValueError("Both factors must be distributions")

            pdClass = self.data[frozenset((type(d0), type(d1)))]
            pd = pdClass.fromUnordered(d0, d1)
            return(pd)

    instance = None  # type: "ProductDict"

    def __init__(self):
        if not ProductDict.instance:
            ProductDict.instance = ProductDict.__ProductDict()

    def __getattr__(self, name):
        return getattr(self.instance, name)
