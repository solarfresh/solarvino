from .type import (Point, FacialLandmark5)


class OutputParserBase:
    @staticmethod
    def find_layer_by_name(name, layers):
        suitable_layers = [layer_name for layer_name in layers if name in layer_name]
        if not suitable_layers:
            raise ValueError('Suitable layer for "{}" output is not found'.format(name))

        if len(suitable_layers) > 1:
            raise ValueError('More than 1 layer matched to "{}" output'.format(name))

        return suitable_layers[0]


class FacialLandmark5Parser(OutputParserBase):
    def __init__(self, layers):
        if len(layers) != 1:
            raise ValueError('Network must have only one output.')
        self.output_name, output_data = next(iter(layers.items()))

    def __call__(self, outputs):
        kpts = outputs[self.output_name][0]
        kpts = kpts.reshape((-1, 2))
        points = [Point(kpt[0], kpt[1]) for kpt in kpts]
        return FacialLandmark5(points=points)
