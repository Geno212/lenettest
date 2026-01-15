class max_pool_2d_validate:
    def __init__(self):
        pass

    def definition(self):
        return "MaxPool2d", self.validate

    def validate(
        self,
        layers,
        index,
        width,
        height,
        channels,
        features_after_1st_FC,
        flattened,
    ):
        params = layers[index]["params"]
        width = (width - params["kernel_size"] + 2 * params["padding"]) // params[
            "stride"
        ] + 1
        height = (height - params["kernel_size"] + 2 * params["padding"]) // params[
            "stride"
        ] + 1
        return (
            layers,
            index,
            width,
            height,
            channels,
            features_after_1st_FC,
            flattened,
        )
