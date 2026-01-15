class conv_2d_validate:
    def __init__(self):
        pass

    def definition(self):
        return "Conv2d", self.validate

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
        # stride and kernel must be greater than 0
        width = (width - params["kernel_size"] + 2 * params["padding"]) // params[
            "stride"
        ] + 1
        height = (height - params["kernel_size"] + 2 * params["padding"]) // params[
            "stride"
        ] + 1
        channels = params["out_channels"]
        layers[index]["params"] = params
        return (
            layers,
            index,
            width,
            height,
            channels,
            features_after_1st_FC,
            flattened,
        )
