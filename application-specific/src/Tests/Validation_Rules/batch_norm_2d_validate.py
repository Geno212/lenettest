class batch_norm_2d_validate:
    def __init__(self):
        pass

    def definition(self):
        return "BatchNorm2d", self.validate

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
        params["num_features"] = channels
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
