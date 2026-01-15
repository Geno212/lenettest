class flatten_validate:
    def __init__(self):
        pass

    def definition(self):
        return "Flatten", self.validate

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
        if params["start_dim"] == 0 and params["end_dim"] == -1:
            flattened = True
        return (
            layers,
            index,
            width,
            height,
            channels,
            features_after_1st_FC,
            flattened,
        )
