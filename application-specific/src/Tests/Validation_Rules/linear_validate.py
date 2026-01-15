class linear_validate:
    def __init__(self):
        pass

    def definition(self):
        return "Linear", self.validate

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
        if not flattened:
            flattened = True
            flatten = {
                "type": "Flatten",
                "params": {
                    "start_dim": 1,
                    "end_dim": -1,
                },
            }
            if index > 0 and layers[index - 1]["type"] == "Flatten":
                layers[index - 1]["params"] = flatten["params"]
            else:
                layers.insert(index, flatten)
                index += 1
        if features_after_1st_FC is not None:
            params["in_features"] = features_after_1st_FC
        else:
            params["in_features"] = int(channels * width * height)
        features_after_1st_FC = params["out_features"]
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
