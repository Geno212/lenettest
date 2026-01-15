class between_comes:
    def __init__(self):
        pass

    def definition(self):
        return "<-->", self.test

    def test(self, layers, layer_idx, rule_tokens) -> bool:
        # print("between_comes")
        layer_1, layer_2 = rule_tokens[0][1:-1].split(",")
        if layers[layer_idx]["type"] != layer_1:
            return True
        if layer_idx >= len(layers) - 1:
            return True
        for i in range(layer_idx + 1, len(layers)):
            if layers[i]["type"] == layer_2:
                return False
            if layers[i]["type"] == rule_tokens[2]:
                return True
        return True
