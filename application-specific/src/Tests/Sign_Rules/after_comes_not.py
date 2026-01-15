class after_comes_not:
    def __init__(self):
        pass

    def definition(self):
        return "-->x", self.test

    def test(self, layers, layer_idx, rule_tokens) -> bool:
        # print("after_comes_not")
        if layers[layer_idx]["type"] != rule_tokens[0]:
            return True
        if layer_idx >= len(layers) - 1:
            return True
        for i in range(layer_idx + 1, len(layers)):
            if layers[i]["type"] == rule_tokens[2]:
                return False
        return True
