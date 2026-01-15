class after_comes:
    def __init__(self):
        pass

    def definition(self):
        return "-->", self.test

    def test(self, layers, layer_idx, rule_tokens) -> bool:
        # print("after_comes")
        if layers[layer_idx]["type"] != rule_tokens[0]:
            return True
        if layer_idx >= len(layers) - 1:
            return False
        for i in range(layer_idx + 1, len(layers)):
            if layers[i]["type"] == rule_tokens[2]:
                return True
        return False
