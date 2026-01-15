class before_comes_not:
    def __init__(self):
        pass

    def definition(self):
        return "<--x", self.test

    def test(self, layers, layer_idx, rule_tokens) -> bool:
        # print("before_comes_not")
        if layers[layer_idx]["type"] != rule_tokens[0]:
            return True
        if layer_idx <= 0:
            return True
        for i in range(layer_idx):
            if layers[i]["type"] == rule_tokens[2]:
                return False
        return True
