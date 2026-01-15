class before_comes_not_directly:
    def __init__(self):
        pass

    def definition(self):
        return "<-x", self.test

    def test(self, layers, layer_idx, rule_tokens) -> bool:
        # print("before_comes_not_directly")
        if layers[layer_idx]["type"] != rule_tokens[0]:
            return True
        if layer_idx <= 0:
            return True
        if layers[layer_idx - 1]["type"] != rule_tokens[2]:
            return True
        return False
