class before_comes_directly:
    def __init__(self):
        pass

    def definition(self):
        return "<-", self.test

    def test(self, layers, layer_idx, rule_tokens) -> bool:
        # print("before_comes_directly")
        if layers[layer_idx]["type"] != rule_tokens[0]:
            return True
        if layer_idx <= 0:
            return False
        if layers[layer_idx - 1]["type"] == rule_tokens[2]:
            return True
        return False
