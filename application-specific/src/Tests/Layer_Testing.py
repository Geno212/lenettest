class LayerTesting:
    def __init__(self) -> None:
        pass

    def format_string_layer(self, layer_type, params_names, params_value_widgets):
        tempstring = f"torch.nn.{layer_type}("

        if layer_type == "Conv2d" or layer_type == "ConvTranspose2d":
            tempstring += "in_channels = 15,"
        elif layer_type == "BatchNorm2d":
            tempstring += "num_features = 3,"
        # elif layer_type == 'MaxPool2d' or layer_type == 'AvgPool2d':
        #         width = (width - params['kernel_size']) // params['stride'] + 1
        #         height = (height - params['kernel_size']
        #                   ) // params['stride'] + 1
        # elif layer_type == 'Linear':
        #         if not flattened:
        #             flattened = True
        #             flatten = {
        #                 'type': 'Flatten',
        #                 'params': {
        #                     'start_dim': 0,
        #                     'end_dim': -1
        #                 }
        #             }
        #             if i > 0 and architecture['layers'][i-1]['type'] == 'Flatten':
        #                 architecture['layers'][i -1]['params'] = flatten['params']
        #             else:
        #                 self.create_layer_node(flatten, i)
        #                 self.add_layer_name(flatten, layer_freqs)
        #                 i += 1
        #         if features_after_1st_FC is not None:
        #             params['in_features'] = features_after_1st_FC
        #         else:
        #             params['in_features'] = int(channels * width * height)
        #         features_after_1st_FC = params['out_features']
        # elif layer_type == 'Flatten':
        #         if params['start_dim'] == 0 and params['end_dim'] == -1:
        #             flattened = True
        #     i += 1

        for i in range(len(params_value_widgets)):
            param_value = self.get_widget_data(params_value_widgets[i])
            if param_value != "":
                tempstring += f"{params_names[i]}={param_value},"

        tempstring += ")"
        print(tempstring)
        return tempstring

    def test_layer(self, layer_type, params_names, params_value_widgets):
        dlg = self.create_dialogue_error()
        tempstring = self.format_string_layer(
            layer_type,
            params_names,
            params_value_widgets,
        )
        try:
            exec(tempstring)
            return True
        except Exception as e:
            print("error", e)
            dlg.setText(str(e))
            dlg.exec()
            return False
