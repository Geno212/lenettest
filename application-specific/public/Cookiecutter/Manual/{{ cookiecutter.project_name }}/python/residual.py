import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        device =torch.device("{{cookiecutter.misc_params.device.value}}" if torch.cuda.is_available() else "cpu")
        {%- set conv_idxs = [] -%}
        {%- set batch_norm_idxs = [] -%}
        {%- set _layers = cookiecutter.residual.layers.list if cookiecutter.residual.layers is mapping else cookiecutter.residual.layers -%}
        {%- for layer in _layers -%}
        {%- if layer.type == 'Conv2d' -%}
        {%- set temp = conv_idxs.append(loop.index) -%}
        {%- elif layer.type == 'BatchNorm2d' -%}
        {%- set temp = batch_norm_idxs.append(loop.index) -%}
        {%- endif -%}
        {%- endfor -%}

        {% for layer in cookiecutter.residual.layers.list %}
        self.{{ layer.name }} = nn.{{ layer.type }}(
            {%- if layer.type == 'Conv2d' and conv_idxs|length == 1 %}
            in_channels=in_channels,
            out_channels=in_channels,
            {%- elif layer.type == 'Conv2d' and loop.index == conv_idxs[0] and conv_idxs|length > 1 %}
            in_channels=in_channels,
            out_channels=out_channels,
            {%- elif layer.type == 'Conv2d' and loop.index == conv_idxs[-1] and conv_idxs|length > 1 %}
            in_channels=out_channels,
            out_channels=in_channels,
            {%- elif layer.type == 'Conv2d' %}
            in_channels=out_channels,
            out_channels=out_channels,

            {%- elif layer.type == 'BatchNorm2d' and batch_norm_idxs[0] < conv_idxs[0] %}
            num_features=in_channels,
            {%- elif layer.type == 'BatchNorm2d' and loop.index == batch_norm_idxs[-1] and conv_idxs[-1] < batch_norm_idxs[-1] %}
            num_features=in_channels,
            {%- elif layer.type == 'BatchNorm2d' %}
            num_features=out_channels,  
            {%- endif -%}

            {% for param in layer.params -%}
            {%- if param == "device" -%}
            {{param}}=device,
            {%- elif param != 'out_channels' and param != 'num_features' %}
            {{ param }}={{ layer.params[param] }},
            {%- endif %}
            {%- endfor %}
        )
        {%- endfor %}


    def forward(self, x):
        identity = x
        {% for layer in cookiecutter.residual.layers.list %}
        x = self.{{ layer.name }}(x)
        {% endfor %}
        x += identity

        return x