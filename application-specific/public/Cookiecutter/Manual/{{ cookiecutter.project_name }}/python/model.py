# import the necessary packages
import torch
from torch import nn
from python.residual import ResidualBlock

class CNN(nn.Module):
    def __init__(self):     
        device =torch.device("{{cookiecutter.misc_params.device.value}}" if torch.cuda.is_available() else "cpu")
 
        super(CNN,self).__init__()
        {%- set _layers = cookiecutter.layers.list if cookiecutter.layers is mapping else cookiecutter.layers -%}
        {%- for layer in _layers -%}
        {%- if layer.type == 'Residual_Block' -%}
        self.{{layer.name}} = ResidualBlock( {% for param in layer.params %}
            {{param}} = {{layer.params[param]}},
        {%- endfor %}
        )
        {%- else %}
        self.{{layer.name}} = nn.{{layer.type}}( {% for param in layer.params %}
        {%- if param == "device" %}
            {{param}}=device,
            {%- else %}
            {{param}}={{layer.params[param]}},
        {%- endif %}
        {%- endfor %}
        )
        {% endif %}
        {% endfor %}

    def forward(self, x):
        {% set _layers = cookiecutter.layers.list if cookiecutter.layers is mapping else cookiecutter.layers %}
        {% for layer in _layers -%}
        x = self.{{layer.name}}(x)
        {% endfor %}
        return x
