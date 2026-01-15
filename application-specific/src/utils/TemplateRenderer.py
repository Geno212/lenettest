import os
import traceback

from jinja2 import Environment, FileSystemLoader


class JinjaTools:
    def convert_jinja_to_code(
        self,
        template_relative_path: str,
        output_relative_path: str,
        parameters: dict,
    ):
        # get the template path
        stack = traceback.extract_stack()
        base_dir = os.path.dirname(stack[-2].filename)
        template_absolute_path = os.path.normpath(
            os.path.join(base_dir, template_relative_path),
        )
        template_filename = os.path.basename(template_absolute_path)
        template_dir = os.path.dirname(template_absolute_path)

        # define the environment and get the template
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template(template_filename)

        # render the template using the defined parameters
        train_code = template.render(parameters)

        # get the output path
        output_absolute_path = os.path.normpath(
            os.path.join(base_dir, output_relative_path),
        )

        # save the rendered template as a .py file
        with open(output_absolute_path, "w") as f:
            f.write(train_code)
