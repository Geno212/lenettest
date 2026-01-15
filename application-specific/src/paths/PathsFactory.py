import os


# import  src.paths.SystemPaths as SystemPaths
class PathFactory:
    def get_paths(self, model_name):
        publicdir = os.path.normpath(os.path.join(self.basedir, "../../public/"))
        if model_name.lower().startswith("yolo"):
            # If the model name starts with "yolo", use YOLOX paths
            return (
                os.path.normpath(os.path.join(publicdir, "jinja_templates/YOLOX")),
                os.path.normpath(os.path.join(publicdir, "./Cookiecutter/YOLOX")),
                os.path.normpath(
                    os.path.join(publicdir, "./Cookiecutter/YOLOX/cookiecutter.json"),
                ),
            )

        # Otherwise, use default paths
        return (
            os.path.normpath(os.path.join(publicdir, "jinja_templates/Transfer")),
            os.path.normpath(os.path.join(publicdir, "./Cookiecutter/Pretrained")),
            os.path.normpath(
                os.path.join(publicdir, "./Cookiecutter/Pretrained/cookiecutter.json"),
            ),
        )
