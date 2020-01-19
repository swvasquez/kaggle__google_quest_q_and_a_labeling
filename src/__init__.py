import pathlib


def project_paths():
    root_path = pathlib.Path(__file__).parents[1]
    config_path = root_path / 'config.yaml'
    path_dict = {'root': root_path, 'config': config_path}
    return path_dict
