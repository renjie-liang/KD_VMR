import yaml

def load_yaml(filename):
    with open(filename, encoding='utf8') as fr:
        return yaml.safe_load(fr)
