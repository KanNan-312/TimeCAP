import json
import os


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def write_json(path, obj):
    ensure_dir(os.path.dirname(path))
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_text(path, text):
    ensure_dir(os.path.dirname(path))
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)


def read_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
