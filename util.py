def file_str(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()
