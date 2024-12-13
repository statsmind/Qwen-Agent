import hashlib
import os
import pathlib


class Cache:
    def __init__(self, prefix: str):
        self.cache_dir = pathlib.Path.home().joinpath(f".cache/{prefix}")
        os.makedirs(self.cache_dir, exist_ok=True)

    def put(self, key: str, content: str):
        hash_key = hashlib.md5(key.encode('utf-8')).hexdigest()
        hash_file = self.cache_dir.joinpath(f'{hash_key[0]}/{hash_key[1]}/{hash_key}')
        hash_file.parent.mkdir(parents=True, exist_ok=True)

        with open(hash_file, 'w', encoding='utf') as fp:
            fp.write(content)

    def get(self, key: str) -> str:
        hash_key = hashlib.md5(key.encode('utf-8')).hexdigest()
        hash_file = self.cache_dir.joinpath(f'{hash_key[0]}/{hash_key[1]}/{hash_key}')

        if hash_file.exists() and hash_file.stat().st_size > 0:
            return hash_file.open('r', encoding='utf-8').read()
        else:
            return None