"""Git checkout or verified provider-neutral package identity."""
import json
import subprocess
from pathlib import Path
from firm_data import sha256


def repository_revision(root=Path('.')):
    root=root.resolve()
    package=root/'firm_gpu_package_manifest.json'
    if package.exists():
        manifest=json.loads(package.read_text())
        for asset in manifest['assets']:
            path=root/asset['path']
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root) or sha256(path)!=asset['sha256']:
                raise ValueError('Package payload changed: '+asset['path'])
        return manifest['git_sha']
    return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
