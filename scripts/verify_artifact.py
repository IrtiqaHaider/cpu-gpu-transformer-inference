#!/usr/bin/env python3
"""Verify immutable code/data hashes; documentation may be personalized separately."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]

def verify(root=ROOT):
    manifest=json.loads((root/'release_checks/artifact_manifest.json').read_text())
    errors=[]
    for name,digest in manifest['sha256'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.append(name+' missing or unsafe');continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            errors.append(name+' changed')
    if errors:raise ValueError('\n'.join(errors))
    return {'status':'verified','immutable_files_checked':len(manifest['sha256']),
            'scope':manifest['scope']}

if __name__=='__main__':
    try: print(json.dumps(verify(),indent=2))
    except (OSError,ValueError,KeyError) as exc:sys.exit('Artifact verification failed: '+str(exc))
