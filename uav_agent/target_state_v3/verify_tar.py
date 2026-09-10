"""Standard-library-only, non-extracting verifier usable on the PC/WSL."""
from hashlib import sha256
import json
from pathlib import PurePosixPath
import tarfile

PROTOCOL="instance_evidence_pilot_v3"


def file_sha(path):
    digest=sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tar(path,*,expected_contract=None,expected_episode=None,expected_manifest_sha=None):
    with tarfile.open(path,"r:") as tar:
        members=tar.getmembers()
        names=[m.name for m in members]
        if len(members)>1000 or len(set(names))!=len(names) or sum(m.size for m in members)>1024**3:
            raise ValueError("oversized/duplicate pilot archive")
        for m in members:
            p=PurePosixPath(m.name)
            if not m.isfile() or p.is_absolute() or ".." in p.parts or "\\" in m.name or m.size>64*1024**2:
                raise ValueError("unsafe archive member")
        if "episode_manifest.json" not in names:
            raise ValueError("missing episode manifest")
        raw=tar.extractfile("episode_manifest.json").read()
        if expected_manifest_sha is not None and sha256(raw).hexdigest()!=expected_manifest_sha:
            raise ValueError("archive differs from retained episode")
        manifest=json.loads(raw)
        if manifest.get("protocol")!=PROTOCOL:
            raise ValueError("not a V3 pilot archive")
        if expected_contract is not None and manifest["contract"]!=expected_contract:
            raise ValueError("archive contract mismatch")
        if expected_episode is not None and manifest["episode_id"]!=expected_episode:
            raise ValueError("archive episode mismatch")
        if len(manifest["captures"])!=manifest["physical_capture_count"] or len(set(manifest["captures"]))!=len(manifest["captures"]):
            raise ValueError("invalid physical capture count")
        if set(names)!=set(manifest["sha256"])|{"episode_manifest.json"}:
            raise ValueError("archive member manifest mismatch")
        for name,expected in manifest["sha256"].items():
            digest=sha256()
            stream=tar.extractfile(name)
            for block in iter(lambda:stream.read(1024*1024),b""):
                digest.update(block)
            if digest.hexdigest()!=expected:
                raise ValueError(f"archive asset checksum mismatch: {name}")
        return manifest
