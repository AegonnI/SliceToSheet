"""Build, verify and archive the portable Windows distribution."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from importlib.metadata import distributions

ROOT = Path(__file__).resolve().parent


def main():
    if sys.platform != 'win32':
        raise SystemExit('Build the Windows release on Windows x64.')
    version = (ROOT / 'VERSION').read_text(encoding='utf-8').strip()
    dist_root=ROOT/'dist'/('release-'+version)
    # Do not collect unrelated DLLs from the host's document/media tools.
    # In particular their ICU can shadow the Windows ICU used by Qt.
    environment = os.environ.copy()
    for name in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV'):
        environment.pop(name, None)
    environment['PATH'] = os.pathsep.join([str(Path(sys.executable).parent),
                                         str(Path(os.environ['SystemRoot']) / 'System32'),
                                         os.environ['SystemRoot']])
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--clean', '--noconfirm', '--distpath', str(dist_root), str(ROOT / 'SliceToSheet.spec')],
                   cwd=ROOT, env=environment, check=True)
    folder = dist_root / 'SliceToSheet'
    for name in ('README.md', 'QUICKSTART_RU.txt', 'VERSION', 'THIRD_PARTY_NOTICES.md', 'CHANGELOG.md'):
        shutil.copy2(ROOT / name, folder / name)
    shutil.copytree(ROOT / 'docs', folder / 'docs', dirs_exist_ok=True)

    # Keep all installed distribution notices, including dependencies and tools.
    licenses = folder / 'licenses'
    licenses.mkdir(exist_ok=True)
    for dist in distributions():
        for entry in dist.files or []:
            if entry.name.lower().startswith(('license', 'copying', 'notice', 'authors')):
                source = Path(dist.locate_file(entry))
                if source.is_file():
                    destination = licenses / dist.metadata['Name'] / str(entry)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    python_license = Path(sys.base_prefix) / 'LICENSE.txt'
    if python_license.is_file():
        shutil.copy2(python_license, licenses / 'PYTHON_LICENSE.txt')

    report = ROOT / 'build' / 'installation-check.json'
    report.unlink(missing_ok=True)
    environment['PATH'] = os.pathsep.join([str(Path(os.environ['SystemRoot']) / 'System32'), os.environ['SystemRoot']])
    completed = subprocess.run([str(folder / 'SliceToSheet.exe'), '--check-installation', str(report)],
                               cwd=os.environ.get('TEMP', str(ROOT)), env=environment, timeout=480)
    if completed.returncode or not report.is_file() or not json.loads(report.read_text(encoding='utf-8'))['ok']:
        raise SystemExit('Portable installation check failed. See build/installation-check.json.')
    releases = ROOT / 'releases'
    releases.mkdir(exist_ok=True)
    archive = Path(shutil.make_archive(str(releases / ('SliceToSheet-%s-Windows-x64' % version)),
                                      'zip', root_dir=dist_root, base_dir='SliceToSheet'))
    with archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    archive.with_suffix('.zip.sha256').write_text(digest + '  ' + archive.name + '\n', encoding='ascii')
    print('Verified portable release:', archive)
    print(report.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
