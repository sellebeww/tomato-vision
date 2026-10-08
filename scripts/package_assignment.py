"""Package the reviewed IS794 deliverables and reproducible local project."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    submission = ROOT / 'submission'
    dataset = submission / 'dataset/TomatoVision_Dataset.zip'
    validation = json.loads((submission / 'audit/deliverable_validation.json').read_text())
    assert validation['images'] == 300 and validation['report_docx_valid']
    assert dataset.is_file()
    files = set()

    def add(relative):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.add(path)

    def tree(relative, extensions):
        for path in (ROOT / relative).rglob('*'):
            if path.is_file() and path.suffix in extensions and '__pycache__' not in path.parts:
                files.add(path)

    for name in ('README.md', 'requirements.txt', 'requirements-submission.txt',
                 'requirements-lock-macos-arm64.txt', 'data/annotations.csv', 'data/sessions.csv',
                 'data/reference_review.csv', 'outputs/experiments/own_v2/protocol.json',
                 'outputs/assignment_audit/selective_validation.json',
                 'notebooks/IS794_TomatoVision_Project.ipynb',
                 'submission/IS794_Laporan_TomatoVision.docx', 'submission/README.md',
                 'submission/dataset/TomatoVision_Dataset.zip'):
        add(name)
    for folder, extensions in (
        ('src', {'.py'}), ('scripts', {'.py', '.sh'}), ('configs', {'.json'}),
        ('web', {'.html', '.css', '.js', '.png'}), ('tests', {'.py', '.mjs', '.json'}),
        ('data/prepared/7d208f30a6b6e8d7', {'.csv', '.json'}),
        ('data/reference_import', {'.jpg', '.jpeg', '.png', '.JPG', '.md'}),
        ('outputs/reference_review', {'.csv', '.json', '.md'}),
        ('outputs/experiments/own_v3_augmented', {'.keras', '.json', '.csv', '.py', '.png'}),
        ('submission/audit', {'.md', '.json'}), ('submission/figures', {'.png'}),
        ('submission/presentasi', {'.pptx'}), ('docs', {'.md'})):
        tree(folder, extensions)
    archive = submission / 'IS794-Deep Learning-ClassXX-GroupXX-Presentation Final.zip'
    if archive.exists():
        raise FileExistsError(f'Refusing to replace {archive}')
    pending = archive.with_suffix('.pending.zip')
    try:
        with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
            for path in sorted(files):
                z.write(path, str(path.relative_to(ROOT)),
                        compress_type=zipfile.ZIP_STORED if path.suffix == '.zip' else zipfile.ZIP_DEFLATED)
            z.writestr('MULAI_DI_SINI.txt',
                       'Ekstrak semua isi ZIP. Pasang requirements-submission.txt pada Python 3.11.\n'
                       'Buka notebooks/IS794_TomatoVision_Project.ipynb lalu Run All.\n'
                       'Notebook mengekstrak dataset dan memulihkan 60 sumber training secara otomatis.\n'
                       'Laporan Word: submission/IS794_Laporan_TomatoVision.docx.\n'
                       'Identitas tim dari file terpisah; ganti ClassXX/GroupXX sebelum pengumpulan.\n'
                       'Sesuai permintaan pemilik, laporan disediakan DOCX; tugas resmi meminta PDF.\n'
                       'Baca audit/KESESUAIAN_IS794.md dalam submission untuk batas kesesuaian.\n')
        with zipfile.ZipFile(pending) as z:
            assert z.testzip() is None
        pending.replace(archive)
    finally:
        pending.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with archive.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    report = {'archive': archive.name, 'files': len(files) + 1, 'bytes': archive.stat().st_size,
              'sha256': digest.hexdigest(), 'zip_integrity_checked': True,
              'identities_supplied_separately': True, 'report_format': 'docx'}
    (submission / 'audit/project_package.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
