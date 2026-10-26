"""Create an improved copy using saved geometry; never reslice or redrill."""
import argparse
from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline import atomic_json,_file_hash,save_config
from slicer import Settings,run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('original',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args()
    original=args.original.resolve();output=args.output.resolve();model=args.source.resolve()
    if output==original:raise ValueError('Для копии выберите отдельную папку.')
    with closing(sqlite3.connect((original/'progress.sqlite').as_uri()+'?mode=ro',uri=True)) as source:
        meta={k:json.loads(v) for k,v in source.execute('SELECT key,value FROM meta')}
        if not meta.get('complete'):raise ValueError('Исходный расчёт ещё не завершён.')
        if _file_hash(model)!=meta['source_hash']:raise ValueError('Исходная STEP/STL-модель не совпадает по SHA-256.')
        if not output.exists():
            output.mkdir(parents=True,exist_ok=False)
            print('Copying saved geometry…',flush=True)
            with closing(sqlite3.connect(output/'progress.sqlite')) as destination:source.backup(destination)
            for name in ('assembly.json','assembly.stl','.prepared_shell.npz','REPORT.txt'):
                if (original/name).is_file():shutil.copy2(original/name,output/name)
            cfg=Settings.from_saved(meta['settings']);cfg.nest_sheets=True;cfg.sheet_gap=7.
            save_config(output/'settings.json',cfg)
            with closing(sqlite3.connect(output/'progress.sqlite')) as destination:
                with destination:
                    for key,value in [('source_original',meta['source']),('source',str(model)),('settings',asdict(cfg))]:
                        destination.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,json.dumps(value,ensure_ascii=False)))
    print('Reexporting saved parts without slicing or drilling…',flush=True)
    report=run(model,output,resume=True,log=lambda line:print(line,flush=True))
    with closing(sqlite3.connect(output/'progress.sqlite',uri=True)) as destination:
        destination.execute('ATTACH DATABASE ? AS original',((original/'progress.sqlite').as_uri()+'?mode=ro',))
        modified=destination.execute('SELECT count(*) FROM parts p JOIN original.parts o ON p.id=o.id WHERE p.mesh!=o.mesh').fetchone()[0]
        counts=destination.execute('SELECT (SELECT count(*) FROM parts),(SELECT count(*) FROM original.parts)').fetchone()
        if modified or counts[0]!=counts[1]:raise ValueError('3D-геометрия изменилась при повторном экспорте.')
    provenance=dict(original_job=str(original),source_sha256=meta['source_hash'],
                    parts_verified=counts[0],geometry_blobs_unchanged=True,modified_geometry_blobs=modified,
                    source_connections=len(json.loads((original/'assembly.json').read_text(encoding='utf-8'))['connections']),
                    result_connections=len(report['connections']),sheet_nesting=report.get('sheet_nesting'))
    atomic_json(output/'reexport_provenance.json',provenance)
    print(json.dumps(provenance,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
