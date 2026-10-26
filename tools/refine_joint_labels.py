"""Retry incomplete pin labels jointly, without changing any cut contour."""
import argparse
from contextlib import closing,contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import os
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import shapely
from pipeline import unpack
from slicer import Settings,Part
from sections import export_sections,section_path,EXPORT_REVISION
from nesting import read_item


@contextmanager
def exclusive(job,enabled):
    if not enabled:
        yield;return
    with (job/'job.lock').open('r+b') as lock:
        if os.name=='nt':
            import msvcrt
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path)
    parser.add_argument('--preview',type=Path,required=True);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();job=args.job.resolve()
    report=json.loads((job/'assembly.json').read_text(encoding='utf-8'))
    cfg=Settings.from_saved(report['settings']);cfg._label_addresses={p['id']:p['marking'] for p in report['parts']}
    by_part={p['id']:[] for p in report['parts']}
    for r in report['connections']:
        by_part[r['part_a']].append(r);by_part[r['part_b']].append(r)
    results=[]
    with exclusive(job,args.apply),closing(sqlite3.connect(job/'progress.sqlite')) as database:
        for p in report['parts']:
            old=sum('маркировка не помещается' in n for n in p.get('marking_warnings',[]))
            if not old or not p['links']:continue
            blob=database.execute('SELECT mesh FROM parts WHERE id=?',(p['id'],)).fetchone()[0]
            part=Part(p['id'],tuple(p['cell']),unpack(blob))
            metrics,notes=export_sections(part,args.preview,report['page_mm'],cfg,by_part[p['id']])
            new=sum('маркировка не помещается' in n for n in notes)
            if new>=old:continue
            before=read_item(job/'sections_a4',p['id'],'svg').shape
            after=read_item(args.preview,p['id'],'svg').shape
            if shapely.normalize(before).wkb!=shapely.normalize(after).wkb:
                if before.symmetric_difference(after).area>1e-7 or before.hausdorff_distance(after)>1e-7:
                    raise ValueError('Контуры изменились: '+p['id'])
            results.append(dict(id=p['id'],before=old,after=new,geometry_unchanged=True))
            if args.apply:
                for extension in ('svg','dxf') if cfg.export_format=='both' else (cfg.export_format,):
                    source=section_path(args.preview,p['id'],extension)
                    target=section_path(job/'sections_a4',p['id'],extension)
                    target.parent.mkdir(parents=True,exist_ok=True);os.replace(source,target)
                p.update(metrics=metrics,section_checks=part.mesh.metadata['section_checks'],marking_warnings=notes,export_revision=EXPORT_REVISION)
                database.execute('UPDATE parts SET info=? WHERE id=?',(json.dumps(p),p['id']))
        if args.apply and results:
            database.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('complete','false'));database.commit()
    args.preview.mkdir(parents=True,exist_ok=True)
    (args.preview/'refinement.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(parts_improved=len(results),labels_recovered=sum(p['before']-p['after'] for p in results),applied=args.apply),indent=2))


if __name__=='__main__':main()
