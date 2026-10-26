"""Read every saved 3-D part and compare native middle sections to old drawings."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import manifold3d
from pipeline import unpack
from slicer import Part,as_solid
from sections import section_contours,part_metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=sqlite3.connect((args.job/'progress.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    if source.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:raise ValueError('SQLite integrity check failed')
    meta={k:json.loads(v) for k,v in source.execute('SELECT key,value FROM meta')}
    report=json.loads((args.job/'assembly.json').read_text(encoding='utf-8'))
    output=sqlite3.connect(args.output/'geometry.sqlite')
    output.execute('CREATE TABLE IF NOT EXISTS parts (id TEXT PRIMARY KEY,data TEXT)')
    ids=[r[0] for r in source.execute('SELECT id FROM parts ORDER BY layer,id')]
    if set(ids)!={p['id'] for p in report['parts']}:raise ValueError('Part index mismatch')
    start=time.monotonic()
    for i,pid in enumerate(ids):
        if output.execute('SELECT 1 FROM parts WHERE id=?',(pid,)).fetchone():continue
        cell,blob=source.execute('SELECT cell,mesh FROM parts WHERE id=?',(pid,)).fetchone()
        try:
            mesh=unpack(blob);solid=as_solid(mesh)
            status=str(solid.status())
            part=Part(pid,tuple(json.loads(cell)),mesh)
            _,contours=section_contours(part)
            data=dict(watertight=bool(mesh.is_watertight),winding_consistent=bool(mesh.is_winding_consistent),
                finite=bool(np.isfinite(mesh.vertices).all()),native_status=status,
                volume_mm3=float(mesh.volume),triangles=len(mesh.faces),bounds=mesh.bounds.tolist(),
                metrics=part_metrics(part,contours),section_checks=mesh.metadata.get('section_checks',{}))
        except Exception as error:data=dict(error=repr(error))
        output.execute('INSERT INTO parts VALUES (?,?)',(pid,json.dumps(data)))
        if i%100==0:
            output.commit();print('3D checked %d/%d; %.1fs'%(i+1,len(ids),time.monotonic()-start),flush=True)
    output.commit()
    rows=[(pid,json.loads(data)) for pid,data in output.execute('SELECT id,data FROM parts')]
    good=[(pid,d) for pid,d in rows if 'error' not in d]
    summary=dict(parts=len(rows),database_complete=meta.get('complete'),
        invalid_parts=[pid for pid,d in rows if 'error' in d or not d.get('watertight') or not d.get('winding_consistent') or not d.get('finite') or d.get('native_status')!='Error.NoError'],
        native_section_islands=sum(d['section_checks']['material_islands']>1 for _,d in good),
        microcontour_parts=sum(bool(d['section_checks']['omitted_microcontours']) for _,d in good),
        omitted_microcontours=sum(len(d['section_checks']['omitted_microcontours']) for _,d in good),
        total_area_change_mm2=sum(d['section_checks']['area_change_mm2'] for _,d in good),
        native_total_cut_mm=sum(d['metrics']['total_cut_mm'] for _,d in good),
        volume_after_holes_mm3=sum(d['volume_mm3'] for _,d in good),
        saved_volume_after_holes_mm3=report['volume_after_holes'],
        connections=len(report['connections']),connected_groups=len(report['connected_groups']),
        parts_without_connections=sum(not p['links'] for p in report['parts']),
        shell=report.get('shell'),elapsed_seconds=time.monotonic()-start)
    (args.output/'geometry_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    source.close();output.close()


if __name__=='__main__':main()
