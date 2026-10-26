"""Independently check every packed SVG/DXF against source contours and layout."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import shapely
from shapely import affinity
from tools.audit_exports import parse_svg,check_dxf,inspect


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('job',type=Path)
    parser.add_argument('source_audit',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    layout=json.loads((args.job/'Sheets'/'layout.json').read_text(encoding='utf-8'))
    summary=layout['summary'];page=summary['page_mm'];placed=layout['placements']
    assembly=json.loads((args.job/'assembly.json').read_text(encoding='utf-8'))
    all_ids={p['id'] for p in assembly['parts']}
    counts=Counter(p['id'] for p in placed)
    excluded=set(summary['excluded_empty_sections'])
    if set(counts)&excluded or any(n!=1 for n in counts.values()) or set(counts)|excluded!=all_ids:
        raise ValueError('Missing, repeated or incorrectly excluded source parts')
    source_db=sqlite3.connect(args.source_audit/'audit.sqlite')
    sources={pid:(json.loads(data),shapely.from_wkb(blob)) for pid,data,blob in
             source_db.execute('SELECT id,data,geometry FROM parts')}
    source_db.close()
    if any(not sources[pid][1].is_empty for pid in excluded):raise ValueError('Excluded nonempty part')
    args.output.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(args.output/'audit.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS sheets (number INTEGER PRIMARY KEY, signature TEXT, result TEXT)')
    started=time.monotonic()
    for number in range(1,summary['sheets']+1):
        svg=args.job/'Sheets'/'SVG'/f'Sheet_{number:05d}.svg'
        dxf=args.job/'Sheets'/'DXF'/f'Sheet_{number:05d}.dxf'
        placements=[p for p in placed if p['sheet']==number]
        digest=hashlib.sha256(svg.read_bytes()+dxf.read_bytes()+json.dumps(placements).encode()).hexdigest()
        if db.execute('SELECT 1 FROM sheets WHERE number=? AND signature=?',(number,digest)).fetchone():continue
        errors=[]
        data=parse_svg(svg)
        metrics,_=inspect(data)
        cad=check_dxf(dxf,data)
        if data['scaled'] or data['reflected']:errors.append('scale_or_reflection')
        if [data['width'],data['height']]!=page:errors.append('page_size')
        if cad['issues'] or cad['labels_without_fit']:errors.append('svg_dxf_disagreement')
        if metrics['invalid_contours'] or metrics['open_cut_mm']>1e-7:errors.append('invalid_contours')
        if metrics['outside_page']:errors.append('outside_page')
        if metrics['text_outside']:errors.append('text_outside_material')
        if any(t['angle']<-90-1e-6 or t['angle']>90+1e-6 for t in data['texts']):errors.append('upside_down_text')
        root=ET.parse(svg).getroot()
        if [c.get('data-part') for c in root if c.get('data-part')]!=[p['id'] for p in placements]:
            errors.append('part_group_index')
        expected=[];shapes=[];source_text=[]
        edge=math.inf;gap=math.inf
        for p in placements:
            original,shape=sources[p['id']]
            angle=math.radians(p['rotation']);cs,ss=math.cos(angle),math.sin(angle)
            rotation=np.array([[cs,-ss],[ss,cs]])
            for c in original['contours']:
                expected.append(np.asarray(c['points'])@rotation.T+[p['tx'],p['ty']])
            source_text.extend(t['text'] for t in original['texts'])
            q=affinity.translate(affinity.rotate(shape,p['rotation'],origin=(0,0)),p['tx'],p['ty'])
            x0,y0,x1,y1=q.bounds;edge=min(edge,x0,y0,page[0]-x1,page[1]-y1)
            if shapes:
                if np.any(shapely.intersects(shapes,q)):errors.append('overlap')
                gap=min(gap,float(np.min(shapely.distance(shapes,q))))
            shapes.append(q)
        coordinate_error=0.
        if len(expected)!=len(data['contours']):errors.append('source_contour_count')
        for points,contour in zip(expected,data['contours']):
            actual=np.array(contour['points'])
            if actual.shape!=points.shape:errors.append('source_vertex_count');continue
            coordinate_error=max(coordinate_error,float(np.max(np.abs(actual-points))))
        if coordinate_error>1e-7:errors.append('source_contour_transform')
        if source_text!=[t['text'] for t in data['texts']]:errors.append('source_labels')
        if edge<summary['edge_mm']-1e-5:errors.append('edge_clearance')
        if gap<summary['required_gap_mm']-1e-5:errors.append('part_clearance')
        metadata=json.loads(next(n.text for n in root if n.tag.rsplit('}',1)[-1]=='metadata'))
        if abs(metadata['cut_mm']-metrics['total_cut_mm'])>.001:errors.append('perimeter_metadata')
        if abs(metadata['net_area_mm2']-metrics['area_mm2'])>.001:errors.append('area_metadata')
        result=dict(number=number,parts=len(placements),errors=sorted(set(errors)),
                    min_edge_mm=edge,min_gap_mm=None if math.isinf(gap) else gap,
                    coordinate_error_mm=coordinate_error,dxf=cad,
                    text_outside=metrics['text_outside'],cut_mm=metrics['total_cut_mm'])
        db.execute('INSERT OR REPLACE INTO sheets VALUES (?,?,?)',(number,digest,json.dumps(result)))
        if number%25==0:
            db.commit();print(f'Checked {number}/{summary["sheets"]} sheets; {time.monotonic()-started:.1f}s',flush=True)
    db.commit()
    records=[json.loads(r[0]) for r in db.execute('SELECT result FROM sheets WHERE number<=? ORDER BY number',(summary['sheets'],))]
    actual={ext:len(list((args.job/'Sheets'/ext.upper()).glob('*.'+ext))) for ext in ('svg','dxf')}
    result=dict(sheets=len(records),expected_sheets=summary['sheets'],files=actual,
                placed_parts=len(placed),excluded_empty_sections=len(excluded),source_parts=len(all_ids),
                errors=[r for r in records if r['errors']],
                minimum_edge_mm=min(r['min_edge_mm'] for r in records),
                minimum_gap_mm=min(r['min_gap_mm'] for r in records if r['min_gap_mm'] is not None),
                maximum_coordinate_error_mm=max(r['coordinate_error_mm'] for r in records),
                total_cut_mm=math.fsum(r['cut_mm'] for r in records),
                elapsed_seconds=time.monotonic()-started)
    result['ok']=not result['errors'] and result['sheets']==result['expected_sheets'] and all(n==result['sheets'] for n in actual.values())
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    db.close()
    if not result['ok']:raise SystemExit(1)


if __name__=='__main__':main()
