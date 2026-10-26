"""Read every SVG/DXF pair without changing the source; save a resumable audit."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import ezdxf

NUMBER = r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'


def numbers(value):
    return [float(x) for x in re.findall(NUMBER, value)]


def parse_svg(path):
    root = ET.fromstring(path.read_bytes())
    width, height = float(root.get('width').removesuffix('mm')), float(root.get('height').removesuffix('mm'))
    data = dict(width=width, height=height, contours=[], texts=[], leaders=[], scaled=False, reflected=False)
    def transform_matrix(value):
        result=np.eye(3)
        for name,args in re.findall(r'([A-Za-z]+)\(([^)]*)\)',value):
            v=numbers(args);m=np.eye(3)
            if name=='translate':m[:2,2]=[v[0],v[1] if len(v)>1 else 0.]
            elif name=='matrix':
                a,b,c,d,e,f=v;m=np.array([[a,c,e],[b,d,f],[0,0,1.]])
            elif name=='rotate':
                angle=math.radians(v[0]);cs,ss=math.cos(angle),math.sin(angle)
                m[:2,:2]=[[cs,-ss],[ss,cs]]
                if len(v)==3:
                    center=np.array(v[1:]);m[:2,2]=center-m[:2,:2]@center
            elif name=='scale':m[0,0]=v[0];m[1,1]=v[1] if len(v)>1 else v[0]
            else:raise ValueError('Unsupported source transform: '+name)
            result=result@m
        return result
    def mapped(points,m):
        value=np.column_stack([points,np.ones(len(points))])@m.T
        value[:,1]=height-value[:,1]
        return value[:,:2]
    def walk(node,parent):
        m=parent@transform_matrix(node.get('transform',''))
        linear=m[:2,:2]
        if not np.allclose(linear.T@linear,np.eye(2),rtol=0,atol=1e-8):data['scaled']=True
        if np.linalg.det(linear)<0:data['reflected']=True
        tag=node.tag.rsplit('}',1)[-1]
        if tag=='polyline':
            points=mapped(np.array(numbers(node.get('points'))).reshape(-1,2),m)
            closed=len(points)>2 and np.linalg.norm(points[0]-points[-1])<1e-6
            data['contours'].append(dict(points=points[:-1].tolist() if closed else points.tolist(),closed=bool(closed)))
        elif tag=='text':
            if node.get('data-layer')=='CHECK_REQUIRED':return
            point=mapped(np.array([[float(node.get('x')),float(node.get('y'))]]),m)[0]
            angle=-math.degrees(math.atan2(linear[1,0],linear[0,0]))
            data['texts'].append(dict(text=''.join(node.itertext()),x=float(point[0]),y=float(point[1]),angle=angle,
                height=float(node.get('font-size'))*float(np.linalg.norm(linear[:,1])),
                width=float(node.get('textLength','0'))*float(np.linalg.norm(linear[:,0]))))
        elif tag=='path':
            values=numbers(node.get('d',''))
            if len(values)==4:
                points=mapped(np.array(values).reshape(2,2),m)
                data['leaders'].append(points.ravel().tolist())
        elif tag=='line':
            points=mapped(np.array([[float(node.get('x1')),float(node.get('y1'))],[float(node.get('x2')),float(node.get('y2'))]]),m)
            data['leaders'].append(points.ravel().tolist())
        for child in node:walk(child,m)
    walk(root,np.eye(3))
    return data


def inspect(data):
    import shapely
    from shapely.geometry import Polygon, LineString, box
    records=[]; valid=[]; total=0.; open_length=0.; aligned=0; short=0; segments=0
    for c in data['contours']:
        p=np.asarray(c['points'])
        delta=np.diff(np.vstack([p,p[0]]) if c['closed'] else p,axis=0)
        lengths=np.linalg.norm(delta,axis=1)
        length=float(lengths.sum()); total+=length
        segments+=len(lengths)
        short+=int(np.count_nonzero((lengths>.25)&(lengths<3.1)))
        aligned+=int(np.count_nonzero((lengths>.25)&(lengths<3.1)&(np.min(np.abs(delta),axis=1)<1e-5)))
        if not c['closed']:
            open_length+=length
            records.append(dict(closed=False,length=length,area=0,valid=False,reason='open'))
            continue
        poly=Polygon(p)
        good=poly.is_valid
        repaired=shapely.make_valid(poly) if not good else poly
        if not repaired.is_empty:
            valid.append(repaired)
        rectangle=poly.minimum_rotated_rectangle
        coords=np.asarray(rectangle.exterior.coords) if rectangle.geom_type=='Polygon' else np.empty((0,2))
        thickness=float(np.linalg.norm(np.diff(coords,axis=0),axis=1).min()) if len(coords) else 0
        records.append(dict(closed=True,length=length,area=float(poly.area),valid=good,
            reason='' if good else shapely.is_valid_reason(poly),min_width=thickness))
    from sections import even_odd
    material=even_odd(valid)
    if not material.is_valid:
        material=shapely.make_valid(material)
    polygons=[g for g in shapely.get_parts(material) if g.geom_type=='Polygon']
    if material.geom_type=='Polygon':polygons=[material]
    text_outside=[]
    for t in data['texts']:
        # Conservative rectangle used by the source label placer.
        from shapely import affinity
        rect=affinity.rotate(box(t['x'],t['y']-t['height']*.3,t['x']+t['width'],t['y']+t['height']*1.2),t.get('angle',0),origin=(t['x'],t['y']))
        outside=float(rect.difference(material.buffer(.03)).area)
        if outside>.05:
            text_outside.append(dict(text=t['text'],outside_mm2=outside))
    points=np.vstack([c['points'] for c in data['contours']]) if data['contours'] else np.zeros((1,2))
    return dict(contours=records,total_cut_mm=total,open_cut_mm=open_length,
        area_mm2=float(material.area),islands=len(polygons),island_areas_mm2=sorted((float(p.area) for p in polygons),reverse=True),
        invalid_contours=sum(c['closed'] and not c['valid'] for c in records),
        thin_contours=sum(c['closed'] and (c['area']<.01 or c['min_width']<.01) for c in records),
        short_axis_segments=aligned,short_segments=short,segments=segments,
        bounds=np.r_[points.min(axis=0),points.max(axis=0)].tolist(),
        outside_page=bool(np.any(points.min(axis=0)<-.0001) or np.any(points.max(axis=0)>[data['width']+.0001,data['height']+.0001])),
        text_outside=text_outside,labels=len(data['texts']),leaders=len(data['leaders'])), material


def check_dxf(path,data):
    doc=ezdxf.readfile(path)
    issues=[]
    if doc.units!=4:issues.append('units_not_mm')
    audit=doc.audit()
    if audit.has_errors:issues.append('dxf_audit_error')
    cut=[e for e in doc.modelspace() if e.dxf.layer=='CONTOURS']
    if len(cut)!=len(data['contours']):issues.append('contour_count')
    maximum=0.
    for entity, contour in zip(cut,data['contours']):
        actual=np.asarray([(v[0],v[1]) for v in entity.get_points()])
        expected=np.asarray(contour['points'])
        if actual.shape!=expected.shape:issues.append('vertex_count');continue
        maximum=max(maximum,float(np.max(np.abs(actual-expected))))
        if entity.closed!=contour['closed']:issues.append('closure_mismatch')
    if maximum>.00011:issues.append('contour_coordinates')
    text=[e for e in doc.modelspace() if e.dxftype()=='TEXT' and e.dxf.layer=='LABELS']
    if [e.dxf.text for e in text]!=[t['text'] for t in data['texts']]:issues.append('text_mismatch')
    baseline_error=0.;fit_error=0.
    for entity,label in zip(text,data['texts']):
        point=np.array([label['x'],label['y']])
        baseline_error=max(baseline_error,float(np.linalg.norm(np.array(tuple(entity.dxf.insert)[:2])-point)))
        if entity.dxf.halign==5:
            angle=math.radians(label.get('angle',0))
            end=point+label['width']*np.array([math.cos(angle),math.sin(angle)])
            fit_error=max(fit_error,float(np.linalg.norm(np.array(tuple(entity.dxf.align_point)[:2])-end)))
    if baseline_error>.00011:issues.append('label_position_mismatch')
    if fit_error>.00011:issues.append('label_fit_mismatch')
    return dict(issues=sorted(set(issues)),coordinate_error_mm=maximum,
                label_baseline_error_mm=baseline_error,label_fit_error_mm=fit_error,
                labels_without_fit=sum(e.dxf.halign!=5 for e in text))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--geometry-deps',type=Path)
    parser.add_argument('--only-json',type=Path,help='Recheck selected IDs in an existing complete audit')
    args=parser.parse_args()
    if args.geometry_deps:sys.path.append(str(args.geometry_deps.resolve()))
    args.output.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(args.output/'audit.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS parts (id TEXT PRIMARY KEY,data TEXT,metrics TEXT,geometry BLOB,sha TEXT)')
    if 'dxf_sha' not in [r[1] for r in db.execute('PRAGMA table_info(parts)')]:
        db.execute('ALTER TABLE parts ADD COLUMN dxf_sha TEXT')
    missing={}
    report=(args.source.parent/'REPORT.txt').read_text(encoding='utf-8')
    for line in report.splitlines():
        match=re.match(r'(L\d+_Y\d+_X\d+_C\d+): маркировка не помещается на сечении: (.*)',line)
        if match:missing.setdefault(match[1],[]).append(match[2])
    started=time.monotonic()
    files=sorted(args.source.rglob('*.svg'))
    review=args.source.parent/'Inspection'/'Rejected'/'SVG'
    if review.is_dir():files+=sorted(review.rglob('*.svg'))
    selected=None
    if args.only_json:
        selection=json.loads(args.only_json.read_text(encoding='utf-8'))
        selected={row['id'] if isinstance(row,dict) else row for row in selection}
    for index,path in enumerate(files):
        if selected is not None and path.stem not in selected:continue
        dxf=(path.parent.parent.parent/'DXF'/path.parent.name/(path.stem+'.dxf')) if path.parent.parent.name=='SVG' else path.with_suffix('.dxf')
        svg_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        dxf_sha=hashlib.sha256(dxf.read_bytes()).hexdigest() if dxf.is_file() else ''
        previous=db.execute('SELECT metrics FROM parts WHERE id=? AND sha=? AND dxf_sha=?',(path.stem,svg_sha,dxf_sha)).fetchone()
        if previous and json.loads(previous[0]).get('missing_labels',[])==missing.get(path.stem,[]):continue
        db.execute('DELETE FROM parts WHERE id=?',(path.stem,))
        try:
            data=parse_svg(path)
            metrics,geometry=inspect(data)
            metrics.update(scaled=data.get('scaled',False),reflected=data.get('reflected',False))
            metrics['missing_labels']=missing.get(path.stem,[])
            metrics['dxf']=check_dxf(dxf,data) if dxf.exists() else dict(issues=['missing_dxf'])
            metrics['source_svg']=str(path.resolve())
            db.execute('INSERT INTO parts (id,data,metrics,geometry,sha,dxf_sha) VALUES (?,?,?,?,?,?)',(path.stem,json.dumps(data),json.dumps(metrics),geometry.wkb,svg_sha,dxf_sha))
        except Exception as error:
            db.execute('INSERT INTO parts (id,data,metrics,geometry,sha,dxf_sha) VALUES (?,?,?,?,?,?)',(path.stem,'{}',json.dumps(dict(error=repr(error))),'',svg_sha,dxf_sha))
        if index%100==0:
            db.commit();print(f'Audited {index+1}/{len(files)}; {time.monotonic()-started:.1f}s',flush=True)
    db.commit()
    rows=[(pid,json.loads(m)) for pid,m in db.execute('SELECT id,metrics FROM parts ORDER BY id')]
    complete=[(pid,m) for pid,m in rows if 'error' not in m]
    summary=dict(files=len(files),pairs_audited=len(complete),errors=[(pid,m['error']) for pid,m in rows if 'error' in m],
        parts_with_open=sum(m['open_cut_mm']>1e-7 for _,m in complete),
        open_cut_mm=math.fsum(m['open_cut_mm'] for _,m in complete),
        total_cut_mm=math.fsum(m['total_cut_mm'] for _,m in complete),
        parts_with_invalid=sum(m['invalid_contours']>0 for _,m in complete),
        parts_with_islands=sum(m['islands']>1 for _,m in complete),
        parts_with_thin_contours=sum(m['thin_contours']>0 for _,m in complete),
        parts_with_missing_labels=sum(bool(m['missing_labels']) for _,m in complete),
        missing_labels=sum(len(m['missing_labels']) for _,m in complete),
        dxf_issues=[(pid,m['dxf']['issues']) for pid,m in complete if m['dxf']['issues']],
        labels_without_fit=sum(m['dxf'].get('labels_without_fit',0) for _,m in complete),
        scaled_parts=[pid for pid,m in complete if m.get('scaled',False)],
        reflected_parts=[pid for pid,m in complete if m.get('reflected',False)],
        outside_page=[pid for pid,m in complete if m['outside_page']],
        elapsed_seconds=time.monotonic()-started)
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    db.close()


if __name__=='__main__':main()
