"""Millimetre section exports and manufacturing measurements.

Cut lengths measure one pass along the actual polygon segments. Labels, leader
lines, tool compensation and repeated depth passes are never included.
"""
import math
import re
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


METRIC_FIELDS = ('net_area_mm2', 'outer_cut_mm', 'inner_cut_mm', 'open_cut_mm',
                 'total_cut_mm', 'surface_area_mm2', 'outer_contours',
                 'inner_contours', 'open_contours')
EXPORT_REVISION = 3


def even_odd(polygons):
    """Binary XOR avoids Shapely's deprecated symmetric_difference_all path."""
    import shapely
    result=shapely.Polygon()
    for polygon in polygons:result=result.symmetric_difference(polygon)
    return result


def section_path(folder, part_id, extension, review=False):
    base=Path(folder).parent/'Inspection'/'Rejected' if review else Path(folder)
    return base/extension.upper()/part_id.split('_')[0]/(part_id+'.'+extension)


def section_shape(part):
    """Regularize sub-0.01 mm numerical slits, recording every omission."""
    from slicer import as_solid
    import manifold3d
    import shapely
    key='fabrication_section_v2'
    if key in part.mesh._cache:return part.mesh._cache[key]
    solid=as_solid(part.mesh)
    if solid.status()!=manifold3d.Error.NoError:
        raise ValueError(part.id+': некорректный объём для экспорта сечения.')
    z=float(part.mesh.bounds[:,2].mean())
    raw=solid.slice(z)
    kept=[];removed=[];zero_area_paths=0.;repaired_rings=0
    def micro_polygon(polygon):
        if polygon.area>=.1:return False
        rectangle=polygon.minimum_rotated_rectangle
        points=np.asarray(rectangle.exterior.coords) if rectangle.geom_type=='Polygon' else np.empty((0,2))
        width=float(np.linalg.norm(np.diff(points,axis=0),axis=1).min()) if len(points) else 0.
        if width<.01:
            removed.append(dict(area_mm2=float(polygon.area),width_mm=width));return True
        return False
    for polygon in raw.to_polygons():
        p=np.asarray(polygon)
        area=abs(float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))/2))
        if area<.1:
            rectangle=shapely.Polygon(p).minimum_rotated_rectangle
            points=np.asarray(rectangle.exterior.coords) if rectangle.geom_type=='Polygon' else np.empty((0,2))
            width=float(np.linalg.norm(np.diff(points,axis=0),axis=1).min()) if len(points) else 0.
            if width<.01:
                removed.append(dict(area_mm2=area,width_mm=width));continue
        candidate=shapely.Polygon(p)
        if not candidate.is_valid:
            repaired_rings+=1;candidate=shapely.make_valid(candidate)
        def polygonal(geometry):
            nonlocal zero_area_paths
            if geometry.geom_type=='Polygon':return [geometry]
            if geometry.geom_type in ('LineString','LinearRing'):
                zero_area_paths+=geometry.length;return []
            if geometry.geom_type=='Point' or geometry.is_empty:return []
            result=[]
            for child in shapely.get_parts(geometry):
                result.extend(polygonal(child))
            return result
        kept.extend(p for p in polygonal(candidate) if not micro_polygon(p))
    material=even_odd(kept)
    if not material.is_valid:material=shapely.make_valid(material)
    loops=[];roles=[]
    for polygon in shapely.get_parts(material):
        if polygon.geom_type!='Polygon' or polygon.is_empty:continue
        if micro_polygon(polygon):continue
        loops.append(np.asarray(polygon.exterior.coords)[:-1]);roles.append('outer')
        for ring in polygon.interiors:
            if micro_polygon(shapely.Polygon(ring)):continue
            loops.append(np.asarray(ring.coords)[:-1]);roles.append('inner')
    section=manifold3d.CrossSection(loops,manifold3d.FillRule.EvenOdd)
    part.mesh.metadata['section_checks']=dict(
        numerical_cleanup_width_mm=.01, numerical_cleanup_max_contour_area_mm2=.1,
        omitted_microcontours=removed, area_change_mm2=float(section.area()-raw.area()),
        repaired_touching_rings=repaired_rings,omitted_zero_area_path_mm=float(zero_area_paths),
        material_islands=0 if section.is_empty() else len(section.decompose()))
    result=(z,section)
    part.mesh._cache[key]=result
    part.mesh._cache[key+'_loops']=loops
    part.mesh._cache[key+'_roles']=roles
    return result


def _inside(point, polygon):
    x, y = point
    a = polygon
    b = np.roll(a, -1, axis=0)
    crossing = (a[:, 1] > y) != (b[:, 1] > y)
    if not np.any(crossing):
        return False
    a, b = a[crossing], b[crossing]
    return bool(np.count_nonzero(x < (b[:, 0]-a[:, 0]) * (y-a[:, 1]) /
                                (b[:, 1]-a[:, 1]) + a[:, 0]) % 2)


def section_contours(part):
    """Slice the indexed native solid: avoid trimesh's path stitching artifacts."""
    z,section=section_shape(part)
    contours = []
    if not section.is_empty():
        for index,line in enumerate(part.mesh._cache['fabrication_section_v2_loops']):
            points = np.asarray(line, dtype=float)
            if len(points) < 2:
                continue
            closed = len(points)>2
            polygon = points[:-1] if np.linalg.norm(points[0]-points[-1]) <= 1e-8 else points
            length = float(np.linalg.norm(polygon-np.roll(polygon,1,axis=0),axis=1).sum())
            area = abs(float(np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1) -
                                    polygon[:, 1]*np.roll(polygon[:, 0], -1))/2)) if closed else 0.0
            contours.append(dict(points=polygon, closed=closed, length=length, area=area,
                role=part.mesh._cache['fabrication_section_v2_roles'][index]))
    return z, contours


def part_metrics(part, contours=None):
    if contours is None:
        _, contours = section_contours(part)
    by_role = {role: [c for c in contours if c['role'] == role]
               for role in ('outer', 'inner', 'open')}
    result = dict(net_area_mm2=math.fsum(c['area'] * (-1 if c['role'] == 'inner' else 1)
                                       for c in contours if c['closed']),
                  surface_area_mm2=float(part.mesh.area))
    for role in by_role:
        result[role+'_cut_mm'] = math.fsum(c['length'] for c in by_role[role])
        result[role+'_contours'] = len(by_role[role])
    result['total_cut_mm'] = math.fsum(result[r+'_cut_mm'] for r in by_role)
    # Native parallel reductions can differ in insignificant last bits. This
    # resolution is far below STL coordinate precision for fabrication sizes.
    return {key: round(value, 6) if isinstance(value, float) else value
            for key, value in result.items()}


def export_dxf(part, destination, page, cfg, records=(), contours=None, labels=None):
    """Write upright CAD labels and unchanged section contours at scale 1:1."""
    import ezdxf
    from ezdxf.enums import TextEntityAlignment
    from marking import svg_labels
    if contours is None:
        z, contours = section_contours(part)
    else:
        z = float(part.mesh.bounds[:, 2].mean())
    if labels is None:
        labels, notes = svg_labels(part, page, cfg, records, z)
    else:
        notes = []
    document = ezdxf.new('R2010')
    document.units = 4
    document.header['$MEASUREMENT'] = 1
    for name, color, plot in [('CONTOURS', 7, 1), ('LABELS', 2, 1), ('LEADERS', 8, 1),
                              ('SHEET_FRAME', 9, 0), ('PERIMETER_INFO',3,0), ('CHECK_REQUIRED',1,0)]:
        document.layers.new(name, dxfattribs={'color': color, 'plot': plot})
    document.styles.new('LABEL', dxfattribs={'font': 'cour.ttf'})
    document.appids.new('BOAT_LABEL')
    model = document.modelspace()
    padding = cfg.svg_padding
    origin = part.mesh.bounds[0, :2]
    for contour in contours:
        points = contour['points'] - origin + cfg.margin + padding
        model.add_lwpolyline(points, close=contour['closed'],
                             dxfattribs={'layer': 'CONTOURS'})
    # The shared label placer emits SVG baselines in downward Y. CAD baselines
    # use upward Y; only coordinates are converted, keeping text rotation zero.
    for fragment in labels:
        element = ET.fromstring(fragment)
        if element.tag == 'text':
            position=np.array([float(element.get('x'))+padding,page[1]-float(element.get('y'))+padding])
            transform=element.get('transform','')
            angle=-float(re.findall(r'[-+]?(?:\d*\.\d+|\d+)',transform)[0]) if transform else 0.
            width=float(element.get('textLength','0'))
            entity=model.add_text(element.text or '', dxfattribs={
                'layer': 'LABELS', 'style': 'LABEL', 'height': float(element.get('font-size')),
                'insert': tuple(position), 'rotation': angle})
            if width:
                end=position+width*np.array([math.cos(math.radians(angle)),math.sin(math.radians(angle))])
                entity.set_placement(tuple(position),tuple(end),align=TextEntityAlignment.FIT)
            if element.get('data-connection'):
                entity.set_xdata('BOAT_LABEL',[(1000,element.get('data-connection'))])
        elif element.tag == 'path':
            values = [float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',
                                                   element.get('d', ''))]
            if len(values) == 4:
                model.add_line((values[0]+padding, page[1]-values[1]+padding),
                               (values[2]+padding, page[1]-values[3]+padding),
                               dxfattribs={'layer': 'LEADERS'})
    width, height = page[0]+2*padding, page[1]+2*padding
    metrics=part_metrics(part,contours)
    from marking import address
    own=address(part.id,getattr(cfg,'_label_addresses',None))
    for index,text in enumerate([f'PART {own} | SOURCE {part.id}',
            'CUT %.6f mm | OUTER %.6f | INNER %.6f | OPEN %.6f'%(metrics['total_cut_mm'],metrics['outer_cut_mm'],metrics['inner_cut_mm'],metrics['open_cut_mm']),
            'NET AREA %.6f mm2 | SCALE 1:1 | MM'%metrics['net_area_mm2']]):
        model.add_text(text,dxfattribs={'layer':'PERIMETER_INFO','height':2.5,'insert':(0,-8-index*4)})
    if not contours:
        model.add_text('NO MANUFACTURABLE MID-SECTION - REVIEW 3D PART '+own,
                       dxfattribs={'layer':'CHECK_REQUIRED','height':3,'insert':(10,10)})
    model.add_lwpolyline([(0, 0), (width, 0), (width, height), (0, height)], close=True,
                         dxfattribs={'layer': 'SHEET_FRAME'})
    document.set_modelspace_vport(height=height, center=(width/2, height/2))
    document.saveas(str(destination))
    return notes


def _export_svg(part, destination, page, cfg, contours, labels):
    """Write the already measured section; text baselines remain upright."""
    padding = cfg.svg_padding
    document = (page[0]+2*padding, page[1]+2*padding)
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="%smm" height="%smm" viewBox="0 0 %s %s">' % (*document,*document),
             '<rect width="100%" height="100%" fill="white"/>',
             '<g transform="translate(%s %s)">' % (padding,padding),
             '<g fill="none" stroke="black" stroke-width="0.2">']
    origin = part.mesh.bounds[0,:2]
    for contour in contours:
        xy = contour['points']-origin+cfg.margin
        if contour['closed']:
            xy = np.vstack((xy,xy[0]))
        points = ' '.join('%.12f,%.12f' % (x,page[1]-y) for x,y in xy)
        lines.append('<polyline points="%s"/>' % points)
    # Every fabrication label is placed in the material. The old page header
    # could cross a full-page contour, so it is deliberately omitted here.
    lines += ['</g>', *labels]
    if not contours:
        from marking import address
        own=address(part.id,getattr(cfg,'_label_addresses',None))
        lines += ['<text x="10" y="20" fill="#b02020" font-size="4" data-layer="CHECK_REQUIRED">%s — REVIEW 3D</text>'%own,
                  '<text x="10" y="30" fill="#b02020" font-size="3" data-layer="CHECK_REQUIRED">NO MANUFACTURABLE MID-SECTION</text>']
    import json
    from html import escape
    lines += ['</g>','<metadata>'+escape(json.dumps(dict(source_part=part.id,metrics=part_metrics(part,contours),section_checks=part.mesh.metadata.get('section_checks',{}))))+'</metadata>','</svg>']
    Path(destination).write_text('\n'.join(lines),encoding='utf-8')


def export_sections(part, folder, page, cfg, records=()):
    """Atomically export selected section formats and return metrics + notes."""
    import os
    from marking import svg_labels
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    z, contours = section_contours(part)
    labels, notes = svg_labels(part,page,cfg,records,z)
    formats = ('svg', 'dxf') if cfg.export_format == 'both' else (cfg.export_format,)
    for extension in formats:
        target = section_path(folder,part.id,extension,review=not contours)
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary = target.with_name(target.name+'.tmp')
        if extension == 'svg':
            _export_svg(part,temporary,page,cfg,contours,labels)
        else:
            export_dxf(part,temporary,page,cfg,records,contours,labels)
        os.replace(temporary, target)
    metrics=part_metrics(part,contours)
    if metrics['outer_contours']>1:
        notes.append(part.id+': среднее сечение содержит несколько отдельных областей материала; проверьте 3D-деталь перед изготовлением.')
    omitted=part.mesh.metadata.get('section_checks',{}).get('omitted_microcontours',[])
    if omitted:
        notes.append(part.id+': из чертежа исключены численные микроконтуры: %d; ширина <0,01 мм и площадь <0,1 мм². 3D-геометрия сохранена.'%len(omitted))
    checks=part.mesh.metadata.get('section_checks',{})
    if checks.get('repaired_touching_rings'):
        notes.append(part.id+': касающиеся кольца сечения разделены без изменения площади; вырожденные ответвления вынесены из контура реза.')
    return metrics, list(dict.fromkeys(notes))


def aggregate_metrics(parts):
    """Group section stock area/cut length and separate cut-part 3-D surface."""
    groups = defaultdict(list)
    for part in parts:
        groups[part['layer_number']].append(part)
    def combine(values):
        return dict(parts=len(values), **{field: math.fsum(p['metrics'][field] for p in values)
                                         for field in METRIC_FIELDS})
    layers = [dict(layer_number=k, **combine(values)) for k, values in sorted(groups.items())]
    return dict(totals=combine(parts), layers=layers,
                definitions={
                    'net_area_mm2': 'Middle section material area; internal contours and dowel holes subtracted.',
                    'outer_cut_mm': 'One pass along external closed contours.',
                    'inner_cut_mm': 'One pass along all internal closed contours, including cavities and dowel holes.',
                    'open_cut_mm': 'Open section polylines, if any; require review.',
                    'total_cut_mm': 'Outer + internal + open lengths; labels, leaders, tool offsets and repeated depth passes excluded.',
                    'surface_area_mm2': '3-D triangle surface area of cut parts after drilling, including cut faces and hole walls; not stock area or original hull surface.'})


def write_metrics_csv(destination, parts, summary):
    import csv
    fields = ['scope', 'id', 'layer_number', 'parts', *METRIC_FIELDS]
    with Path(destination).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for p in parts:
            writer.writerow(dict(scope='part', id=p['id'], layer_number=p['layer_number'],
                                 parts=1, **p['metrics']))
        for layer in summary['layers']:
            writer.writerow(dict(scope='layer', **layer))
        writer.writerow(dict(scope='total', **summary['totals']))
