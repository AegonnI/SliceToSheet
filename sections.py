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
    """Return the same middle-section polylines used by the SVG exporter."""
    z = float(part.mesh.bounds[:, 2].mean())
    section = part.mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    contours = []
    if section is not None:
        for line in section.discrete:
            points = np.asarray(line[:, :2], dtype=float)
            if len(points) < 2:
                continue
            closed = bool(len(points) > 2 and np.linalg.norm(points[0]-points[-1]) <= 1e-6)
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            polygon = points[:-1] if closed else points
            area = abs(float(np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1) -
                                    polygon[:, 1]*np.roll(polygon[:, 0], -1))/2)) if closed else 0.0
            contours.append(dict(points=polygon, closed=closed, length=length, area=area))
    for contour in contours:
        if not contour['closed']:
            contour['role'] = 'open'
            continue
        depth = sum(other['closed'] and other['area'] > contour['area']+1e-9 and
                    _inside(contour['points'][0], other['points'])
                    for other in contours if other is not contour)
        contour['role'] = 'inner' if depth % 2 else 'outer'
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
                              ('SHEET_FRAME', 9, 0)]:
        document.layers.new(name, dxfattribs={'color': color, 'plot': plot})
    document.styles.new('LABEL', dxfattribs={'font': 'txt'})
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
            model.add_text(element.text or '', dxfattribs={
                'layer': 'LABELS', 'style': 'LABEL', 'height': float(element.get('font-size')),
                'insert': (float(element.get('x'))+padding,
                           page[1]-float(element.get('y'))+padding), 'rotation': 0})
        elif element.tag == 'path':
            values = [float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',
                                                   element.get('d', ''))]
            if len(values) == 4:
                model.add_line((values[0]+padding, page[1]-values[1]+padding),
                               (values[2]+padding, page[1]-values[3]+padding),
                               dxfattribs={'layer': 'LEADERS'})
    width, height = page[0]+2*padding, page[1]+2*padding
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
        points = ' '.join('%.4f,%.4f' % (x,page[1]-y) for x,y in xy)
        lines.append('<polyline points="%s"/>' % points)
    # Every fabrication label is placed in the material. The old page header
    # could cross a full-page contour, so it is deliberately omitted here.
    lines += ['</g>', *labels, '</g>', '</svg>']
    Path(destination).write_text('\n'.join(lines),encoding='utf-8')


def export_sections(part, folder, page, cfg, records=()):
    """Atomically export selected section formats and return metrics + notes."""
    import os
    from marking import svg_labels
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    z, contours = section_contours(part)
    labels, notes = svg_labels(part,page,cfg,records,z)
    formats = ('svg', 'dxf') if cfg.export_format == 'both' else (cfg.export_format,)
    for extension in formats:
        target = folder/(part.id+'.'+extension)
        temporary = target.with_name(target.name+'.tmp')
        if extension == 'svg':
            _export_svg(part,temporary,page,cfg,contours,labels)
        else:
            export_dxf(part,temporary,page,cfg,records,contours,labels)
        os.replace(temporary, target)
    return part_metrics(part, contours), list(dict.fromkeys(notes))


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
