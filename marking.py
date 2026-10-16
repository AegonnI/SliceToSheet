"""Assembly addresses and printable labels placed within the material section."""
from html import escape
import numpy as np
import manifold3d


def address(part_id):
    layer, row, column, component = (int(token[1:]) for token in part_id.split('_'))
    return f'L{layer}.R{row}.C{column}.{component}'


def links_for(part_id, records):
    result = []
    for r in records:
        other = r['part_b'] if r['part_a'] == part_id else r['part_a']
        direction = ('UP' if r['part_a'] == part_id else 'DOWN') if r['axis'] == 'Z' else 'SIDE'
        result.append(dict(connection=r['id'], direction=direction, part=other, address=address(other),
                           x=r['x'], y=r['y'], z=r['z']))
    return result


def svg_labels(part, page, cfg, records, z):
    from slicer import as_solid
    section = as_solid(part.mesh).slice(z)
    origin = part.mesh.bounds[0,:2]
    low, high = part.mesh.bounds[:,:2]
    labels, notes = [], []
    free = section

    def place(text, near=None):
        nonlocal free
        for size in (2.5, 2.0, 1.6):
            width, height = len(text)*size*.62, size*1.5
            if np.any(high-low < [width+2, height+2]): continue
            xs=np.linspace(low[0]+1, high[0]-width-1, 17)
            ys=np.linspace(low[1]+1, high[1]-height-1, 17)
            candidates=[np.array([x,y]) for x in xs for y in ys]
            center=(low+high)/2 if near is None else np.asarray(near)
            candidates.sort(key=lambda xy: float(np.linalg.norm(xy+[width/2,height/2]-center)))
            for xy in candidates:
                box=manifold3d.CrossSection.square([width,height]).translate(xy)
                if (box-free).area() > 1e-5: continue
                free=free-manifold3d.CrossSection.square([width+1.6,height+1.6]).translate(xy-.8)
                x,y=xy-origin+cfg.margin
                labels.append('<text x="%.3f" y="%.3f" font-family="monospace" font-size="%.2f" textLength="%.3f" lengthAdjust="spacingAndGlyphs">%s</text>' %
                              (x,page[1]-y-size*.3,size,width,escape(text)))
                if near is not None:
                    start=np.array([x+width/2,page[1]-y-height/2])
                    end=np.array([near[0]-origin[0]+cfg.margin,page[1]-(near[1]-origin[1]+cfg.margin)])
                    delta=start-end;distance=np.linalg.norm(delta)
                    if distance > 0:
                        end += delta/distance*((cfg.diameter+cfg.clearance)/2+.6)
                    labels.append('<path d="M %.3f %.3f L %.3f %.3f" fill="none" stroke="#777" stroke-width="0.12"/>' % (*start,*end))
                return
        notes.append(part.id+': маркировка не помещается на сечении: '+text)

    place(address(part.id))
    for link in links_for(part.id, records):
        place(link['direction']+' '+link['address']+' '+link['connection'], [link['x'],link['y']])
    return labels, notes
