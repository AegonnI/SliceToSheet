"""Render comparisons from actual file coordinates and depth slices for review."""
from pathlib import Path
import argparse
import sqlite3
import json
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline import unpack
from slicer import as_solid
from tools.audit_exports import parse_svg
from sections import section_path


def draw(ax,data,title):
    for c in data['contours']:
        p=np.asarray(c['points']);p=np.vstack([p,p[0]]) if c['closed'] else p
        ax.plot(p[:,0],p[:,1],color='#253a46',linewidth=.6)
    for a,b,c,d in data['leaders']:ax.plot([a,c],[b,d],color='#8596a2',linewidth=.4)
    points=np.vstack([c['points'] for c in data['contours']])
    low,high=points.min(axis=0),points.max(axis=0)
    ax.set_xlim(low[0]-5,high[0]+5);ax.set_ylim(low[1]-5,high[1]+5)
    ax.set_aspect('equal');ax.set_title(title,fontsize=10);ax.set_xlabel('мм');ax.set_ylabel('мм')
    ax.figure.canvas.draw()
    scale=ax.bbox.height/(ax.get_ylim()[1]-ax.get_ylim()[0])*72/ax.figure.dpi
    for t in data['texts']:
        ax.text(t['x'],t['y'],t['text'],fontsize=t['height']*scale,color='#135d8f',
                family='DejaVu Sans Mono',rotation=t.get('angle',0),rotation_mode='anchor',va='baseline')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('original',type=Path);parser.add_argument('improved',type=Path)
    parser.add_argument('output',type=Path);parser.add_argument('--parts',nargs='+',required=True)
    parser.add_argument('--depths',nargs='+',type=float,default=[9.5,10.5,11.,15.])
    args=parser.parse_args();original=args.original.resolve();improved=args.improved.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True);ids=args.parts
    fig,axes=plt.subplots(len(ids),2,figsize=(10,4.7*len(ids)),dpi=160,squeeze=False)
    for row,pid in enumerate(ids):
        before=original/'sections_a4'/(pid+'.svg')
        after=section_path(improved/'sections_a4',pid,'svg')
        draw(axes[row,0],parse_svg(before),'Исходный экспорт · '+pid)
        draw(axes[row,1],parse_svg(after),'Нативное сечение и новая маркировка')
    fig.tight_layout();fig.savefig(output/'before_after.png',bbox_inches='tight');plt.close(fig)
    database=sqlite3.connect((original/'progress.sqlite').as_uri()+'?mode=ro',uri=True)
    blob=database.execute('SELECT mesh FROM parts WHERE id=?',(ids[0],)).fetchone()[0]
    mesh=unpack(blob);solid=as_solid(mesh)
    fig,axes=plt.subplots(1,len(args.depths),figsize=(3.25*len(args.depths),6),dpi=160,squeeze=False)
    origin=mesh.bounds[0,:2]
    for ax,z in zip(axes[0],args.depths):
        section=solid.slice(z)
        for polygon in section.to_polygons():
            points=np.asarray(polygon)-origin;p=np.vstack([points,points[0]])
            ax.plot(p[:,0],p[:,1],color='#135d8f',linewidth=.65)
        ax.set_title('Глубина %.1f мм\nОбластей: %d'%(z,len(section.decompose())),fontsize=10)
        ax.set_aspect('equal');ax.set_xlabel('мм');ax.set_ylim(-3,mesh.extents[1]+3);ax.set_xlim(-3,mesh.extents[0]+3)
    fig.suptitle('Одна и та же замкнутая 3D-деталь: связь между областями зависит от глубины среза',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(output/'depth_slices.png',bbox_inches='tight');plt.close(fig)
    local=mesh.copy();local.apply_translation(-mesh.bounds[0]);local.export(output/(ids[0]+'.stl'))
    database.close();print(output)


if __name__=='__main__':main()
