"""Disk-backed, transactional processing. Geometry checkpoints are buffered in bounded layer batches."""
import io
import os
import json
import time
import hashlib
import sqlite3
import struct
from pathlib import Path
from dataclasses import asdict
import numpy as np
import trimesh

VERSION = 1

class Paused(Exception):
    pass


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def save_config(path, cfg):
    cfg.validate()
    atomic_json(path, {'version': 1, 'settings': asdict(cfg)})


def load_config(path):
    from slicer import Settings
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if data.get('version') != 1: raise ValueError('Неподдерживаемая версия настроек.')
    cfg = Settings.from_saved(data['settings']); cfg.validate()
    return cfg


def pack(mesh):
    f = io.BytesIO()
    np.savez(f, vertices=mesh.vertices, faces=mesh.faces)
    return f.getvalue()


def unpack(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        return trimesh.Trimesh(vertices=data['vertices'], faces=data['faces'], process=False)


def run_job(source, output, cfg=None, log=print, resume=False, stop=None, progress=None):
    from slicer import Settings, Part, load_model, slicing_plan, iter_cut, place_connectors, export_svg
    output = Path(output)
    source = Path(source)
    def cancel():
        if stop is not None and stop.is_set(): raise Paused('Расчёт приостановлен. Сохранённые этапы можно продолжить.')
    def notify(stage, value, total):
        if progress: progress(stage, value, total)
    cancel()
    # OS file lock is automatically released after a crash; two apps cannot write one job.
    if not resume: output.mkdir(parents=True, exist_ok=False)
    elif not (output/'progress.sqlite').is_file(): raise ValueError('В папке нет сохранённого расчёта новой версии.')
    lock = (output/'job.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
            try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError: raise ValueError('Этот расчёт уже открыт другой программой.')
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (output/'log.txt').open('a',encoding='utf-8',buffering=65536) as journal:
            def logged(line):
                journal.write(time.strftime('%Y-%m-%d %H:%M:%S ')+line+'\n')
                log(line)
            try:
                return _locked(source,output,cfg,logged,resume,cancel,notify)
            except Exception as error:
                logged(str(error));raise
    finally:
        lock.close()


def _locked(source, output, cfg, log, resume, cancel, notify):
    from slicer import Settings, Part, load_model, slicing_plan, iter_cut, place_connectors, export_svg
    db = sqlite3.connect(str(output/'progress.sqlite'))
    db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA cache_size=-8192')
    db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS parts (id TEXT PRIMARY KEY,layer INTEGER,cell TEXT,mesh BLOB,info TEXT)')
    db.execute('CREATE INDEX IF NOT EXISTS part_layer ON parts(layer)')
    db.execute('CREATE TABLE IF NOT EXISTS connections (id TEXT PRIMARY KEY,data TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS warnings (text TEXT)')
    def get(key, default=None):
        row=db.execute('SELECT value FROM meta WHERE key=?',(key,)).fetchone()
        return default if row is None else json.loads(row[0])
    def put(key,value):
        db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',(key,json.dumps(value,ensure_ascii=False)))
    pending_parts = {}
    def save_parts(parts):
        for p in parts:
            p.mesh._cache.clear()
            pending_parts[p.id] = p
    def checkpoint():
        for p in pending_parts.values():
            db.execute('INSERT OR REPLACE INTO parts(id,layer,cell,mesh) VALUES (?,?,?,?)',
                       (p.id,int(p.cell[2]),json.dumps([int(x) for x in p.cell]),pack(p.mesh)))
        db.commit()
        pending_parts.clear()
    def read_layers(k):
        result = {row[0]: Part(row[0],tuple(json.loads(row[1])),unpack(row[2])) for row in db.execute(
            'SELECT id,cell,mesh FROM parts WHERE layer IN (?,?) ORDER BY layer,id',(k,k+1))
            if row[0] not in pending_parts}
        # Work on copies: an interrupted boolean must not corrupt the last complete layer.
        result.update({pid: Part(pid,p.cell,p.mesh.copy()) for pid,p in pending_parts.items() if p.cell[2] in (k,k+1)})
        return [result[pid] for pid in sorted(result, key=lambda pid: (result[pid].cell[2], pid))]
    def fingerprint(check_stop=True):
        h=hashlib.sha256()
        with source.open('rb') as f:
            while True:
                if check_stop:cancel()
                chunk=f.read(4*1024*1024)
                if not chunk: break
                h.update(chunk)
        return h.hexdigest()
    try:
        if resume:
            if get('version') != VERSION: raise ValueError('Несовместимая версия сохранённого расчёта.')
            saved=Settings.from_saved(get('settings'));saved.validate()
            if cfg is not None and asdict(cfg)!=asdict(saved): raise ValueError('Настройки отличаются от сохранённого расчёта.')
            cfg=saved
            if fingerprint()!=get('source_hash'): raise ValueError('Выбран другой или изменённый файл модели. Продолжение запрещено.')
            if get('complete',False):
                (output/'INCOMPLETE.txt').unlink(missing_ok=True)
                return json.loads((output/'assembly.json').read_text(encoding='utf-8'))
        else:
            cfg=cfg or Settings();cfg.validate()
            with db:
                put('version',VERSION);put('settings',asdict(cfg));put('source_hash',fingerprint(check_stop=False))
                put('source',str(source.resolve()));put('cut_next',0);put('connect_next',0)
                put('raw_volume',0);put('connections',[]);put('warnings',[]);put('occupied',[])
            save_config(output/'settings.json',cfg)
        (output/'INCOMPLETE.txt').write_text('Расчёт не завершён. Продолжите его через программу. Не используйте комплект для изготовления.',encoding='utf-8')
        count=get('layer_count')
        if count is None or get('cut_next',0)<count:
            cancel();log('Загрузка и проверка модели…')
            mesh,transform=load_model(source,cfg)
            steps,page,counts=slicing_plan(mesh,cfg);count=int(counts[2])
            with db:
                put('layer_count',count);put('steps',steps.tolist());put('page',page)
                put('size',mesh.extents.tolist());put('transform',transform.tolist());put('source_volume',float(mesh.volume))
            log('Нарезка: %d слоёв; сохранение каждые %d слоёв.'%(count,cfg.checkpoint_layers))
            started=time.perf_counter();last=started
            for k,parts in iter_cut(mesh,cfg,lambda _:None,range(get('cut_next'),count),cancel):
                volume=sum(p.mesh.volume for p in parts)
                save_parts(parts);put('raw_volume',get('raw_volume')+volume);put('cut_next',k+1)
                if (k+1)%cfg.checkpoint_layers == 0 or k+1 == count: checkpoint()
                elapsed=time.perf_counter()-last;last=time.perf_counter()
                log('Нарезка %d/%d: %d деталей, %.2f с; рассчитано.'%(k+1,count,len(parts),elapsed))
                notify('Нарезка',k+1,count)
                del parts
            del mesh
        raw=get('raw_volume');original=get('source_volume')
        if abs(raw-original)>max(.01,original*1e-4): raise ValueError('Контроль объёма нарезки не пройден.')
        if db.execute('SELECT COUNT(*) FROM parts').fetchone()[0]==0: raise ValueError('После нарезки нет деталей.')
        steps=np.array(get('steps'));page=get('page')
        occupied=[(np.array(a),np.array(b)) for a,b in get('occupied',[])]
        records=[json.loads(row[0]) for row in db.execute('SELECT data FROM connections ORDER BY rowid')]
        warnings=[row[0] for row in db.execute('SELECT text FROM warnings ORDER BY rowid')]
        from keys import KeyRegistry
        registry=KeyRegistry(cfg.key_tolerance)
        seen_keys=set()
        for record in records:
            if record.get('key_id') and record['key_id'] not in seen_keys:
                registry.add(record['key_signature']);seen_keys.add(record['key_id'])
        for k in range(get('connect_next',0),count):
            cancel();start=time.perf_counter();parts=read_layers(k)
            new,notes,_=place_connectors(parts,steps,cfg,lambda _:None,owner_layer=k,occupied=occupied,cancel=cancel,key_registry=registry)
            for r in new: r['id']='K%04d'%(len(records)+1);records.append(r)
            warnings.extend(notes)
            # Earlier envelopes cannot intersect future probes below this conservative boundary.
            threshold=(k+1)*cfg.thickness-cfg.length-(cfg.diameter+cfg.clearance)/2-cfg.wall
            occupied=[(a,b) for a,b in occupied if b[2]>=threshold]
            save_parts(parts)
            db.executemany('INSERT INTO connections VALUES (?,?)',[(r['id'],json.dumps(r)) for r in new])
            db.executemany('INSERT INTO warnings VALUES (?)',[(note,) for note in notes])
            put('occupied',[(a.tolist(),b.tolist()) for a,b in occupied]);put('connect_next',k+1)
            if (k+1)%cfg.checkpoint_layers == 0 or k+1 == count: checkpoint()
            del parts
            log('Отверстия %d/%d: +%d шпилек, %.2f с; рассчитано.'%(k+1,count,len(new),time.perf_counter()-start))
            notify('Отверстия',k+1,count)
        ids=[row[0] for row in db.execute('SELECT id FROM parts ORDER BY layer,id')]
        # Connected components in O(parts + connections).
        adjacency={p:set() for p in ids}
        part_records={p:[] for p in ids}
        for r in records:
            adjacency[r['part_a']].add(r['part_b']);adjacency[r['part_b']].add(r['part_a'])
            part_records[r['part_a']].append(r);part_records[r['part_b']].append(r)
        groups=[];remaining=set(ids)
        while remaining:
            todo=[remaining.pop()];group=[]
            while todo:
                p=todo.pop();group.append(p)
                for q in adjacency[p]:
                    if q in remaining: remaining.remove(q);todo.append(q)
            groups.append(sorted(group))
        final_notes=list(warnings)
        if cfg.four_holes:
            for pid in ids:
                layer=int(pid.split('_')[0][1:])
                for key, direction, required in [('part_b','вниз',2 if layer>1 else 0),
                                                  ('part_a','вверх',2 if layer<count else 0)]:
                    actual=sum(r['axis']=='Z' and r[key]==pid for r in part_records[pid])
                    if actual < required:
                        final_notes.append(f'{pid}: {direction} {actual}/{required} отверстий; не хватает доступных соседей или безопасных мест.')
        if len(ids)>1:
            final_notes.extend(p+': НЕТ СОЕДИНЕНИЙ. Требуется ручная доработка.' for p in ids if not adjacency[p])
        if len(groups)>1: final_notes.append('Сборка состоит из %d несвязанных групп деталей.'%len(groups))
        for folder in ('stl','sections_a4'): (output/folder).mkdir(exist_ok=True)
        for index,pid in enumerate(ids):
            cancel()
            row=db.execute('SELECT cell,mesh,info FROM parts WHERE id=?',(pid,)).fetchone()
            if row[2] is not None and (output/'stl'/(pid+'.stl')).is_file() and (output/'sections_a4'/(pid+'.svg')).is_file(): continue
            part=Part(pid,tuple(json.loads(row[0])),unpack(row[1]))
            offset=part.mesh.bounds[0].copy();local=part.mesh.copy();local.apply_translation(-offset)
            target=output/'stl'/(pid+'.stl');tmp=target.with_suffix('.tmp')
            tmp.write_bytes(local.export(file_type='stl'));os.replace(tmp,target)
            target=output/'sections_a4'/(pid+'.svg');tmp=target.with_suffix('.tmp')
            marking_notes=export_svg(part,tmp,page,cfg,part_records[pid]);os.replace(tmp,target)
            from marking import address, links_for
            info=dict(id=pid,cell=part.cell,assembly_offset=offset.tolist(),layer_number=part.cell[2]+1,
                      layer_start_mm=part.cell[2]*cfg.thickness,
                      layer_end_mm=min((part.cell[2]+1)*cfg.thickness,get('size')[2]),
                      size=part.mesh.extents.tolist(),volume=float(part.mesh.volume),
                      marking=address(pid),links=links_for(pid,part_records[pid]),marking_warnings=marking_notes)
            db.execute('UPDATE parts SET info=? WHERE id=?',(json.dumps(info),pid))
            next_layer = int(ids[index+1].split('_')[0][1:])-1 if index+1 < len(ids) else None
            if next_layer != part.cell[2] and ((part.cell[2]+1)%cfg.checkpoint_layers == 0 or next_layer is None): db.commit()
            if index%25==0 or index==len(ids)-1:
                log('Экспорт %d/%d'%(index+1,len(ids)));notify('Экспорт',index+1,len(ids))
            del part,local
        info=[json.loads(row[0]) for row in db.execute('SELECT info FROM parts ORDER BY layer,id')]
        final_notes.extend(note for p in info for note in p.get('marking_warnings',[]))
        # Write the assembly as a stream; never concatenate every mesh in RAM.
        notify('Сборка STL',0,len(ids));total_faces=0
        with (output/'assembly.stl.tmp').open('wb') as f:
            f.write(b'STL A4 streaming assembly'.ljust(80,b'\0'));f.write(struct.pack('<I',0))
            for index,p in enumerate(info):
                cancel()
                local=trimesh.load(output/'stl'/(p['id']+'.stl'),force='mesh',process=False)
                local.apply_translation(p['assembly_offset'])
                data=local.export(file_type='stl');total_faces+=len(local.faces)
                if total_faces>0xffffffff: raise ValueError('Общая сборка слишком велика для STL; отдельные детали сохранены.')
                f.write(data[84:]);del local,data
                if index%100==0: notify('Сборка STL',index+1,len(ids))
            f.seek(80);f.write(struct.pack('<I',total_faces));f.flush();os.fsync(f.fileno())
        os.replace(output/'assembly.stl.tmp',output/'assembly.stl')
        import csv
        fields=['id','part_a','part_b','axis','x','y','z','hole_diameter','dowel_diameter','dowel_length','depth_each','type']
        if cfg.unique_keys:fields+=['key_id','key_signature']
        with (output/'connections.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
        report=dict(settings=asdict(cfg),source=str(source),source_to_assembly=get('transform'),size=get('size'),
                    layer_count=count,parts=info,connections=records,warnings=final_notes,connected_groups=groups,
                    volume_before_holes=raw,volume_after_holes=sum(p['volume'] for p in info))
        if cfg.unique_keys:
            report['key_validation']=dict(scope='complete_interface_triplets',keys=registry.count,
                center_tolerance_mm=cfg.key_tolerance,distance_tolerance_mm=2*cfg.key_tolerance,
                note='Проверены полные тройки каждого стыка. Смешанные тройки отверстий от разных стыков и все возможные положения целых деталей не проверены.')
        atomic_json(output/'assembly.json',report)
        (output/'REPORT.txt').write_text('Деталей: %d\nШпилек: %d\nСлоёв: %d\n\n'%(len(ids),len(records),count)+
            '\n'.join(final_notes or ['Все соединения размещены.'])+
            '\n\nSVG — среднее сечение, а не контур всей объёмной детали. Печатать 100%.\n'
            'Координаты CSV общие; для локальных вычтите assembly_offset из JSON.\n'
            'Маркировка L.R.C.компонент: слой, строка, столбец, компонент; номера начинаются с 1.\n'
            'DOWN — к нижнему слою, UP — к верхнему, SIDE — внутри слоя; K — общий номер шпильки.\n'
            'Алгоритм проверяет геометрию, но не прочность и не порядок физической сборки.\n'+
            (report['key_validation']['note']+'\n' if cfg.unique_keys else ''),encoding='utf-8')
        with db: put('complete',True)
        (output/'INCOMPLETE.txt').unlink(missing_ok=True)
        notify('Готово',1,1);log('Готово: %d деталей, %d шпилек, %d замечаний.'%(len(ids),len(records),len(final_notes)))
        return report
    except Paused:
        checkpoint()
        raise
    finally:
        db.close()
