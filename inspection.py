"""Human-readable, searchable manufacturing checks with one row per part."""
import csv
from html import escape
from pathlib import Path
import time


def write_inspection(output,report):
    from sections import section_path
    output=Path(output);destination=output/'Inspection';destination.mkdir(parents=True,exist_ok=True)
    parts=report['parts']
    previous_exports=output/'backups'/('reclassified_exports_'+str(time.time_ns()))
    counts=dict(total=len(parts),islands=0,empty=0,cleanup=0,missing_labels=0,without_connections=0)
    rows=[];csv_rows=[]
    for p in parts:
        checks=p.get('section_checks',{});notes=p.get('marking_warnings',[])
        kinds=[]
        if p['metrics']['outer_contours']>1:counts['islands']+=1;kinds.append('Отдельные области сечения')
        if not p['metrics']['outer_contours']:counts['empty']+=1;kinds.append('Нет пригодного контура')
        if checks.get('omitted_microcontours'):counts['cleanup']+=1;kinds.append('Удалены численные микроконтуры')
        if any('маркировка не помещается' in n for n in notes):counts['missing_labels']+=1;kinds.append('Не вся маркировка поместилась')
        if not p.get('links'):counts['without_connections']+=1;kinds.append('Без соединений шпильками')
        links=[]
        for extension in ('svg','dxf'):
            path=section_path(output/'sections_a4',p['id'],extension)
            if not p['metrics']['outer_contours']:
                review=section_path(output/'sections_a4',p['id'],extension,True)
                if path.is_file():
                    if review.is_file():
                        backup=previous_exports/path.relative_to(output/'sections_a4')
                        backup.parent.mkdir(parents=True,exist_ok=True);path.replace(backup)
                    else:
                        review.parent.mkdir(parents=True,exist_ok=True);path.replace(review)
                path=review
            if path.is_file():links.append('<a href="../%s">%s</a>'%(path.relative_to(output).as_posix(),extension.upper()))
        description='; '.join(kinds) or 'Проверено'
        csv_rows.append(dict(print_number=p['marking'],source_part_id=p['id'],layer=p['layer_number'],
            net_area_mm2=p['metrics']['net_area_mm2'],total_cut_mm=p['metrics']['total_cut_mm'],
            section_islands=checks.get('material_islands',p['metrics']['outer_contours']),
            microcontours_removed=len(checks.get('omitted_microcontours',[])),
            pin_connections=len(p.get('links',[])),status=description))
        rows.append('<tr data-review="%s"><td>%s</td><td>%s</td><td>%s</td><td>%.3f</td><td>%.3f</td><td>%s</td></tr>'%
            ('1' if kinds else '0',escape(p['marking']),escape(p['id']),' · '.join(links),p['metrics']['net_area_mm2'],
             p['metrics']['total_cut_mm'],escape(description)))
    with (destination/'parts_checks.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        if csv_rows:
            writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    nesting=report.get('sheet_nesting')
    nesting_text='' if not nesting else '<p>Дополнительная раскладка: %d листов для %d деталей; зазор ≥ %.3g мм, поля ≥ %.3g мм. Исходные отдельные страницы сохранены. Глобальный минимум числа листов не доказан. <a href="../Sheets/layout.json">Карта раскладки</a>.</p>'%(nesting['sheets'],nesting['parts'],nesting['required_gap_mm'],nesting['edge_mm'])
    text='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка деталей SliceToSheet</title>
<style>body{font:15px system-ui;margin:24px;color:#193041;background:#f6f8fa}p{max-width:1200px;line-height:1.5}table{border-collapse:collapse;width:100%%;background:white}td,th{padding:7px;border:1px solid #d8e0e7;text-align:left}th{position:sticky;top:0;background:#e4edf4}input{padding:10px;margin:12px 8px 12px 0}a{color:#1263a1}.notice{background:#fff5d9;padding:14px;max-width:1200px}</style>
<h1>Проверка деталей</h1>
<p>Всего: %(total)d. Несколько областей сечения: %(islands)d. Нет пригодного среднего контура: %(empty)d. Численные микроконтуры очищены у %(cleanup)d деталей. Неполная маркировка: %(missing_labels)d. Без шпилечных соединений: %(without_connections)d.</p>
<div class="notice">SVG/DXF показывают среднее сечение объёмной детали. Несколько областей на срезе могут соединяться на другой глубине 3D-детали. По такому срезу нельзя получить связную заготовку одним сквозным резом. Ступени численной оболочки не сглаживались произвольно. Очистка исключает только контуры шириной менее 0,01 мм и площадью менее 0,1 мм²; изменения зафиксированы в assembly.json. Исходная 3D-геометрия сохранена. Граф шпилек не учитывает клей и иные способы соединения.</div>
<p>Печатный номер — три числа через пробел. Дополнительным фрагментам одной пространственной ячейки выделены свободные номера отдельных страниц; повторов нет. Соответствие исходным идентификаторам — <a href="../print_addresses.csv">print_addresses.csv</a>.</p>
'''%counts
    text+=nesting_text+'<p><a href="parts_checks.csv">Таблица всех деталей CSV</a></p><input id="search" placeholder="Номер, исходный файл или замечание"><label><input id="only" type="checkbox">Только с замечаниями</label><table><thead><tr><th>Номер</th><th>Исходная деталь</th><th>Файлы</th><th>Площадь, мм²</th><th>Рез, мм</th><th>Проверки</th></tr></thead><tbody>'+''.join(rows)+'''</tbody></table><script>
const rows=[...document.querySelectorAll('tbody tr')];function filter(){const q=document.querySelector('#search').value.toLowerCase(),only=document.querySelector('#only').checked;for(const r of rows)r.hidden=(!r.textContent.toLowerCase().includes(q))||(only&&r.dataset.review!=='1')}document.querySelector('#search').oninput=filter;document.querySelector('#only').onchange=filter;
</script></html>'''
    (destination/'index.html').write_text(text,encoding='utf-8')
    return counts
