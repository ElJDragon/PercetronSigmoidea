"""Compila el informe unificado sin instalar paquetes ni ejecutar experimentos."""
from pathlib import Path
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidencias/transacciones'
command=['pdflatex','--disable-installer','-interaction=nonstopmode','-halt-on-error','main.tex']
results=[]
for attempt in range(1,4):
    with (E/f'compilacion_pasada_{attempt}.txt').open('w',encoding='utf-8') as f:
        completed=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,text=True)
    log=(ROOT/'main.log').read_text(encoding='utf-8',errors='replace')
    record=dict(attempt=attempt,exit_code=completed.returncode,overfull='Overfull' in log,undefined='undefined' in log,rerun='Rerun to get' in log or 'Rerun to get outlines' in log)
    results.append(record)
    if completed.returncode:raise SystemExit('Compilación fallida; consultar registro de pasada')
    if attempt>=2 and not any(record[k] for k in ['overfull','undefined','rerun']):break
match=re.search(r'Output written on main.pdf \((\d+) pages',log)
summary=dict(command=command,passes=results,pages=int(match[1]) if match else None,
             successful=not any(results[-1][k] for k in ['overfull','undefined','rerun']),duplicate_destinations='duplicate ignored' in log)
(E/'compilacion.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary))
raise SystemExit(0 if summary['successful'] and not summary['duplicate_destinations'] else 1)
