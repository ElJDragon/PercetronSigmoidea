"""Registra la inspección humana del render final y verifica integridad documental."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
EVID=ROOT/'evidencias/transacciones'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

render=read(EVID/'render.json')
visual=dict(status='reviewed',pdf_sha256=sha(ROOT/'main.pdf'),pages=render['pages'],
    method='Poppler a 75 dpi; inspección de todas las páginas en hojas de contacto y lectura de páginas completas seleccionadas.',
    contact_sheets_reviewed=render['contact_sheets'],all_pages_reviewed=list(range(1,render['pages']+1)),
    full_pages_reviewed=[37,39,40,43,45,48,51,54],
    checks=dict(tables_readable=True,equations_readable=True,figures_complete=True,
                captions_readable=True,no_clipping=True,no_overlaps=True,pagination_reviewed=True),
    pagination_adjustments=['Resumen separado del índice general.',
        'Índice de figuras compacto sin entrada aislada.',
        'Métricas de desarrollo junto a explicación de convergencia.',
        'Interpretación PCA anterior a sus gráficos para evitar párrafo aislado.'])
save(EVID/'revision_visual.json',visual)
subprocess.run([sys.executable,str(ROOT/'herramientas/verificar_transacciones_informe.py')],cwd=BASE,check=True)
before=read(EVID/'integridad_experimentos_antes.json')
after={n:sha(BASE/n) for n in before}
assert before==after
sales=read(EVID/'ventas_preservadas.json')
assert all(sha(BASE/n)==v for n,v in sales.items())
save(EVID/'integridad_experimentos_despues.json',after)
manifest=EVID/'manifiesto_documental.json'
paths=[p for p in ROOT.rglob('*') if p.is_file() and p!=manifest
       and not any(part in {'__pycache__','version_previa'} for part in p.relative_to(ROOT).parts)
       and p.suffix.lower() in {'.tex','.pdf','.png','.json','.py','.md','.txt','.log','.aux','.toc','.lof','.lot'}]
validation=read(EVID/'verificacion_informe.json')
save(manifest,dict(status='successful',pages=validation['pages'],figures=39,tables=26,
    experimental_files_preserved=len(after),sales_artifacts_preserved=len(sales),
    pdf_sha256=sha(ROOT/'main.pdf'),files={str(p.relative_to(BASE)):sha(p) for p in sorted(paths)},
    scope='Actualización documental local; sin Git, SQL, entrenamiento, predicciones ni PCA nuevos.'))
print(json.dumps(dict(status='successful',pages=validation['pages'],checks=validation['documentary_checks'],
    experimental_files_preserved=len(after),sales_artifacts_preserved=len(sales),visual_review='reviewed')))
