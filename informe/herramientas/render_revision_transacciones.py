"""Renderiza y extrae texto del PDF final para revisión visual documentada."""
from pathlib import Path
import json
import re
import subprocess
from PIL import Image,ImageDraw,ImageOps

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'revision/transacciones'
DEST.mkdir(parents=True,exist_ok=True)
info=subprocess.run(['pdfinfo','main.pdf'],cwd=ROOT,check=True,capture_output=True,text=True).stdout
pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1])
for pattern in ('pagina-*.png','contacto_*.png'):
    for old in DEST.glob(pattern):
        assert old.resolve().parent == DEST.resolve()
        old.unlink()
subprocess.run(['pdftoppm','-r','75','-png','main.pdf',str(DEST/'pagina')],cwd=ROOT,check=True)
subprocess.run(['pdftotext','-layout','main.pdf',str(ROOT/'evidencias/transacciones/texto_pdf.txt')],cwd=ROOT,check=True)
files=sorted(DEST.glob('pagina-*.png'))
assert len(files)==pages
sheets=[]
for index in range(0,len(files),12):
    selected=files[index:index+12];canvas=Image.new('RGB',(1000,4*375),'#cfd4d9');draw=ImageDraw.Draw(canvas)
    for j,p in enumerate(selected):
        im=Image.open(p).convert('RGB');im.thumbnail((238,343))
        x=(j%4)*250+(250-im.width)//2;y=(j//4)*375+23
        canvas.paste(im,(x,y));draw.text(((j%4)*250+8,(j//4)*375+5),f'PDF {index+j+1}',fill='black')
    target=DEST/f'contacto_{index//12+1:02}.png';canvas.save(target);sheets.append(target.name)
(ROOT/'evidencias/transacciones/render.json').write_text(json.dumps(dict(pages=pages,dpi=75,rendered_pages=len(files),contact_sheets=sheets,pdfinfo=info),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(pages=pages,rendered_pages=len(files),contact_sheets=sheets)))
