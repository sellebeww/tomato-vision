"""Jalankan sel Python notebook berurutan tanpa server/port Jupyter.

Notebook ini tidak memakai magic atau state kernel khusus. Output rich display
dan stdout berasal dari eksekusi aktual. Ini bukan uji melalui kernel Jupyter.
"""
from pathlib import Path
import base64, contextlib, io, sys
import nbformat
import IPython.display
from IPython.core.formatters import DisplayFormatter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def run():
    p=ROOT/'notebooks/IS794_UTS_Dataset_EDA.ipynb'
    notebook=nbformat.read(p,as_version=4)
    formatter=DisplayFormatter()
    state={'__name__':'__main__'}
    count=0
    original_display=IPython.display.display
    try:
        for cell in notebook.cells:
            if cell.cell_type!='code': continue
            count+=1; outputs=[]
            def display(*objects,**kwargs):
                for obj in objects:
                    data,metadata=formatter.format(obj)
                    data={k:base64.b64encode(v).decode() if isinstance(v,bytes) else v for k,v in data.items()}
                    outputs.append(nbformat.v4.new_output('display_data',data=data,metadata=metadata))
            IPython.display.display=display
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                exec(compile(cell.source,f'{p.name}:cell{count}','exec'),state)
            if stream.getvalue(): outputs.insert(0,nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue()))
            cell.outputs=outputs; cell.execution_count=count
    finally: IPython.display.display=original_display
    notebook.metadata['verification']={'method':'sequential Python execution without Jupyter kernel',
                                       'code_cells':count,'date':'2026-10-14',
                                       'limitation':'Local Jupyter kernel launch blocked by sandbox localhost port policy.'}
    nbformat.validate(notebook); nbformat.write(notebook,p)
    print(f'{count} sel kode berhasil dijalankan berurutan; output aktual disimpan. Kernel Jupyter belum diuji.')

if __name__=='__main__': run()
