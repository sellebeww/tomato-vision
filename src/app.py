"""Local-only prediction and labeling demo. No external service or upload storage."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import secrets
import shutil
import threading
from urllib.parse import urlparse
import pandas as pd
from PIL import Image, ImageOps
from src.config import DATA_DIR, ROOT_DIR, resolve_path
from src.manifest import init_labels, validate_annotation_rows
from src.predict import Predictor
from src.workspace import annotation_snapshot, save_annotations, import_own_photo, dataset_readiness
from src.jobs import TrainingJobs


def make_handler(predictor=None):
    token=secrets.token_hex(24)
    lock=threading.Lock()
    path=DATA_DIR/'annotations.csv'
    state={'predictor':predictor}
    jobs=TrainingJobs()

    def dashboard():
        current=state['predictor']
        model=None
        if current:
            evaluation=current.run/'evaluation.json'
            model={key:current.meta[key] for key in ('mode','classes','config','best_epoch','epochs_run',
                   'model_parameters','validation','train_clean','production_ready','dataset_fingerprint')}
            model['run']=str(current.run.relative_to(ROOT_DIR)) if current.run.is_relative_to(ROOT_DIR) else current.run.name
            model['evaluation']=json.loads(evaluation.read_text()) if evaluation.exists() else None
        return {'title':'Klasifikasi Tingkat Kesegaran Tomat Menggunakan Convolutional Neural Network (CNN) Berbasis TensorFlow',
                'dataset':dataset_readiness(),'model':model,'job':jobs.status()}

    class Handler(BaseHTTPRequestHandler):
        def send(self,status,body,content_type='application/json'):
            if isinstance(body,(dict,list)):
                body=json.dumps(body,ensure_ascii=False).encode()
            elif isinstance(body,str):
                body=body.encode()
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' blob: data:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def local_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}

        def do_GET(self):
            if not self.local_host():
                return self.send(403,{'error':'Localhost only'})
            route=urlparse(self.path).path
            try:
                if route=='/':
                    return self.send(200,(ROOT_DIR/'web/index.html').read_bytes(),'text/html; charset=utf-8')
                if route in {'/logo.png','/favicon.png','/apple-touch-icon.png'}:
                    return self.send(200,(ROOT_DIR/'web'/route[1:]).read_bytes(),'image/png')
                if route in {'/app.js','/style.css'}:
                    kind='text/javascript' if route.endswith('.js') else 'text/css'
                    return self.send(200,(ROOT_DIR/'web'/route[1:]).read_bytes(),kind+'; charset=utf-8')
                if route=='/api/status':
                    current=state['predictor']
                    return self.send(200,{'csrf':token,'model_loaded':current is not None,
                                         'mode':current.meta['mode'] if current else None})
                if route in {'/api/dashboard','/api/report'}:
                    with lock:
                        return self.send(200,dashboard())
                if route=='/api/job':
                    return self.send(200,jobs.status())
                if route=='/api/annotations':
                    with lock:
                        return self.send(200,annotation_snapshot())
                if route in {'/artifact/confusion_matrix.png','/artifact/learning_curve.png'}:
                    current=state['predictor']
                    artifact=current.run/route.rsplit('/',1)[-1] if current else None
                    if not artifact or not artifact.exists():
                        return self.send(404,{'error':'Grafik belum tersedia.'})
                    return self.send(200,artifact.read_bytes(),'image/png')
                if route=='/api/photos':
                    return self.send(200,pd.read_csv(path,keep_default_na=False).to_dict('records'))
                if route.startswith('/photo/'):
                    digest=route.rsplit('/',1)[-1]
                    df=pd.read_csv(path,keep_default_na=False)
                    matches=df[df.sha256==digest]
                    if len(matches)!=1:
                        return self.send(404,{'error':'Photo not found'})
                    with Image.open(resolve_path(matches.iloc[0].filepath)) as image:
                        image=ImageOps.exif_transpose(image).convert('RGB')
                        image.thumbnail((640,640))
                        buffer=io.BytesIO()
                        image.save(buffer,format='JPEG',quality=85)
                    return self.send(200,buffer.getvalue(),'image/jpeg')
                return self.send(404,{'error':'Not found'})
            except (ValueError,OSError,Image.DecompressionBombError) as error:
                return self.send(400,{'error':str(error)})

        def do_POST(self):
            if not self.local_host() or self.headers.get('X-CSRF-Token')!=token:
                return self.send(403,{'error':'Invalid session token'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=14_000_000:
                    return self.send(413,{'error':'Maximum upload 10 MB'})
                if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                    return self.send(415,{'error':'JSON required'})
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict):
                    raise ValueError('JSON object required')
                if self.path=='/api/predict':
                    if state['predictor'] is None:
                        return self.send(409,{'error':'No model. Train first or launch with --run.'})
                    raw=base64.b64decode(data['image'],validate=True)
                    if len(raw)>10_000_000:
                        return self.send(413,{'error':'Maximum upload 10 MB'})
                    with lock:
                        result=state['predictor'].predict(io.BytesIO(raw))
                    return self.send(200,result)
                if self.path=='/api/labels':
                    with lock:
                        snapshot=save_annotations(data['rows'],data.get('revision'))
                    return self.send(200,{'saved':len(snapshot['rows']),'revision':snapshot['revision']})
                if self.path=='/api/import':
                    if data.get('own_photo') is not True:
                        raise ValueError('Konfirmasi bahwa foto ini Anda potret sendiri.')
                    raw=base64.b64decode(data['image'],validate=True)
                    with lock:
                        result=import_own_photo(raw,data['name'])
                    return self.send(200,result)
                if self.path=='/api/train':
                    with lock:
                        result=jobs.start(data.get('epochs',40))
                    return self.send(202,result)
                if self.path=='/api/reload-model':
                    with lock:
                        state['predictor']=Predictor()
                    return self.send(200,{'loaded':True})
                if self.path=='/api/cancel-training':
                    return self.send(200,jobs.cancel())
                return self.send(404,{'error':'Not found'})
            except (ValueError,KeyError,TypeError,OSError,Image.DecompressionBombError) as error:
                return self.send(400,{'error':str(error)})

    Handler.training_jobs=jobs
    return Handler


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run')
    parser.add_argument('--port',type=int,default=7860)
    parser.add_argument('--labels-only',action='store_true')
    args=parser.parse_args()
    if not (DATA_DIR/'annotations.csv').exists():
        init_labels()
    predictor=None
    if not args.labels_only:
        try:
            predictor=Predictor(args.run)
        except (OSError,ValueError,KeyError) as error:
            print(f'Labeling is available; model unavailable: {error}',flush=True)
    handler=make_handler(predictor)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler)
    print(f'Tomato Vision: http://127.0.0.1:{server.server_port}',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Stopping server; unfinished training artifacts are preserved.',flush=True)
    finally:
        handler.training_jobs.cancel()
        server.server_close()


if __name__=='__main__':
    main()
