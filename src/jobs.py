"""One bounded local training job; shell input and arbitrary commands are not accepted."""
from datetime import datetime, timezone
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
from src.config import OUTPUTS_DIR, ROOT_DIR
from src.manifest import prepare, relative
from src.workspace import dataset_readiness

class TrainingJobs:
    def __init__(self):
        self.lock=threading.Lock()
        self.process=None
        self.log=None
        self.info={'state':'idle'}

    def status(self):
        with self.lock:
            if self.process is not None and self.info['state'] in {'running','cancelling'}:
                code=self.process.poll()
                if code is not None:
                    self.info['state']='cancelled' if self.info['state']=='cancelling' else ('succeeded' if code==0 else 'failed')
                    self.info['exit_code']=code
                    self.log.close()
                    (ROOT_DIR/self.info['log']).with_suffix('.json').write_text(json.dumps(self.info,indent=2))
            result=dict(self.info)
            if 'log' in result:
                with (ROOT_DIR/result['log']).open('rb') as stream:
                    stream.seek(0,2)
                    stream.seek(max(0,stream.tell()-12000))
                    result['tail']=stream.read().decode('utf-8',errors='replace')
            return result

    def start(self, epochs=40):
        if type(epochs) is not int or not 1<=epochs<=100 :
            raise ValueError('Epoch harus 1–100.')
        self.status()
        with self.lock:
            if self.info['state'] in {'running','cancelling'}:
                raise ValueError('Training masih berjalan.')
            readiness=dataset_readiness()
            if not readiness['can_train']:
                raise ValueError('Dataset belum siap: '+' '.join(readiness['blockers']))
            dataset=prepare('own')
            name=datetime.now(timezone.utc).strftime('own_%Y%m%d_%H%M%S_')+secrets.token_hex(3)
            output=OUTPUTS_DIR/'experiments'/name
            logs=OUTPUTS_DIR/'jobs'
            logs.mkdir(parents=True,exist_ok=True)
            log_path=logs/(name+'.log')
            self.log=log_path.open('w')
            command=[sys.executable,'-u','-m','src.experiment','--dataset',str(dataset),
                     '--output',str(output),'--epochs',str(epochs),'--samples-per-class','300']
            try:
                self.process=subprocess.Popen(command,cwd=ROOT_DIR,stdout=self.log,stderr=subprocess.STDOUT,
                                              start_new_session=os.name!='nt')
            except OSError:
                self.log.close()
                raise
            self.info={'state':'running','output':relative(output),'log':relative(log_path),
                       'dataset':relative(dataset),'epochs':epochs}
            (logs/(name+'.json')).write_text(json.dumps(self.info,indent=2))
            return dict(self.info)

    def cancel(self):
        """Stop only the process group started by this manager, keeping all artifacts."""
        self.status()
        with self.lock:
            if self.info['state']=='running' and self.process.poll() is None:
                if os.name=='nt':
                    subprocess.run(['taskkill','/PID',str(self.process.pid),'/T','/F'],check=True,
                                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                else:
                    try:
                        os.killpg(self.process.pid,signal.SIGTERM)
                    except ProcessLookupError:
                        self.info['note']='Training ended before cancellation signal'
                self.info['state']='cancelling'
        return self.status()
