"""Read-only checks against a running local demo; does not save labels."""
import base64
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BASE='http://127.0.0.1:7860'
def call(path,body=None,token=None,host=None):
    headers={'Content-Type':'application/json'}
    if token:
        headers['X-CSRF-Token']=token
    if host:
        headers['Host']=host
    request=Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
    try:
        with urlopen(request,timeout=30) as response:
            return response.status,response.read()
    except HTTPError as error:
        return error.code,error.read()

def main():
    status,html=call('/')
    assert status==200 and b'Tomato Vision' in html
    status,body=call('/api/status')
    assert status==200
    settings=json.loads(body)
    assert settings['model_loaded']
    status,body=call('/api/photos')
    rows=json.loads(body)
    assert status==200 and rows
    status,body=call('/photo/'+rows[0]['sha256'])
    assert status==200 and body[:2]==b'\xff\xd8'
    assert call('/api/predict',{'image':'bad'})[0]==403
    assert call('/api/status',host='external.example')[0]==403
    assert call('/api/predict',{'image':'not base64'},settings['csrf'])[0]==400
    assert call('/api/labels',{'rows':[]},settings['csrf'])[0]==400
    image=Path('data/raw/TOM001_segar.png').read_bytes()
    status,body=call('/api/predict',{'image':base64.b64encode(image).decode()},settings['csrf'])
    result=json.loads(body)
    assert status==200 and abs(sum(result['probabilities'].values())-1)<1e-5
    assert result['mode']==settings['mode']
    assert abs(sum(result['raw_probabilities'].values())-1)<1e-5
    assert result['stability']['views']==4
    assert 0<=result['stability']['agreement']<=1
    assert 'edge_variance' in result['quality']
    status,body=call('/api/dashboard')
    dashboard=json.loads(body)
    assert status==200 and dashboard['dataset']['own_images']==len(rows)
    assert call('/api/annotations')[0]==200
    assert call('/api/report')[0]==200
    assert call('/artifact/confusion_matrix.png')[0]==200
    assert call('/style.css')[0]==200 and call('/app.js')[0]==200
    assert call('/api/import',{'own_photo':False},settings['csrf'])[0]==400
    assert call('/api/train',{'epochs':101},settings['csrf'])[0]==400
    if not dashboard['dataset']['can_train']:
        assert call('/api/train',{'epochs':1},settings['csrf'])[0]==400
    assert call('/api/predict',[],settings['csrf'])[0]==400
    if dashboard['job']['state']=='idle':
        assert call('/api/cancel-training',{},settings['csrf'])[0]==200
    print(json.dumps({'http_checks':'passed (original and extended endpoints)','photos':len(rows),'prediction':result},indent=2))

if __name__=='__main__':
    main()
