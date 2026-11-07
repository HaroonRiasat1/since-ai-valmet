"""Featherless image client with explicit modality checks and secret-safe errors."""
from dataclasses import dataclass,field
from pathlib import Path
from urllib.parse import quote,urlparse
import base64,json,os
import cv2,numpy as np
import requests

class VisionResponseError(RuntimeError):
    """Safe error text with provider output available for artifact preservation."""
    def __init__(self,message,response_text='',usage=None,finish_reason=None):
        super().__init__(message);self.response_text=response_text;self.usage=usage or {};self.finish_reason=finish_reason

@dataclass
class VisionSettings:
    api_key: str=field(repr=False)
    base_url: str='https://api.featherless.ai/v1'
    model: str='Qwen/Qwen3-VL-32B-Instruct'
    segmentation_model: Path=Path('models/sam2.1_l.pt')
    device: str='cpu'
    max_tokens: int=8192
    timeout: int=180
    max_objects: int=40

def load_settings(path: Path):
    values={}
    for line in path.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#'):continue
        if line.startswith('export '):line=line[7:]
        if '=' not in line:raise ValueError('Malformed .env entry; expected NAME=value')
        key,value=line.split('=',1);value=value.strip()
        if len(value)>=2 and value[0]==value[-1] and value[0] in "\"'":value=value[1:-1]
        values[key.strip()]=value
    def get(name,default=''):return os.environ.get(name,values.get(name,default))
    key=get('FEATHERLESS_API_KEY')
    if not key or key.lower() in {'your_api_key','your-key','paste_your_key_here'}:raise ValueError(f'Add FEATHERLESS_API_KEY to {path.resolve()}')
    if any(c.isspace() for c in key):raise ValueError('The API key contains whitespace; correct the .env entry')
    base=get('FEATHERLESS_BASE_URL','https://api.featherless.ai/v1').rstrip('/')
    parsed=urlparse(base)
    if parsed.scheme!='https' or parsed.netloc!='api.featherless.ai' or parsed.path!='/v1':raise ValueError('Expected the configured Featherless HTTPS /v1 endpoint')
    def integer(name,default,minimum,maximum):
        try:v=int(get(name,str(default)))
        except ValueError:raise ValueError(f'{name} must be an integer') from None
        if not minimum<=v<=maximum:raise ValueError(f'{name} is outside its supported range')
        return v
    return VisionSettings(key,base,get('FEATHERLESS_VISION_MODEL','Qwen/Qwen3-VL-32B-Instruct'),
        Path(get('ROOMSEG_SEGMENTATION_MODEL','models/sam2.1_l.pt')),get('ROOMSEG_DEVICE','cpu'),
        integer('ROOMSEG_VLM_MAX_TOKENS',8192,128,32768),integer('ROOMSEG_VLM_TIMEOUT',180,10,600),
        integer('ROOMSEG_VLM_MAX_OBJECTS',40,1,100))

def parse_json(text):
    text=text.strip()
    if text.startswith('```'):
        lines=text.splitlines();text='\n'.join(lines[1:-1]).strip()
    try:return json.loads(text)
    except (json.JSONDecodeError,TypeError):raise ValueError('Vision model did not return valid JSON; raw response saved for inspection') from None

def validate_objects(data,max_objects=40):
    if not isinstance(data,dict) or not isinstance(data.get('objects'),list):raise ValueError('Expected a JSON object containing an objects array')
    if len(data['objects'])>max_objects:raise ValueError('Vision model returned more objects than the configured pilot limit')
    seen=set();result=[]
    for obj in data['objects']:
        if not isinstance(obj,dict):raise ValueError('Invalid vision object')
        tag=obj.get('temporary_id');box=obj.get('bbox_xyxy');category=obj.get('category')
        if not isinstance(tag,str) or not tag or tag in seen:raise ValueError('Vision IDs must be nonempty and unique')
        if not isinstance(category,str) or not category.strip():raise ValueError('Vision object lacks a category')
        if not isinstance(box,list) or len(box)!=4 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) for x in box):raise ValueError('Expected four finite normalized XYXY coordinates')
        if min(box)<0 or max(box)>1000 or box[0]>=box[2] or box[1]>=box[3]:raise ValueError('Invalid or out-of-range normalized object box')
        if not isinstance(obj.get('visible_evidence'),str) or not obj['visible_evidence'].strip():raise ValueError('Each candidate needs visible supporting evidence')
        seen.add(tag);result.append(obj)
    return result

class FeatherlessVision:
    def __init__(self,settings,session=None):
        self.settings=settings;self.session=session or requests.Session();self.calls=0
        self.headers={'Authorization':'Bearer '+settings.api_key,'Content-Type':'application/json'}

    def _request(self,method,path,**kwargs):
        try:r=self.session.request(method,self.settings.base_url+path,headers=self.headers,timeout=self.settings.timeout,**kwargs)
        except requests.RequestException:raise RuntimeError('Featherless connection failed or timed out; no request was automatically retried') from None
        if r.status_code>=400:raise RuntimeError(f'Featherless returned HTTP {r.status_code}; check model access, account credits, or key configuration')
        try:return r.json()
        except ValueError:raise RuntimeError('Featherless returned a non-JSON API response') from None

    def check_model(self):
        m=self._request('GET','/models/'+quote(self.settings.model,safe=''))
        if m.get('id')!=self.settings.model:raise RuntimeError('Model detail response does not match the selected model')
        modalities=m.get('input_modalities',[])
        vision=(m.get('features',{}).get('image_input') is True or m.get('vision_supported') is True or 'image' in modalities)
        if not vision:raise RuntimeError('Selected Featherless model is not confirmed to accept images; refusing a text-only fallback')
        return dict(model=m['id'],image_input=True,context_length=m.get('context_length'),
                    availability=m.get('availability'),status=m.get('status'),authenticated=True)

    def ask(self,prompt,images,*,temperature=0,top_p=None,top_k=None,presence_penalty=None,repetition_penalty=None):
        content=[dict(type='text',text=prompt)]
        for rgb in images:
            ok,encoded=cv2.imencode('.png',np.ascontiguousarray(rgb[:,:,::-1]))
            if not ok:raise ValueError('Cannot encode the pilot image')
            content.append(dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(encoded).decode('ascii'))))
        self.calls+=1
        payload=dict(model=self.settings.model,messages=[dict(role='user',content=content)],temperature=temperature,max_tokens=self.settings.max_tokens,stream=False)
        payload.update({k:v for k,v in dict(top_p=top_p,top_k=top_k,presence_penalty=presence_penalty,repetition_penalty=repetition_penalty).items() if v is not None})
        data=self._request('POST','/chat/completions',json=payload)
        try:
            choice=data['choices'][0];text=choice['message']['content']
            if choice.get('finish_reason')=='length':raise VisionResponseError('Vision response was truncated; reduce object count or increase output token limit',text or '',data.get('usage',{}),'length')
            if not isinstance(text,str) or not text.strip():raise RuntimeError('Vision model returned no final text')
        except (KeyError,IndexError,TypeError):raise RuntimeError('Unexpected Featherless completion response') from None
        return text,data.get('usage',{})

DISCOVERY_PROMPT='''Inspect this industrial equipment-room image. Identify distinct, visible,
meaningful equipment and surrounding objects. Prioritize complete pump housings,
valve bodies, actuators, gauges, tanks, motors, electrical cabinets, and storage
equipment. A pipe run between real fittings/junctions can be an asset, but paint
bands, isolated surface patches, labels, walls, floor, ceiling, and scan artifacts
are not assets. Include a part separately only when visibly distinct and useful
for inspection. Do not guess unseen equipment or exact device models.

Return ONLY JSON: {"objects":[{"temporary_id":"p01","category":"pressure gauge",
"description":"round dial above the middle pump","bbox_xyxy":[xmin,ymin,xmax,ymax],
"visible_evidence":"visible features supporting this object",
"parent_temporary_id":null,"needs_closeup":false}]}.
All box coordinates MUST be normalized to 0..1000, X first then Y, relative to
the entire supplied image. Provide tight approximate boxes around whole objects.
Use a broad category or "unidentified component" when uncertain. Maximum {limit}
objects. Do not invent candidates merely to reach the limit.'''
