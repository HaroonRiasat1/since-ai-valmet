"""Image-by-image, evidence-linked vocabulary discovery for text grounding."""
from collections import defaultdict
from pathlib import Path
from dataclasses import replace
from urllib.parse import quote
import hashlib,html,json,re,time
import cv2
from .io import atomic_json,digest_file,fingerprint,run_lock
from .vlm import FeatherlessVision,VisionResponseError,load_settings,parse_json

# Qwen's model-card VL defaults avoid greedy decoding. Text-only consolidation
# keeps the client's deterministic defaults for exact membership copying.
VISION_GENERATION=dict(temperature=.7,top_p=.8,top_k=20,presence_penalty=1.5,repetition_penalty=1.)

DISCOVER='''Inspect this industrial equipment-room preview. List the distinct visible
object TYPES useful for text-grounded segmentation: pumps, motors, valves,
handwheels, actuators, gauges, tanks, cabinets, pipes/fittings, and surrounding
chairs, carts, hand trucks, ladders, tools, containers, boxes, doors and appliances.
Do not assume these objects exist. Search systematically for small instruments
as well as large equipment. Distinguish a circulation pump/control head from a
valve actuator when visible evidence supports it. A green arrow/pipe label is not
a gauge; a gauge needs a round instrument body or dial. Ignore paint patches,
scan artifacts and arbitrary surface fragments. Include a useful serviceable
part separately only when visibly distinct. Do not guess makes or hidden parts.
Use a broad category or 'unknown component' if uncertain. This task is object
vocabulary, not counting instances. Combine repeated identical types in this
image, but retain visibly different handle/mechanism forms. Colour, size and a
partly hidden black centre are attributes, not automatically different classes.

Return ONLY JSON {"observations":[{"category":"handwheel valve",
"parent_category":"valve","description":"red valve body with a black round handwheel",
"attributes":{"color":["red","black"],"size":[],"shape":["round"]},
"distinguishing_features":["handwheel"],"uncertainty":"low",
"visible_evidence":"round handwheel attached to an inline valve body",
"detection_phrase":"red handwheel valve"}]}.
Use short English phrases, descriptions and evidence of at most 18 words each,
at most 3 attributes per field and 3 distinguishing features. Uncertainty is
low, medium or high, not a numeric confidence. Maximum 35 observations. Return
an empty array if nothing meaningful is visible. Never fill a quota.'''

# Same discovery task, compact wire representation. Expanded public JSON remains
# identical. Earlier verbose-format results are explicitly compatible caches.
DISCOVER_COMPACT=DISCOVER.split('\nReturn ONLY JSON:')[0]+'''
Return ONLY JSON {"observations":[[category,parent_category,description,
colors,sizes,shapes,distinguishing_features,uncertainty,visible_evidence,detection_phrase]]}.
Each row has EXACTLY 10 columns in that order. Colors/sizes/shapes/features are
arrays of at most 3 short strings; use [] when unknown. Other columns are short
English strings. Uncertainty is low, medium or high. Description at most 10
words, evidence at most 6 words, detection phrase at most 6 words. Example row:
["chair","chair","green fabric chair with metal legs",["green"],[],[],["metal legs"],"low","seat, backrest and legs visible","green fabric chair"].
Maximum 35 rows; combine repeated identical types. Do not fill a quota. No text
outside JSON. Do not repeat field names in rows.'''

CLASSIFY='''Consolidate these visible object-type observations, not physical instances.
For EACH group_id return exactly one canonical parent and optional subtype.
Parent is a broad category (valve, pump, chair, tank, cabinet, pipe fitting ...).
Subtype is only a visibly supported, structurally distinct form, such as
handwheel valve versus lever-operated valve. Colour, size, view angle, material,
missing detail and 'black centre' alone must NOT create subtypes. 'Red valve'
and 'red valve with black centre' therefore map to the same broad valve type
unless independent visible mechanism evidence supports a subtype. Preserve a
real handle/mechanism distinction. Unknown/ambiguous forms stay at the broad
parent: do not infer hidden handles. Use singular short English noun phrases,
no colours in canonical names, no makes/models. Descriptions must be grounded
in the supplied evidence. Every output needs its input group_id; no new IDs.
Return ONLY JSON {"assignments":[{"group_id":"g...","parent":"valve",
"subtype":"handwheel valve","description":"Valve with a round manual handwheel"}]}.
Use null subtype for generic or uncertain forms. Input:\n'''

CLASSIFY_GROUPED=CLASSIFY.split('Return ONLY JSON:')[0].replace(
    'For EACH group_id return exactly one canonical parent and optional subtype.',
    'Assign every input to one canonical parent and optional subtype.').replace(
    'Every output needs its input group_id; no new IDs.',
    'Every input must be represented exactly once; no invented members.')+'''
Compare the inputs together and merge synonymous/partial descriptions of the
same structural type into one output type. Preserve different visible mechanisms.
Return ONLY JSON {"types":[{"parent":"valve","subtype":"handwheel valve",
"description":"Valve with a round manual handwheel","members":[0,2,5]},
{"parent":"valve","subtype":null,"description":"Generic valve","members":[1]}]}.
Use each input's small numeric index in members; do not copy long group hashes.
All input indices MUST occur exactly once across types. Descriptions at most
10 words. Input:\n'''

MERGE_PARENTS='''Merge ONLY strictly synonymous category names. This is NOT a task
to group related things into a broad functional bucket. Default to one cluster
per input when equivalence is uncertain. Keep hose separate from cable/wire,
box separate from bucket or rolled material, hand truck separate from cart,
pipe separate from pipe fitting or pipe label, vacuum cleaner separate from
generic appliance, and pump separate from valve/actuator/motor. A shared use or
part/whole relationship is NOT synonymy. Cabinet/electrical enclosure may be
synonyms if both input IDs actually exist. Use the name of one supplied member
as canonical name, no invented names. Return each input ID exactly once as a
member, never use category strings as IDs. Return ONLY JSON:
{"clusters":[{"name":"canonical input name","description":"short description",
"members":["p000"]}]}. Input objects:\n'''

MERGE_TYPES='''Reconcile the subtype candidates under ONE parent category.
Merge synonyms and incomplete appearance descriptions of the same structural
type. Colour, size, angle, material or a black centre alone do not define a new
type. Preserve supported structural differences such as handwheel versus lever.
Missing evidence is not a difference: uncertain/generic or appearance-only
candidates must map to name:null (the broad parent), not an invented subtype.
All input member IDs must occur exactly once, no invented members. Keep names
short, singular and free of colour words. Return ONLY JSON {"clusters":[
{"name":"handwheel valve","description":"Valve with a round manual handwheel",
"members":["t..."]}]}. Input:\n'''

def norm(value):
    return re.sub(r'[^a-z0-9]+',' ',value.casefold()).strip()

def parse_catalog_response(text):
    try:return parse_json(text)
    except ValueError:
        # Some providers append a prose note despite a JSON-only instruction.
        # Decode one complete leading JSON value without repairing its contents.
        # Schema/evidence validators still apply; a second JSON value is rejected.
        clean=text.strip()
        if clean.startswith('```'):clean=clean.split('\n',1)[-1].lstrip()
        try:
            value,end=json.JSONDecoder().raw_decode(clean)
            tail=clean[end:].strip()
            if tail.startswith('```'):tail=tail[3:].strip()
            if tail.startswith(('{','[',',')):
                # Accept only a stream of complete observation-row arrays, not
                # multiple conflicting wrapper objects or a partial document.
                values=[value]
                while tail:
                    if tail.startswith(','):tail=tail[1:].lstrip()
                    item,stop=json.JSONDecoder().raw_decode(tail);values.append(item);tail=tail[stop:].strip()
                if not all(isinstance(v,list) and len(v)>=6 and all(isinstance(x,str) for x in v[:3]) and isinstance(v[-3],str) and v[-3] in {'low','medium','high'} for v in values):raise ValueError('Multiple JSON documents')
                return dict(observations=values,_serialization_notes=['Complete individual JSON row arrays joined without altering source claims'])
            if tail.startswith(('}',']')):raise ValueError('Unexpected trailing JSON delimiter')
            return value
        except (json.JSONDecodeError,ValueError):raise ValueError('Catalog response has no unambiguous complete leading JSON value; raw response retained') from None

def short(value,label,maximum=240):
    if not isinstance(value,str) or not value.strip() or len(value)>maximum:raise ValueError(f'Invalid {label}')
    return ' '.join(value.split())

def string_list(value,label,maximum=6):
    # Providers occasionally quote an empty array in compact rows. Its meaning
    # is unambiguous and can be normalized without guessing a missing value.
    if value=='[]':value=[]
    elif isinstance(value,str):value=[value]
    if not isinstance(value,list) or len(value)>maximum:raise ValueError(f'Invalid {label}')
    return list(dict.fromkeys(short(v,label,100) for v in value))

def subtype_name(value):
    if value is None or (isinstance(value,str) and norm(value) in {'null','none','n a','unspecified','generic'}):return None
    return short(value,'canonical subtype',100)

def validate_observations(data):
    rows=data.get('observations') if isinstance(data,dict) else data if isinstance(data,list) else None
    if isinstance(rows,list) and len(rows)>=6 and all(isinstance(x,str) for x in rows[:3]) and isinstance(rows[-3],str) and rows[-3] in {'low','medium','high'}:rows=[rows]
    if not isinstance(rows,list) or len(rows)>80:raise ValueError('Expected a bounded array of at most 80 visible type observations')
    result=[]
    for row in rows:
        notes=list(data.get('_serialization_notes',[])) if isinstance(data,dict) else []
        if len(rows)>35:notes.append('Provider exceeded the requested 35-row target; supplied observations retained for consolidation')
        if isinstance(row,list):
            colours={'red','black','blue','white','green','yellow','orange','purple','brown','gray','grey','silver','beige','pink','transparent'}
            if len(row)!=10:
                # Narrow lossless recovery of a provider flattening multiple
                # colour strings before empty size/shape fields. Never infer
                # missing category, evidence, confidence or unknown attributes.
                middle=row[3:-3]
                if len(middle)>4 and middle[-2]==[] and all(isinstance(v,str) and norm(v) in colours for v in middle[:-2]):
                    row=row[:3]+[middle[:-2],[],[],middle[-1]]+row[-3:];notes.append('Flattened explicit colour strings restored to colour array')
                elif len(row)>=6 and isinstance(row[-3],str) and row[-3] in {'low','medium','high'}:
                    # The six essential semantic fields are present. Attribute
                    # positions are ambiguous, so keep their literal strings as
                    # untyped visible features and leave typed fields unknown.
                    features=[]
                    for value in middle:
                        features.extend(string_list(value,'untyped attribute',24))
                    row=row[:3]+[[],[],[],list(dict.fromkeys(features))]+row[-3:]
                    notes.append('Incomplete attribute columns: literal tokens retained as untyped features; color/size/shape unknown')
                else:raise ValueError('Compact observation must contain the six essential semantic fields')
            if any(isinstance(row[i],str) for i in [3,4,5,6]):notes.append('Scalar attribute/feature converted to a one-element array')
            row=dict(category=row[0],parent_category=row[1],description=row[2],
                attributes=dict(color=row[3],size=row[4],shape=row[5]),distinguishing_features=row[6],
                uncertainty=row[7],visible_evidence=row[8],detection_phrase=row[9])
        if not isinstance(row,dict):raise ValueError('Observation must be an object')
        attributes=row.get('attributes')
        if not isinstance(attributes,dict):raise ValueError('Observation attributes must be an object')
        uncertainty=row.get('uncertainty')
        if uncertainty not in {'low','medium','high'}:raise ValueError('Invalid observation uncertainty')
        attrs={k:string_list(attributes.get(k,[]),k) for k in ['color','size','shape']}
        if notes:
            for key in ['size','shape']:
                displaced=[v for v in attrs[key] if norm(v) in colours]
                if displaced:
                    attrs['color']=list(dict.fromkeys(attrs['color']+displaced));attrs[key]=[v for v in attrs[key] if v not in displaced]
                    notes.append(f'Explicit colour string moved from {key} to color')
        result.append(dict(category=short(row.get('category'),'category',100),
            parent_category=short(row.get('parent_category'),'parent category',100),
            description=short(row.get('description'),'description'),visible_evidence=short(row.get('visible_evidence'),'visible evidence'),
            attributes=attrs,
            distinguishing_features=string_list(row.get('distinguishing_features',[]),'distinguishing features',24),
            uncertainty=uncertainty,detection_phrase=short(row.get('detection_phrase'),'detection phrase',160),
            **(dict(normalization_notes=notes) if notes else {})))
    return result

def validate_partition(data,members,nullable=False):
    rows=data.get('clusters') if isinstance(data,dict) else None
    if not isinstance(rows,list):raise ValueError('Expected consolidation clusters')
    expected=set(members);seen=set();names={};result=[]
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Invalid cluster')
        name=subtype_name(row.get('name')) if nullable else short(row.get('name'),'canonical class name',100)
        key=norm(name) if name else None
        ids=row.get('members')
        if not isinstance(ids,list) or not ids or any(not isinstance(i,str) or i not in expected or i in seen for i in ids) or len(set(ids))!=len(ids):raise ValueError('Consolidation has missing, duplicate, or unsupported members')
        seen.update(ids);description=short(row.get('description'),'class description')
        if key in names:names[key]['members'].extend(ids)
        else:
            cluster=dict(name=name,description=description,members=list(ids));names[key]=cluster;result.append(cluster)
    if seen!=expected:raise ValueError('Consolidation did not cover every input member')
    return result

def group_observations(observations):
    groups={}
    for o in observations:
        key=fingerprint(dict(category=norm(o['category']),parent=norm(o['parent_category']),features=sorted(norm(f) for f in o['distinguishing_features'])))[:20]
        gid='g_'+key
        if gid not in groups:groups[gid]=dict(group_id=gid,category=o['category'],parent_category=o['parent_category'],features=o['distinguishing_features'],observations=[])
        groups[gid]['observations'].append(o)
    return sorted(groups.values(),key=lambda g:g['group_id'])

def expand_type_assignments(data,inputs,require_complete=True,issues=None):
    expected={g['group_id'] for g in inputs};result={};conflicts=set();issues=issues if issues is not None else []
    def add(gid,row):
        if not isinstance(gid,str) or gid not in expected:
            if require_complete:raise ValueError('Unsupported type group assignment')
            issues.append(dict(kind='unsupported member rejected',member=gid));return
        candidate=dict(group_id=gid,parent=short(row.get('parent'),'canonical parent',100),
            subtype=subtype_name(row.get('subtype')),description=short(row.get('description'),'canonical description'))
        if gid in conflicts:return
        if gid in result:
            if require_complete:raise ValueError('Duplicate type group assignment')
            old=result[gid];same=norm(old['parent'])==norm(candidate['parent']) and norm(old['subtype'] or '')==norm(candidate['subtype'] or '')
            if not same and norm(old['parent'])==norm(candidate['parent']):
                result[gid]=dict(group_id=gid,parent=old['parent'],subtype=None,description=old['parent'],
                    classification_uncertainty='high',classification_note='Overlapping subtype suggestions; agreed broad parent retained')
                issues.append(dict(kind='overlapping subtypes mapped to agreed parent',group_id=gid));return
            issues.append(dict(kind='redundant membership removed' if same else 'conflicting membership withheld',group_id=gid))
            if not same:result.pop(gid);conflicts.add(gid)
            return
        result[gid]=candidate
    if isinstance(data,dict) and isinstance(data.get('assignments'),list):
        for row in data['assignments']:
            if not isinstance(row,dict):raise ValueError('Invalid type assignment')
            add(row.get('group_id'),row)
    elif isinstance(data,dict) and isinstance(data.get('types'),list):
        for row in data['types']:
            if not isinstance(row,dict) or not isinstance(row.get('members'),list) or not row['members']:raise ValueError('Empty/invalid grouped type')
            for member in row['members']:
                if isinstance(member,str) and member.isdigit():member=int(member)
                if type(member) is int:
                    if not 0<=member<len(inputs):
                        if require_complete:raise ValueError('Unsupported numeric type member')
                        issues.append(dict(kind='unsupported numeric member rejected',member=member));continue
                    member=inputs[member]['group_id']
                add(member,row)
    else:raise ValueError('Missing grouped type assignments')
    if require_complete and set(result)!=expected:raise ValueError('Missing type group assignments')
    return result

def protected_family(name):
    """Conservative guard against merging visibly different broad object types."""
    value=norm(name)
    families=[('pipe label',['pipe label','pipe marker']),('pipe fitting',['pipe fitting','elbow fitting']),
      ('hand truck',['hand truck','dolly']),('vacuum cleaner',['vacuum cleaner']),
      ('cabinet',['cabinet','electrical enclosure']),('cable',['cable','wire']),('hose',['hose']),
      ('pipe',['pipe']),('pump',['pump']),('motor',['motor']),('actuator',['actuator']),('valve',['valve']),
      ('cart',['cart','trolley']),('chair',['chair']),('bucket',['bucket','pail']),('box',['box']),
      ('tank',['tank']),('appliance',['appliance']),('rolled material',['rolled material']),('unknown component',['unknown component'])]
    return next((family for family,terms in families if any(t in value for t in terms)),None)

def reconcile_parent_clusters(data,parents):
    inputs={f'p{i:03}':name for i,name in enumerate(parents)}
    clusters=validate_partition(data,inputs);result=[]
    for c in clusters:
        members=[inputs[key] for key in c['members']]
        protected={protected_family(name) for name in members}-{None}
        if len(protected)>1:
            result.extend(dict(name=name,description=f'Visible {name} category',members=[name]) for name in members)
            continue
        name=c['name'] if norm(c['name']) in {norm(m) for m in members} else min(members,key=lambda m:(len(m),m))
        result.append(dict(c,name=name,members=members))
    return result

class StableIDs:
    def __init__(self,path):
        self.path=path;self.data=json.loads(path.read_text()) if path.exists() else dict(keys={},next=1,redirects={})
        self.claimed=set()
        for key,identifier in list(self.data['keys'].items()):
            if key.startswith('child|') and key.rsplit('|',1)[-1] in {'null','none','n a','unspecified','generic'}:
                self.data['redirects'][identifier]=key.split('|')[1]
    def assign(self,keys):
        keys=sorted(set(keys));existing={self.data['keys'][k] for k in keys if k in self.data['keys']}
        available=existing-self.claimed
        if available:identifier=min(available)
        else:identifier=f'class_{self.data["next"]:06}';self.data['next']+=1
        for old in available-{identifier}:self.data['redirects'][old]=identifier
        for key in keys:self.data['keys'][key]=identifier
        self.claimed.add(identifier)
        return identifier
    def save(self):atomic_json(self.path,self.data)

class Catalog:
    def __init__(self,images,output,env,retry_failed=False):
        self.images=images.resolve();self.output=output.resolve();self.settings=replace(load_settings(env),timeout=300)
        self.client=FeatherlessVision(self.settings);self.retry_failed=retry_failed;self.new_usage=[]
        self.requests=self.output/'requests';self.requests.mkdir(exist_ok=True)
        self.files=sorted(p for p in self.images.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'})
        if not self.files:raise ValueError('No supported preview images found')

    def request_key(self,prompt,images):
        return fingerprint(dict(model=self.settings.model,prompt=prompt,images=[hashlib.sha256(i.tobytes()).hexdigest() for i in images]))

    def request(self,prompt,images=None,force=False):
        images=images or [];key=self.request_key(prompt,images)
        path=self.requests/f'{key}.json'
        if path.exists() and not force:
            saved=json.loads(path.read_text())
            if saved.get('truncated'):raise VisionResponseError('Cached response is truncated; explicit retry required',saved['response'],saved['usage'],'length')
            return parse_catalog_response(saved['response'])
        if len(prompt)>60000:raise ValueError('Consolidation exceeds the bounded request size; split the inputs')
        if path.exists():
            prior=json.loads(path.read_text());archive=self.requests/f'{key}.prior-{fingerprint(prior)[:12]}.json'
            if not archive.exists():atomic_json(archive,prior)
        generation=VISION_GENERATION if images else {}
        try:raw,usage=self.client.ask(prompt,images,**generation)
        except VisionResponseError as exc:
            atomic_json(path,dict(identity=key,model=self.settings.model,generation=generation,prompt=prompt,image_hashes=[hashlib.sha256(i.tobytes()).hexdigest() for i in images],response=exc.response_text,usage=exc.usage,truncated=True,finish_reason=exc.finish_reason))
            self.new_usage.append(exc.usage);raise
        atomic_json(path,dict(identity=key,model=self.settings.model,generation=generation,prompt=prompt,image_hashes=[hashlib.sha256(i.tobytes()).hexdigest() for i in images],response=raw,usage=usage))
        self.new_usage.append(usage)
        return parse_catalog_response(raw)

    def extract(self,files):
        folder=self.output/'images';folder.mkdir(exist_ok=True);previews=self.output/'previews';previews.mkdir(exist_ok=True)
        result=[]
        for index,path in enumerate(files,1):
            started=time.perf_counter();image_id=path.name;image_hash=digest_file(path)
            identity=fingerprint(dict(image=image_hash,prompt=DISCOVER_COMPACT,model=self.settings.model))
            compatible=fingerprint(dict(image=image_hash,prompt=DISCOVER,model=self.settings.model))
            record_path=folder/f'{image_id}.json'
            record=json.loads(record_path.read_text()) if record_path.exists() else None
            if record and record.get('identity') in {identity,compatible} and (record['status']=='complete' or not self.retry_failed):
                result.append(record);print(f'[{index}/{len(files)}] {image_id}: cached {record["status"]}',flush=True);continue
            try:
                image=cv2.imread(str(path))
                if image is None:raise ValueError('Cannot decode image')
                rgb=image[:,:,::-1]
                cached_response=(self.requests/f'{self.request_key(DISCOVER_COMPACT,[rgb])}.json').exists()
                try:rows=validate_observations(self.request(DISCOVER_COMPACT,[rgb]))
                except (ValueError,VisionResponseError):
                    if not (cached_response and record and record['status']=='failed' and self.retry_failed):raise
                    rows=validate_observations(self.request(DISCOVER_COMPACT,[rgb],force=True))
                observations=[dict(id=f'{image_id}:o{i:03}',image_id=image_id,**row) for i,row in enumerate(rows,1)]
                record=dict(image_id=image_id,source_image=str(path),source_sha256=image_hash,identity=identity,status='complete',
                    width=image.shape[1],height=image.shape[0],observations=observations,preview=f'previews/{image_id}.jpg')
                scale=min(1,640/image.shape[1]);cv2.imwrite(str(previews/f'{image_id}.jpg'),cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale))))
            except (ValueError,RuntimeError,OSError) as exc:
                record=dict(image_id=image_id,source_image=str(path),source_sha256=image_hash,identity=identity,status='failed',error=str(exc),observations=[])
                atomic_json(record_path,record);result.append(record)
                print(f'[{index}/{len(files)}] {image_id}: FAILED {exc}',flush=True)
                if re.search(r'HTTP (401|402|403)',str(exc)):raise RuntimeError('Provider access/credit error; stopped further requests') from None
                continue
            atomic_json(record_path,record);result.append(record)
            print(f'[{index}/{len(files)}] {image_id}: {len(record["observations"])} observations ({time.perf_counter()-started:.1f}s)',flush=True)
        return result

    def consolidate(self,observations):
        groups=group_observations(observations)
        if not groups:return [],{}
        assignments={};group_inputs={}
        for g in groups:
            group_inputs[g['group_id']]={k:v for k,v in g.items() if k!='observations'}|dict(examples=[dict(description=o['description'],evidence=o['visible_evidence'],uncertainty=o['uncertainty']) for o in g['observations'][:3]])
        # Seed per-type cache from already validated whole-request artifacts.
        # Adding an image can change batch boundaries without changing a type's
        # evidence. Its classification need not be billed a second time.
        cache=self.output/'type_assignments';cache.mkdir(exist_ok=True)
        model=getattr(getattr(self,'settings',None),'model','test-model')
        def cache_key(item):return fingerprint(dict(model=model,prompt=CLASSIFY,input=item))
        request_folder=getattr(self,'requests',None)
        if request_folder is not None:
            for path in sorted(request_folder.glob('*.json'),key=lambda p:p.stat().st_mtime):
                saved=json.loads(path.read_text())
                prefix=CLASSIFY_GROUPED if saved.get('prompt','').startswith(CLASSIFY_GROUPED) else CLASSIFY if saved.get('prompt','').startswith(CLASSIFY) else None
                if saved.get('model')!=model or prefix is None or saved.get('truncated'):continue
                try:
                    old_inputs=json.loads(saved['prompt'][len(prefix):]);raw=parse_catalog_response(saved['response'])
                    lookup=expand_type_assignments(raw,old_inputs,False)
                    for item in old_inputs:
                        item={k:v for k,v in item.items() if k!='index'}
                        if item['group_id'] not in lookup:continue
                        row=lookup[item['group_id']]
                        short(row.get('parent'),'parent',100);subtype_name(row.get('subtype'));short(row.get('description'),'description')
                        atomic_json(cache/f'{cache_key(item)}.json',dict(assignment=row,source_request=path.name))
                except (ValueError,KeyError,TypeError):continue
        missing=[]
        for g in groups:
            path=cache/f'{cache_key(group_inputs[g["group_id"]])}.json'
            if path.exists():assignments[g['group_id']]=json.loads(path.read_text())['assignment']
            else:missing.append(g)
        print(f'Type classifications cached: {len(assignments)}/{len(groups)}',flush=True)
        def classify(prompt,inputs):
            problems=[];expanded=expand_type_assignments(self.request(prompt),inputs,False,problems)
            if problems:
                directory=self.output/'merge_issues';directory.mkdir(exist_ok=True)
                atomic_json(directory/f'{fingerprint(prompt)}.json',dict(issues=problems,input_group_ids=[i['group_id'] for i in inputs],
                    withheld_group_ids=[i['group_id'] for i in inputs if i['group_id'] not in expanded]))
                print(f'Quarantined {len(problems)} duplicate/unsupported membership statements',flush=True)
            return expanded
        missing.sort(key=lambda g:(norm(g['parent_category']),norm(g['category']),g['group_id']))
        start=0
        while start<len(missing):
            count=min(64,len(missing)-start)
            while True:
                batch=missing[start:start+count];inputs=[group_inputs[g['group_id']] for g in batch]
                wire=[dict(index=i,**item) for i,item in enumerate(inputs)]
                prompt=CLASSIFY_GROUPED+json.dumps(wire)
                if len(prompt)<=55000:break
                if count==1:raise ValueError('One type evidence group exceeds the bounded request size')
                count=max(1,count//2)
            print(f'Consolidating new type groups {start+1}–{start+count} / {len(missing)}',flush=True)
            expanded=classify(prompt,inputs);assignments.update(expanded)
            for gid in expanded:atomic_json(cache/f'{cache_key(group_inputs[gid])}.json',dict(assignment=expanded[gid]))
            remaining=[item for item in inputs if item['group_id'] not in assignments]
            for attempt in range(2):
                if not remaining:break
                print(f'Classifying {len(remaining)} omitted type references (follow-up {attempt+1})',flush=True)
                followup=CLASSIFY_GROUPED+json.dumps([dict(index=i,**item) for i,item in enumerate(remaining)])
                repaired=classify(followup,remaining);assignments.update(repaired)
                for gid in repaired:atomic_json(cache/f'{cache_key(group_inputs[gid])}.json',dict(assignment=repaired[gid]))
                remaining=[item for item in remaining if item['group_id'] not in assignments]
            if remaining:raise ValueError('Consolidation still omitted type references; valid assignments retained for resume')
            start+=count
        # Apply the same schema/uncertainty rules to live and cached assignments.
        for g in groups:
            a=assignments[g['group_id']];a['parent']=short(a.get('parent'),'parent',100);a['subtype']=subtype_name(a.get('subtype'));a['description']=short(a.get('description'),'description')
            if a['subtype'] and norm(a['subtype'])==norm(a['parent']):a['subtype']=None
            if all(o['uncertainty']=='high' for o in g['observations']):
                a.update(parent=g['parent_category'],subtype=None,classification_uncertainty='high',classification_note='High source uncertainty; supplied broad family retained')
            if a.get('classification_note'):
                for observation in g['observations']:
                    observation['classification_note']=a['classification_note'];observation['classification_uncertainty']='high'
        parents=sorted({a['parent'] for a in assignments.values()})
        parent_inputs=[dict(id=f'p{i:03}',name=name) for i,name in enumerate(parents)]
        parent_clusters=reconcile_parent_clusters(self.request(MERGE_PARENTS+json.dumps(parent_inputs)),parents)
        registry=StableIDs(self.output/'class_ids.json');classes=[];mapping={}
        by_id={g['group_id']:g for g in groups}
        for cluster in parent_clusters:
            parent=cluster['name'];parent_id=registry.assign(['parent|'+norm(a) for a in [parent]+cluster['members']])
            gids=[gid for gid,a in assignments.items() if a['parent'] in cluster['members']]
            parent_observations=[o for gid in gids for o in by_id[gid]['observations']]
            root=self.node(parent_id,parent,None,'parent',cluster['description'],cluster['members'],parent_observations)
            classes.append(root);subtypes={};broad=[]
            for gid in gids:
                subtype=assignments[gid]['subtype']
                if subtype is None:broad.append(gid);continue
                key='t_'+fingerprint(norm(subtype))[:16]
                subtypes.setdefault(key,dict(id=key,name=subtype,groups=[]))['groups'].append(gid)
            for gid in broad:
                for o in by_id[gid]['observations']:mapping[o['id']]=parent_id
            if not subtypes:continue
            inputs=[]
            for t in subtypes.values():
                evidence=[o for gid in t['groups'] for o in by_id[gid]['observations']]
                inputs.append(dict(id=t['id'],name=t['name'],features=sorted({f for o in evidence for f in o['distinguishing_features']}),examples=[o['description'] for o in evidence[:3]]))
            print('Reconciling subtypes:',parent,len(inputs),flush=True)
            # Even one candidate can be appearance-only and need mapping to its parent.
            merged=validate_partition(self.request(MERGE_TYPES+json.dumps(dict(parent=parent,candidates=inputs))),list(subtypes),True)
            for c in merged:
                subtype=c['name']
                ids=[gid for key in c['members'] for gid in subtypes[key]['groups']]
                obs=[o for gid in ids for o in by_id[gid]['observations']]
                if subtype is None or norm(subtype)==norm(parent):identifier=parent_id
                else:
                    aliases=[subtypes[key]['name'] for key in c['members']]
                    identifier=registry.assign([f'child|{parent_id}|'+norm(a) for a in [subtype]+aliases])
                    classes.append(self.node(identifier,subtype,parent_id,'subtype',c['description'],aliases,obs))
                for o in obs:mapping[o['id']]=identifier
        registry.save()
        for c in classes:c['direct_observation_ids']=sorted(oid for oid,cid in mapping.items() if cid==c['id'])
        return sorted(classes,key=lambda c:c['id']),mapping

    @staticmethod
    def node(identifier,name,parent_id,kind,description,aliases,observations):
        attributes={k:sorted({v for o in observations for v in o['attributes'][k]}) for k in ['color','size','shape']}
        return dict(id=identifier,name=name,parent_id=parent_id,kind=kind,description=description,
            aliases=sorted(set(aliases)),attributes=attributes,distinguishing_features=sorted({v for o in observations for v in o['distinguishing_features']}),
            observation_ids=sorted(o['id'] for o in observations),image_ids=sorted({o['image_id'] for o in observations}),
            uncertainty_counts={level:sum(o.get('classification_uncertainty',o['uncertainty'])==level for o in observations) for level in ['low','medium','high']})

def make_prompts(images,classes):
    lookup={c['id']:c for c in classes};global_phrases={};per_image=[]
    for image in images:
        phrases={};class_ids=set()
        for o in image['observations']:
            node=lookup[o['class_id']];class_ids.add(node['id']);canonical=node['name'].lower()
            supplied=o['detection_phrase'].lower().rstrip('. ')
            # 'Valve actuator' must not target the valve class merely because
            # the phrase contains the word valve.
            head=norm(supplied.split(' with ',1)[0]).split();noun=norm(canonical).split()
            same_head=bool(noun) and head[-len(noun):]==noun
            specific=supplied if same_head and o['uncertainty']!='high' else ' '.join(o['attributes']['color'][:1]+[canonical])
            candidates=[(specific,node['id'],'specific'),(canonical,node['id'],'class')]
            if node['parent_id']:
                parent=lookup[node['parent_id']];candidates.append((parent['name'].lower(),parent['id'],'fallback'))
            for text,cid,role in candidates:
                text=' '.join(text.rstrip('. ').split());key=(norm(text),cid)
                entry=phrases.setdefault(key,dict(text=text,class_id=cid,role=role,roles=[],observation_ids=[]))
                if role not in entry['roles']:entry['roles'].append(role)
                if o['id'] not in entry['observation_ids']:entry['observation_ids'].append(o['id'])
        entries=list(phrases.values());per_image.append(dict(image_id=image['image_id'],status=image['status'],class_ids=sorted(class_ids),phrases=entries))
        for e in entries:
            key=(norm(e['text']),e['class_id']);entry=global_phrases.setdefault(key,dict(text=e['text'],class_id=e['class_id'],role=e['role'],roles=[],observation_ids=[]))
            entry['roles']=sorted(set(entry['roles']+e['roles']))
            entry['observation_ids']=sorted(set(entry['observation_ids']+e['observation_ids']))
    # One string referring to different classes is explicitly ambiguous, never
    # silently turned into multiple equivalent categories in a flat prompt.
    targets=defaultdict(set)
    for e in global_phrases.values():targets[norm(e['text'])].add(e['class_id'])
    def annotate(entries):
        for e in entries:e['ambiguous']=len(targets[norm(e['text'])])>1
        return entries
    def batches(entries):
        safe=[e for e in entries if not e['ambiguous']]
        return [dict(prompt_text='. '.join(e['text'] for e in safe[i:i+8])+'.',
                     targets=[dict(text=e['text'],class_id=e['class_id'],roles=e['roles']) for e in safe[i:i+8]]) for i in range(0,len(safe),8)]
    global_entries=annotate(list(global_phrases.values()))
    for image in per_image:
        annotate(image['phrases']);image['prompt_text']='. '.join(dict.fromkeys(e['text'] for e in image['phrases'] if not e['ambiguous']))+'.' if image['phrases'] else ''
        image['batches']=batches(image['phrases'])
    return dict(schema_version=1,meaning='Text phrases target category IDs, not physical instance IDs. Broad fallback phrases target the parent class.',
        global_phrases=global_entries,global_prompt_text='. '.join(dict.fromkeys(e['text'] for e in global_entries if not e['ambiguous']))+'.' if global_entries else '',
        global_batches=batches(global_entries),per_image=per_image,classes=[dict(class_id=c['id'],name=c['name'],fallback_class_ids=[c['parent_id']] if c['parent_id'] else [],
            phrases=[e['text'] for e in global_entries if e['class_id']==c['id'] and not e['ambiguous']]) for c in classes])

def validate_outputs(per_image,catalog,prompts,expected_images):
    images=per_image['images'];classes=catalog['classes'];lookup={c['id']:c for c in classes}
    if len(lookup)!=len(classes):raise ValueError('Duplicate class IDs')
    if {i['image_id'] for i in images}!=set(expected_images) or len(images)!=len(expected_images):raise ValueError('Missing/duplicate image records')
    observations={o['id']:o for i in images for o in i['observations']}
    if len(observations)!=sum(len(i['observations']) for i in images):raise ValueError('Duplicate observation IDs')
    for oid,o in observations.items():
        if o['class_id'] not in lookup or oid not in lookup[o['class_id']]['direct_observation_ids']:raise ValueError('Missing observation-to-class mapping')
    for image in images:
        if image['status'] not in {'complete','failed'} or any(o['image_id']!=image['image_id'] for o in image['observations']):raise ValueError('Invalid image status or observation ownership')
    direct=[]
    for c in classes:
        if norm(c['name']) in {'null','none','n a','unspecified','generic'}:raise ValueError('Placeholder leaked into canonical class name')
        parent=c['parent_id']
        if parent is not None and (parent not in lookup or lookup[parent]['parent_id'] is not None or parent==c['id']):raise ValueError('Invalid class parent/cycle')
        if not c['observation_ids'] or not set(c['observation_ids'])<=observations.keys():raise ValueError('Unsupported class or source observation')
        for oid in c['observation_ids']:
            assigned=observations[oid]['class_id']
            if assigned!=c['id'] and lookup[assigned]['parent_id']!=c['id']:raise ValueError('Class support does not match ancestry')
        direct.extend(c['direct_observation_ids'])
        if set(c['image_ids'])!={observations[o]['image_id'] for o in c['observation_ids']}:raise ValueError('Class image provenance mismatch')
    if sorted(direct)!=sorted(observations):raise ValueError('Incomplete or duplicate direct mapping')
    if {i['image_id'] for i in prompts['per_image']}!=set(expected_images):raise ValueError('Missing image prompts')
    for image in prompts['per_image']:
        for entry in image['phrases']:
            if entry['class_id'] not in lookup:raise ValueError('Unknown per-image phrase class')
            for oid in entry['observation_ids']:
                if oid not in observations or observations[oid]['image_id']!=image['image_id'] or oid not in lookup[entry['class_id']]['observation_ids']:raise ValueError('Per-image phrase uses evidence from another image or class')
    for entry in prompts['global_phrases']:
        if entry['class_id'] not in lookup or not entry['observation_ids']:raise ValueError('Unsupported prompt')
        for oid in entry['observation_ids']:
            if oid not in observations or oid not in lookup[entry['class_id']]['observation_ids']:raise ValueError('Prompt has unsupported evidence')
    return dict(state='passed',images=len(images),complete_images=sum(i['status']=='complete' for i in images),failed_images=sum(i['status']=='failed' for i in images),
        observations=len(observations),classes=len(classes),all_observations_mapped=True,all_classes_evidence_linked=True,parents_valid=True,
        limitation='Structural/provenance checks do not establish semantic accuracy or recall.')

def write_report(output,per_image,catalog,prompts,audit):
    images={i['image_id']:i for i in per_image['images']};nodes={c['id']:c for c in catalog['classes']}
    cards=[]
    for c in catalog['classes']:
        examples=''.join(f'<a href="#image-{html.escape(name,quote=True)}"><img loading="lazy" src="{quote(images[name].get("preview",""))}" alt="{html.escape(name,quote=True)}"></a>' for name in c['image_ids'][:3])
        parent=nodes[c['parent_id']]['name'] if c['parent_id'] else 'broad parent / fallback'
        phrases=next(p['phrases'] for p in prompts['classes'] if p['class_id']==c['id'])
        cards.append(f'<article><h3>{html.escape(c["name"])} <small>{c["id"]}</small></h3><p>Parent: {html.escape(parent)}</p><p>{html.escape(c["description"])}</p><p>{len(c["image_ids"])} supporting previews · {len(c["observation_ids"])} observations (not instance counts)</p><p>Detection phrases: {html.escape(" · ".join(phrases[:8]))}</p>'+examples+'</article>')
    views=[]
    for image in per_image['images']:
        rows=''.join(f'<tr><td>{o["id"]}</td><td>{html.escape(nodes[o["class_id"]]["name"])}</td><td>{html.escape(o["description"])} {html.escape(o.get("classification_note",""))}</td><td>{o.get("classification_uncertainty",o["uncertainty"])}</td><td>{html.escape(o["visible_evidence"])}</td></tr>' for o in image['observations'])
        content=f'<img loading="lazy" src="{quote(image["preview"])}">' if image.get('preview') else ''
        views.append(f'<details id="image-{html.escape(image["image_id"],quote=True)}"><summary>{html.escape(image["image_id"])} — {image["status"]}, {len(image["observations"])} observations</summary>'+content+(f'<p>{html.escape(image["error"])}</p>' if image.get('error') else '')+'<table><tr><th>Observation</th><th>Unified class</th><th>Appearance</th><th>Uncertainty</th><th>Visible evidence</th></tr>'+rows+'</table></details>')
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Equipment room vocabulary</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;background:#10151b;color:#e4eaf2}a{color:#8cd3ff}article,details{padding:16px;margin:14px 0;background:#1c2631;border-radius:9px}article img{width:30%;margin:1%}details img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #46505a}small{font-size:12px;color:#aabbcc}input{padding:10px;width:90%;margin:15px 0}</style><h1>Equipment room class vocabulary</h1>'+f'<p>{audit["complete_images"]}/{audit["images"]} previews processed · {audit["observations"]} observations · {audit["classes"]} hierarchy nodes · {audit["failed_images"]} failed images.</p><p>These are model-proposed object TYPES, not unique physical instances. Names and evidence are not expert verified. Colour and missing detail do not define new classes. Broad fallbacks remain linked to parent IDs.</p><p><a href="class_catalog.json">Unified catalog</a> · <a href="per_image.json">Per-image observations</a> · <a href="grounding_prompts.json">Detection phrases</a> · <a href="audit.json">Validation</a></p><input id="filter" placeholder="Filter classes by name or description"><section id="classes">'+''.join(cards)+'</section><h2>All previews and observations</h2>'+''.join(views)+'<script>document.getElementById("filter").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll("article").forEach(c=>c.hidden=!c.textContent.toLowerCase().includes(q));});document.querySelectorAll("a[href^=\"#image-\"]").forEach(a=>a.addEventListener("click",()=>{document.getElementById(a.hash.slice(1)).open=true;}));</script>')

def run_catalog(images,output,env,pilot=False,retry_failed=False):
    output=output.resolve();started=time.perf_counter()
    with run_lock(output):
        try:
            job=Catalog(images,output,env,retry_failed)
            atomic_json(output/'model_check.json',job.client.check_model())
            files=job.files
            if pilot:
                indices=sorted({round(i*(len(files)-1)/4) for i in range(5)});files=[files[i] for i in indices]
            atomic_json(output/'manifest.json',dict(images_directory=str(job.images),model=job.settings.model,pilot=pilot,
                images=[dict(image_id=p.name,sha256=digest_file(p)) for p in files],discovery_prompt_sha256=fingerprint(DISCOVER_COMPACT),compatible_verbose_prompt_sha256=fingerprint(DISCOVER),vision_generation=VISION_GENERATION))
            records=job.extract(files);observations=[o for record in records for o in record['observations']]
            classes,mapping=job.consolidate(observations)
            for record in records:
                for observation in record['observations']:observation['class_id']=mapping[observation['id']]
            per_image=dict(schema_version=1,scene=job.images.parent.name,model=job.settings.model,images=records)
            catalog=dict(schema_version=1,scene=job.images.parent.name,meaning='Observed category vocabulary, not physical instance inventory',
                classes=classes,id_redirects=json.loads((output/'class_ids.json').read_text()).get('redirects',{}) if (output/'class_ids.json').exists() else {})
            prompts=make_prompts(records,classes);audit=validate_outputs(per_image,catalog,prompts,[p.name for p in files])
            for name,data in [('per_image',per_image),('class_catalog',catalog),('grounding_prompts',prompts),('audit',audit)]:atomic_json(output/f'{name}.json',data)
            write_report(output,per_image,catalog,prompts,audit)
            result=dict(state='partial' if audit['failed_images'] else 'complete',pilot=pilot,model=job.settings.model,
                **{k:audit[k] for k in ['images','complete_images','failed_images','observations','classes']},
                new_completion_requests=job.client.calls,new_usage=job.new_usage,seconds=time.perf_counter()-started)
            atomic_json(output/('pilot_result.json' if pilot else 'run.json'),result)
            if (output/'failure.json').exists():(output/'failure.json').unlink()
            print(json.dumps(result,indent=2),flush=True);return result
        except BaseException as exc:
            atomic_json(output/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
