"""Offline finalization using saved visual evidence and type classifications."""
from pathlib import Path
from collections import defaultdict
import json,re,time
from .catalog import (Catalog,StableIDs,CLASSIFY,norm,group_observations,make_prompts,
                      validate_outputs,write_report,protected_family)
from .io import atomic_json,fingerprint

def finish_saved_catalog(output):
    started=time.perf_counter();output=Path(output).resolve()
    manifest=json.loads((output/'manifest.json').read_text());model=manifest['model']
    records=[json.loads((output/'images'/f'{i["image_id"]}.json').read_text()) for i in manifest['images']]
    if any(r['status']!='complete' for r in records):raise ValueError('Every preview must be complete before offline finalization')
    observations=[o for r in records for o in r['observations']]
    groups=group_observations(observations);families=defaultdict(list);forms=defaultdict(list);aliases=defaultdict(set);used=0
    plural={'gauges':'gauge','tanks':'tank','chairs':'chair','ladders':'ladder','containers':'container','doors':'door','tools':'tool','fans':'fan','boxes':'box','appliances':'appliance','cabinets':'cabinet','pumps':'pump','valves':'valve','pipes':'pipe','actuators':'actuator'}
    synonyms={'electrical enclosure':'cabinet','electrical cabinet':'cabinet','control cabinet':'cabinet','pressure gauge':'gauge','temperature gauge':'gauge','wire':'cable','wires':'cable','electrical wire':'cable','cables':'cable','trolley':'cart','dolly':'hand truck'}
    def parent_name(value):
        value=norm(value);value=plural.get(value,value);value=synonyms.get(value,value)
        return protected_family(value) or value or 'unknown component'
    def subtype(parent,text):
        text=norm(text or '')
        rules={
          'valve':[('handwheel','handwheel valve'),('hand wheel','handwheel valve'),('lever','lever-operated valve'),('actuated','actuated valve'),('ball valve','ball valve'),('butterfly','butterfly valve')],
          'chair':[('folding','folding chair'),('office','office chair'),('swivel','office chair'),('star base','office chair')],
          'ladder':[('step','step ladder'),('a frame','step ladder'),('extension','extension ladder')],
          'pipe fitting':[('elbow','elbow fitting'),('tee','tee fitting'),('coupling','pipe coupling'),('flange','pipe flange')],
          'pump':[('circulation','circulation pump')],
          'cart':[('shelf','utility cart'),('utility','utility cart'),('platform','platform cart')]}
        return next((name for word,name in rules.get(parent,[]) if word in text),None)
    for g in groups:
        item={k:v for k,v in g.items() if k!='observations'}|dict(examples=[dict(description=o['description'],evidence=o['visible_evidence'],uncertainty=o['uncertainty']) for o in g['observations'][:3]])
        key=fingerprint(dict(model=model,prompt=CLASSIFY,input=item));path=output/'type_assignments'/f'{key}.json'
        a=json.loads(path.read_text())['assignment'] if path.exists() else None
        if a:used+=1
        parent=parent_name(a['parent'] if a else g['parent_category'])
        uncertain=not a or a.get('classification_uncertainty')=='high' or all(o['uncertainty']=='high' for o in g['observations'])
        child=None if uncertain else subtype(parent,a.get('subtype'))
        aliases[parent].add(a['parent'] if a else g['parent_category'])
        families[parent].extend(g['observations'])
        for o in g['observations']:
            if uncertain:
                o['classification_uncertainty']='high';o['classification_note']='Broad family retained; detailed distinction uncertain or unfinished'
            o['_final_type']=(parent,child)
        if child:forms[(parent,child)].extend(g['observations'])
    registry=StableIDs(output/'class_ids.json');nodes=[];ids={}
    for parent,obs in sorted(families.items()):
        pid=registry.assign(['parent|'+norm(n) for n in [parent]+list(aliases[parent])]);ids[(parent,None)]=pid
        nodes.append(Catalog.node(pid,parent,None,'parent',f'Observed {parent} category',sorted(aliases[parent]),obs))
    for (parent,child),obs in sorted(forms.items()):
        pid=ids[(parent,None)];cid=registry.assign([f'child|{pid}|{norm(child)}']);ids[(parent,child)]=cid
        nodes.append(Catalog.node(cid,child,pid,'subtype',f'Observed {child} form',[child],obs))
    for o in observations:o['class_id']=ids[o.pop('_final_type')]
    for c in nodes:c['direct_observation_ids']=sorted(o['id'] for o in observations if o['class_id']==c['id'])
    registry.save();nodes.sort(key=lambda n:n['id'])
    per=dict(schema_version=1,scene=Path(manifest['images_directory']).parent.name,model=model,images=records)
    redirects={}
    active={c['id'] for c in nodes}
    for old,target in registry.data['redirects'].items():
        seen={old}
        while target in registry.data['redirects'] and target not in seen:seen.add(target);target=registry.data['redirects'][target]
        if target in active and old not in active:redirects[old]=target
    catalog=dict(schema_version=1,scene=per['scene'],meaning='Object types, not unique physical instances; model hypotheses',
        finalization='Fast offline synonym normalization and conservative structural subtypes; no further model requests',classes=nodes,id_redirects=redirects)
    prompts=make_prompts(records,nodes);audit=validate_outputs(per,catalog,prompts,[i['image_id'] for i in manifest['images']])
    for name,data in [('per_image',per),('class_catalog',catalog),('grounding_prompts',prompts),('audit',audit)]:atomic_json(output/f'{name}.json',data)
    write_report(output,per,catalog,prompts,audit)
    result=dict(state='complete',mode='fast-offline-finalization',images=len(records),complete_images=len(records),failed_images=0,
        observations=len(observations),classes=len(nodes),parent_classes=len(families),subtypes=len(forms),
        cached_llm_type_groups_used=used,total_type_groups=len(groups),broad_fallback_groups=len(groups)-used,
        new_completion_requests=0,seconds=time.perf_counter()-started)
    atomic_json(output/'run.json',result)
    if (output/'failure.json').exists():(output/'failure.json').unlink()
    return result
