import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, mkdir, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {spawnSync} from 'node:child_process';

const secret='PRIVATE_PROMPT_TOKEN_RESPONSE_DO_NOT_LOG';
async function runCase(scenario,{cancel=false}={}){
  const root=await mkdtemp(join(tmpdir(),'h3-fake-director-'));
  try{
    const ai=join(root,'node_modules/@earendil-works/pi-ai/dist');
    await mkdir(ai,{recursive:true});await mkdir(join(root,'dist'));
    await writeFile(join(root,'package.json'),JSON.stringify({type:'module'}));
    await writeFile(join(root,'node_modules/@earendil-works/pi-ai/package.json'),JSON.stringify({type:'module'}));
    await writeFile(join(ai,'index.js'),`export const Type=new Proxy({},{get:()=>()=>({})});`);
    await writeFile(join(root,'dist/index.js'),`
export const ModelRuntime={create:async()=>({getModel:(provider,id)=>({provider,id,baseUrl:'https://router.huggingface.co/v1'}),
 async streamSimple(model,context,options){
  if(process.env.H3_FAKE_CASE==='fetch-success'){
   if(options.maxRetries!==0||options.timeoutMs!==45000||options.signal?.marker!=='signal'||options.onResponse?.marker!=='callback')
    throw Error('caller options changed');
   const payload=await options.onPayload({model:model.id,reasoning_effort:'none',max_tokens:10});
   if(payload.callerMarker!=='unchanged')throw Error('caller payload hook changed');
  }
  const request={opaque:'${secret}'};
  const response=await options.fetch(request);
  if(response.marker!=='unchanged')throw Error('response changed');
  return response;
 }})};
export const SessionManager={inMemory:()=>({})};
export const SettingsManager={inMemory:x=>x};
export const createExtensionRuntime=()=>({});
export async function createAgentSession(options){
 const session={messages:[],getActiveToolNames:()=>process.env.H3_FAKE_CASE==='contract'?['forbidden']:['h3_get_state','h3_submit_turn'],subscribe(){},dispose(){},abort(){},
  async prompt(){
   const scenario=process.env.H3_FAKE_CASE;
   if(scenario==='http'){const e=Error('${secret}');e.status=429;throw e;}
   if(scenario==='runtime')throw Error('${secret}');
   if(['fetch-http','fetch-network','fetch-success','fetch-status600','fetch-status-bool','delayed-fetch-http'].includes(scenario)){
    if(scenario==='delayed-fetch-http')await new Promise(resolve=>setTimeout(resolve,25));
    const status=scenario==='fetch-success'?200:scenario==='fetch-status600'?600:scenario==='fetch-status-bool'?true:503;
    const response={status,marker:'unchanged'};
    Object.defineProperty(response,'headers',{get(){throw Error('private headers accessed');}});
    const callback=()=>{};callback.marker='callback';
    try{
     await options.modelRuntime.streamSimple(options.model,{}, {maxRetries:0,timeoutMs:45000,
      signal:{marker:'signal'},onResponse:callback,
      onPayload:async(payload)=>({...payload,callerMarker:'unchanged'}),fetch:async(request)=>{
      if(request.opaque!=='${secret}')throw Error('request changed');
      if(scenario==='fetch-network')throw Error('${secret}');
      return response;
     }});
    }catch(error){if(scenario!=='fetch-network')throw error;}
    this.messages=[{role:'assistant',stopReason:'error',errorMessage:'${secret}'}];
   }
   if(scenario==='assistant')this.messages=[{role:'assistant',stopReason:'error',errorMessage:'${secret}',content:[{text:'${secret}'}]}];
   if(scenario==='aborted')this.messages=[{role:'assistant',stopReason:'aborted',errorMessage:'${secret}'}];
   if(scenario==='tool'){
    try{await options.customTools.find(t=>t.name==='h3_submit_turn').execute('id',{reply:'${secret}',bad:true});}
    catch{this.messages=[{role:'toolResult',isError:true,content:[{text:'${secret}'}]}];}
   }
  }};
 return {session};
}
`);
    const input=JSON.stringify({op:'turn',turn_id:'fixture-turn',text:secret})+'\n'+(cancel?' {"op":"cancel"}\n':'');
    const result=spawnSync(process.execPath,[resolve('research/director/agent.mjs')],{
      input,encoding:'utf8',timeout:10000,env:{PATH:process.env.PATH,HOME:process.env.HOME,
        H3_PI_PACKAGE_ROOT:root,H3_FAKE_CASE:scenario,H3_DIRECTOR_MAX_TURNS:'1',HF_TOKEN:secret}});
    assert.equal(result.status,0,result.stderr);assert.equal(result.error,undefined);
    assert.ok(!result.stdout.includes(secret));assert.ok(!result.stderr.includes(secret));
    return result.stdout.trim().split('\n').map(JSON.parse).find(r=>r.type==='error');
  }finally{await rm(root,{recursive:true,force:true});}
}

for(const [scenario,code,status] of [
  ['http','provider_http_error',429],['assistant','assistant_error',null],
  ['tool','tool_error',null],['missing','missing_proposal',null],
  ['runtime','runtime_error',null],['contract','contract_error',null],['aborted','aborted',null],
  ['fetch-http','provider_http_error',503],['fetch-network','provider_network_error',null],
  ['fetch-success','assistant_error',null],['delayed-fetch-http','provider_http_error',503],
  ['fetch-status600','assistant_error',null],['fetch-status-bool','assistant_error',null],
])test(`sanitized ${scenario} classification without raw text`,async()=>{
  assert.deepEqual(await runCase(scenario),{type:'error',turn_id:'fixture-turn',code:'director_failed',
    diagnostic:{schema:'h3-director-failure-v1',code,http_status:status}});
});
test('explicit cancellation is distinct from missing proposal',async()=>{
  const row=await runCase('missing',{cancel:true});
  assert.equal(row.code,'canceled');
  assert.equal(row.diagnostic.code,'canceled');
  assert.equal(row.diagnostic.http_status,null);
});
