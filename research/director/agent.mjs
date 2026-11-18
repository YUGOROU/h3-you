// Pi SDK harness: no resource discovery, builtin tools, MCP, sessions on disk or device adapters.
import {pathToFileURL} from 'node:url';
import {mkdtemp,rm} from 'node:fs/promises';
import {tmpdir,homedir} from 'node:os';
import {join,resolve} from 'node:path';
import readline from 'node:readline';
import {MODEL,PROVIDER,MAX_CALLS,MAX_OUTPUT,SYSTEM,VISUAL_SYSTEM,STABLE_SELFIE_SYSTEM,BASELINE,VISUAL_BASELINE,makeTools} from './contract.mjs';

const visualIntents=process.env.H3_DIRECTOR_VISUAL_INTENTS==='1';
const stableSelfie=process.env.H3_DIRECTOR_STABLE_SELFIE==='1';
if ((process.env.H3_DIRECTOR_STABLE_SELFIE!==undefined&&!stableSelfie)||(stableSelfie&&!visualIntents)) throw Error('invalid stable selfie selection');
if(process.env.H3_DIRECTOR_VISUAL_INTENTS!==undefined&&!visualIntents)throw Error('invalid visual contract selection');
const emit=data=>process.stdout.write(JSON.stringify(data)+'\n');
const longSession=process.env.H3_DIRECTOR_LONG_SESSION==='1';
const maxTurns=Number(process.env.H3_DIRECTOR_MAX_TURNS||1);
if (!Number.isInteger(maxTurns)||maxTurns<1||maxTurns>(longSession?1000:8)) throw Error('invalid finite turn limit');
const piRoot=process.env.H3_PI_PACKAGE_ROOT||join(homedir(),'.local/lib/node_modules/@earendil-works/pi-coding-agent');
const sdk=await import(pathToFileURL(join(piRoot,'dist/index.js')).href);
const {Type}=await import(pathToFileURL(join(piRoot,'node_modules/@earendil-works/pi-ai/dist/index.js')).href);
const temporary=await mkdtemp(join(tmpdir(),'h3-director-'));
const usage={calls:0,unknown_calls:0,input:0,output:0};
let active=null,epoch=0,turns=0,history=[],closed=false;
const runtime=await sdk.ModelRuntime.create({authPath:join(temporary,'auth.json'),modelsPath:resolve(import.meta.dirname,'models.json'),modelsStorePath:join(temporary,'models-cache.json'),allowModelNetwork:false,refreshOnCreate:false});
const model=runtime.getModel(PROVIDER,MODEL);
if (!model||model.baseUrl!=='https://router.huggingface.co/v1') throw Error('exact HF model unavailable');
const original=runtime.streamSimple.bind(runtime);
runtime.streamSimple=(selected,context,options={})=>{
  if (!active || active.canceled || active.calls>=MAX_CALLS) throw Error('finite model call limit reached');
  if (selected.id!==MODEL||selected.provider!==PROVIDER) throw Error('model fallback forbidden');
  active.calls++;usage.calls++;usage.unknown_calls++;
  const requestState=active;
  const prior=options.onPayload;
  const providerFetch=options.fetch??globalThis.fetch;
  const observedFetch=async(...args)=>{
    // Observe only structural transport outcomes. Do not inspect URL, headers,
    // body, error text or credentials; preserve the exact response/exception.
    try{
      const response=await providerFetch(...args);
      if(Number.isInteger(response?.status)&&response.status>=400&&response.status<=599)
        requestState.providerHttpStatus=response.status;
      return response;
    }catch(error){requestState.providerNetworkError=true;throw error;}
  };
  return original(selected,context,{...options,fetch:observedFetch,maxTokens:MAX_OUTPUT,onPayload:async(payload,...args)=>{
    const body=prior?(await prior(payload,...args))??payload:payload;
    if (body.reasoning_effort!=='none'||body.model!==MODEL || !Number.isInteger(body.max_tokens??body.max_completion_tokens) || (body.max_tokens??body.max_completion_tokens)>MAX_OUTPUT) throw Error('request model/output limit differs');
    return body;
  }});
};

// Only structural flags/status are exported: never exception/message/tool text.
function turnFailureDiagnostic(state,error){
  const messages=state.session?.messages??[];
  const status=state.providerHttpStatus??error?.status;
  const httpStatus=Number.isInteger(status)&&status>=400&&status<=599 ? status : null;
  const code=state.canceled?'canceled':state.failureCode??
    (messages.some(m=>m.role==='toolResult'&&m.isError)?'tool_error':
     httpStatus!==null?'provider_http_error':
     state.providerNetworkError?'provider_network_error':
     messages.some(m=>m.role==='assistant'&&m.stopReason==='error')?'assistant_error':
     messages.some(m=>m.role==='assistant'&&m.stopReason==='aborted')?'aborted':
     !error&&!state.draft?'missing_proposal':'runtime_error');
  return {schema:'h3-director-failure-v1',code,http_status:httpStatus};
}

function loader(){return {
  getExtensions:()=>({extensions:[],errors:[],runtime:sdk.createExtensionRuntime()}),
  getSkills:()=>({skills:[],diagnostics:[]}),getPrompts:()=>({prompts:[],diagnostics:[]}),
  getThemes:()=>({themes:[],diagnostics:[]}),getAgentsFiles:()=>({agentsFiles:[]}),
  getSystemPrompt:()=>stableSelfie?STABLE_SELFIE_SYSTEM:visualIntents?VISUAL_SYSTEM:SYSTEM,getSystemPromptSource:()=>undefined,
  getAppendSystemPrompt:()=>[],getAppendSystemPromptSources:()=>[],extendResources:()=>{},reload:async()=>{},
};}

async function turn(packet){
  if (active) {emit({type:'error',code:'busy'});return;}
  if (turns>=maxTurns) {emit({type:'error',code:'turn_limit'});return;}
  if (typeof packet.text!=='string'||!packet.text.trim()||Buffer.byteLength(packet.text)>2000||typeof packet.turn_id!=='string'||!/^[a-zA-Z0-9_-]{1,64}$/.test(packet.turn_id)) {emit({type:'error',code:'invalid_turn'});return;}
  turns++;const state={id:packet.turn_id,epoch:++epoch,calls:0,canceled:false,draft:null,session:null};active=state;
  let timer;
  try{
    const tools=makeTools(Type,state,{visualIntents,stableSelfie});
    const {session}=await sdk.createAgentSession({cwd:temporary,agentDir:temporary,modelRuntime:runtime,model,thinkingLevel:'off',
      resourceLoader:loader(),noTools:'builtin',tools:tools.map(t=>t.name),customTools:tools,
      sessionManager:sdk.SessionManager.inMemory(temporary),
      settingsManager:sdk.SettingsManager.inMemory({compaction:{enabled:false},retry:{enabled:false,provider:{maxRetries:0,timeoutMs:45000}},cacheWarming:'off',enableInstallTelemetry:false,enableAnalytics:false,defaultTools:[]})});
    state.session=session;
    if (session.getActiveToolNames().sort().join(',')!=='h3_get_state,h3_submit_turn') {
      state.failureCode='contract_error';throw Error('unexpected active tools');
    }
    session.subscribe(event=>{
      if (event.type!=='message_end'||event.message.role!=='assistant')return;
      const u=event.message.usage;
      if (u && Number.isFinite(u.input)&&Number.isFinite(u.output) && u.input+u.output+(u.cacheRead||0)+(u.cacheWrite||0)>0){
        usage.input+=u.input+(u.cacheRead||0)+(u.cacheWrite||0);usage.output+=u.output;usage.unknown_calls--;
      }
    });
    timer=setTimeout(()=>{state.canceled=true;session.abort();},60000);
    await session.prompt(JSON.stringify(visualIntents ? {prior_proposals:history,user_text:packet.text,visual_achievement:'unverified'} : {successful_turns:history,user_text:packet.text}));
    const failed=session.messages.some(m=>(m.role==='assistant'&&['error','aborted'].includes(m.stopReason))||(m.role==='toolResult'&&m.isError));
    if (state.canceled||failed||!state.draft){
      const diagnostic=turnFailureDiagnostic(state,null);
      emit({type:'error',turn_id:state.id,code:state.canceled?'canceled':'director_failed',diagnostic});
      return;
    }
    history.push({user:packet.text,reply:state.draft.reply,...(visualIntents?{visual_intent:state.draft.visual_intent,visual_achievement:'unverified'}:{})});history=history.slice(-8);
    emit({type:'turn',turn_id:state.id,epoch:state.epoch,model:MODEL,baseline:visualIntents?VISUAL_BASELINE:BASELINE,...state.draft,
      status:'proposal_only',effects_executed:false,renderer_attached:false,audio_attached:false});
  }catch(error){emit({type:'error',turn_id:state.id,code:state.canceled?'canceled':'director_failed',
    diagnostic:turnFailureDiagnostic(state,error)});}
  finally{clearTimeout(timer);state.session?.dispose();active=null;if(closed)finish();}
}
let finished=false;
async function finish(){if(finished||active)return;finished=true;emit({type:'usage',...usage,turns});await rm(temporary,{recursive:true,force:true});}
const input=readline.createInterface({input:process.stdin,terminal:false});
input.on('line',line=>{
  try{
    if(Buffer.byteLength(line)>4096)throw Error('large packet');const packet=JSON.parse(line);
    if(packet.op==='cancel'){
      epoch++;if(active){active.canceled=true;active.session?.abort();}
      emit({type:'canceled',epoch});
    }else if(packet.op==='turn')void turn(packet);
    else emit({type:'error',code:'unsupported_operation'});
  }catch{emit({type:'error',code:'invalid_packet'});}
});
input.on('close',()=>{closed=true;if(!active)void finish();});
const lifetime=setTimeout(()=>{closed=true;input.close();if(active){active.canceled=true;active.session?.abort();}else void finish();},longSession?10800000:1200000);lifetime.unref();
for(const signal of ['SIGTERM','SIGINT'])process.on(signal,()=>{closed=true;input.close();if(active){active.canceled=true;active.session?.abort();}else void finish();});
emit({type:'ready',model:MODEL,tools:['h3_get_state','h3_submit_turn'],sensors:false,persistent_session:false,...(visualIntents?{visual_intents:true}:{}),...(stableSelfie?{stable_selfie:true}:{})});
