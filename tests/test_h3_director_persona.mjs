import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {PERSONA,PERSONA_SHA256} from '../research/director/persona.mjs';
import {SYSTEM,MODEL,MAX_OUTPUT,validateDraft,makeTools} from '../research/director/contract.mjs';
const read=path=>readFileSync(new URL('../'+path,import.meta.url),'utf8');
const Type={Object:(p,o)=>({p,o}),String:o=>({type:'string',...o}),Union:v=>({anyOf:v}),Literal:v=>({const:v}),Integer:()=>({type:'integer'}),Unknown:()=>({})};

test('exact approved curated body included once; no private source loading',()=>{
  const approved=read('docs/h3-persona-response-policy-draft-20261026.md').split('```text\n')[1].split('\n```')[0];
  assert.equal(PERSONA,approved);
  assert.equal(createHash('sha256').update(PERSONA).digest('hex'),PERSONA_SHA256);
  assert.equal(PERSONA_SHA256,'b42b6558922344a08155e296a6a74e1dfa889d7280aabd3b0d2ca026a1c4222a');
  assert.equal(SYSTEM.split(PERSONA).length,2);
  for(const file of ['research/director/persona.mjs','research/director/contract.mjs','research/director/agent.mjs','research/h3_personal/director_runtime.py']){
    assert.doesNotMatch(read(file),/chatgpt-personal\.txt|yugo_persona_voice_research|readFile.*persona/);
  }
});

test('actual relay Node path consumes SYSTEM without an opt-in flag',()=>{
  const relay=read('research/run_h3_relay_conversation.py');
  assert.match(relay,/class PiDirector/);
  assert.match(relay,/director\/agent\.mjs/);
  assert.match(relay,/child_environment\(token, max_turns/);
  const agent=read('research/director/agent.mjs');
  assert.match(agent,/SYSTEM.*from '\.\/contract\.mjs'/);
  // The prompt is SYSTEM unless the explicit visual/stable-selfie modes select their own.
  assert.match(agent,/getSystemPrompt:\(\)=>[^,]*\?[^,]*\?[^:]*:SYSTEM/);
  assert.match(agent,/resourceLoader:loader\(\)/);
  assert.match(agent,/noTools:'builtin'/);
  assert.match(agent,/getAgentsFiles:\(\)=>\(\{agentsFiles:\[\]\}\)/);
  assert.equal(MODEL,'Qwen/Qwen3.8-27B:cerebras');
  assert.equal(MAX_OUTPUT,1024);
  assert.match(agent,/body.reasoning_effort!=='none'/);
  const config=JSON.parse(read('research/director/models.json'));
  assert.equal(config.providers['h3-hf-cerebras'].models[0].samplingParams.reasoning_effort,'none');
});

test('conditional clarification, truthful confidence and non-human identity are explicit',()=>{
  for(const phrase of ['実在の本人ではない','会話相手全員をYugoと決めつけない','本人の口癖・一人称・語尾','低リスクで有力な解釈','候補が複数あり答えが大きく変わる','相手の訂正を優先','分からないことを確実と装うことではない','検索・個人記憶ツールは現在使えない','研究計画を達成済み実績にしない']) assert.ok(SYSTEM.includes(phrase),phrase);
  assert.doesNotMatch(SYSTEM,/By default, infer intended speech/);
});

// Authored policy fixtures, NOT model outputs or evidence of factual accuracy.
const fixtures=[
  ['concrete GAP','GAPはどこの国の企業？','衣料品ブランドのGAPなら、米国の企業です。'],
  ['low-risk kana','服のギャップはどこの国？','衣料品のGAPなら米国の企業です。'],
  ['capital','エチオピアの首都は？','エチオピアの首都はアディスアベバです。'],
  ['genuine ambiguity','あのエー社はどこの国？','どのエー社か、業種か正式名を教えてください。'],
  ['correction','服じゃなくて別会社の話','衣料品ではないのですね。会社の正式名を教えてください。'],
  ['ordinary chat','夕飯を簡単にしたい','卵とご飯があれば、卵チャーハンが手軽です。'],
  ['research','家庭向けロボットって？','家ごとに異なるキッチンでも、物をつかんだり調理を手伝ったりできるロボットを目指す研究です。'],
  ['current limits','GAP株の今の価格は？','現在の株価は検索できないため確認できません。証券アプリなどで最新値を確認してください。'],
  ['private memory','昨日の約束を覚えてる？','この会話では昨日の約束を確認できません。内容を教えてください。'],
  ['identity','本人なの？','本人ではなく、Yugoの背景や関心を共有するAI版です。'],
];
for(const [name,input,reply] of fixtures) test(`authored offline fixture: ${name}`,async()=>{
  assert.ok(input.length>0);
  const state={id:'fixture',epoch:1,canceled:false,draft:null};
  const tools=makeTools(Type,state);
  assert.deepEqual(tools.map(t=>t.name),['h3_get_state','h3_submit_turn']);
  const draft={reply,expression:'neutral',motion:'still',tone:'calm'};
  assert.deepEqual(validateDraft(draft),draft);
  const result=await tools[1].execute('fixture',draft);
  assert.equal(result.structuredContent.effects_executed,false);
  assert.equal(state.draft.reply,reply);
});

test('spoken schema and length remain strict',()=>{
  const draft={reply:'あ'.repeat(160),expression:'neutral',motion:'still',tone:'calm'};
  assert.equal(validateDraft(draft).reply.length,160);
  assert.throws(()=>validateDraft({...draft,reply:'あ'.repeat(161)}));
  assert.throws(()=>validateDraft({...draft,confidence:1}));
  assert.throws(()=>validateDraft({...draft,web:'search'}));
});
