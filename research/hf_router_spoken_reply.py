"""Online chat-only subprocess for an otherwise strictly offline model worker.

Input/output are private pipes, never files/logs; HF token env-only. No model loading,
Hub artifact download, automatic retry, credential/body diagnostics, or fallback.
"""
import json
import os
import sys


def main():
    try:
        if os.environ.get('HF_HUB_OFFLINE')!='0' or os.environ.get('TRANSFORMERS_OFFLINE')!='0':
            raise ValueError('chat-only child must have explicit online scope')
        raw=sys.stdin.read(131073)
        if len(raw)>131072:raise ValueError('bounded private chat input')
        r=json.loads(raw)
        if set(r)!={'model','messages','timeout'} or not 0<float(r['timeout'])<=120:raise ValueError('bounded request')
        from huggingface_hub import InferenceClient
        result=InferenceClient(base_url='https://router.huggingface.co/v1',api_key=os.environ['HF_TOKEN'],
            timeout=r['timeout']).chat.completions.create(model=r['model'],messages=r['messages'],
                max_tokens=160,temperature=.8,stream=False).choices[0]
        content=result.message.content
        if not isinstance(content,str) or len(content)>4096:raise ValueError('bounded spoken response')
        sys.stdout.write(json.dumps({'content':content,'finish_reason':result.finish_reason}))
        return 0
    except Exception:
        # Do not print exception/request/credentials/provider body.
        return 1


if __name__=='__main__':raise SystemExit(main())
