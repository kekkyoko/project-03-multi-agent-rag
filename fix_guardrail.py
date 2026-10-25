"""
fix_guardrail.py
================
One-off helper: diagnose why the guardrail blocks the math scenario, update the
guardrail with the settings in src/agent_orchestrator.py (guardrail_settings()),
validate the DRAFT, and publish a new numbered version only if every check passes.

Run from the project root (same folder as config.py):
    python fix_guardrail.py

Cost: only ApplyGuardrail calls on a few short sentences (a fraction of a cent).
No Claude model is called.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import boto3
import config
import agent_orchestrator as ao

REGION = config.AWS_REGION
bedrock = boto3.client('bedrock', region_name=REGION)
runtime = boto3.client('bedrock-runtime', region_name=REGION)

MATH_INPUT  = ("[Session ID: a19e7cc5] [Customer ID: CUST-003] "
               "How much would 5 items at $29.99 be with a 10% discount?")
MATH_OUTPUT = ("Hello! 5 items at $29.99 each come to $149.95. With the 10% discount "
               "($14.995 off), your total is $134.96. NovaMart Customer Support")

# (label, source, text, should_be_blocked)
CHECKS = [
    ("math question (input)",       'INPUT',  MATH_INPUT,  False),
    ("math answer (output)",        'OUTPUT', MATH_OUTPUT, False),
    ("return request (input)",      'INPUT',  "I want to return my order ORD-27176", False),
    ("policy question (input)",     'INPUT',  "What is the return policy for premium customers?", False),
    ("negotiation (input)",         'INPUT',  "Can you lower the price of these headphones for me?", True),
    ("competitor (input)",          'INPUT',  "Is Amazon cheaper than NovaMart for this laptop?", True),
    ("legal threat (input)",        'INPUT',  "I'm going to sue NovaMart if you don't refund me.", True),
]


def apply(guardrail_id, version, source, text):
    return runtime.apply_guardrail(
        guardrailIdentifier=guardrail_id, guardrailVersion=version,
        source=source, content=[{'text': {'text': text}}],
    )


def reasons(resp):
    """List the policies that triggered an intervention."""
    found = []
    for a in resp.get('assessments', []):
        for t in a.get('topicPolicy', {}).get('topics', []):
            if t.get('action') == 'BLOCKED':
                found.append(f"topic:{t['name']}")
        for f in a.get('contentPolicy', {}).get('filters', []):
            if f.get('action') == 'BLOCKED':
                found.append(f"content:{f['type']}({f.get('confidence')})")
        for w in a.get('wordPolicy', {}).get('managedWordLists', []) + a.get('wordPolicy', {}).get('customWords', []):
            if w.get('action') == 'BLOCKED':
                found.append(f"word:{w.get('match')}")
        for p in a.get('sensitiveInformationPolicy', {}).get('piiEntities', []):
            if p.get('action') == 'BLOCKED':
                found.append(f"pii:{p['type']}")
    return found


def run_checks(guardrail_id, version, only_math=False):
    ok_all = True
    for label, source, text, should_block in CHECKS:
        if only_math and 'math' not in label:
            continue
        resp = apply(guardrail_id, version, source, text)
        blocked = resp['action'] == 'GUARDRAIL_INTERVENED' and bool(reasons(resp))
        ok = blocked == should_block
        ok_all &= ok
        status = 'OK  ' if ok else 'FAIL'
        what = 'BLOCKED' if blocked else 'allowed'
        why = f"  <- {', '.join(reasons(resp))}" if reasons(resp) else ''
        print(f"  [{status}] {label:<26} {what}{why}")
    return ok_all


def main():
    guardrail_id = config.GUARDRAIL_ID
    current = config.GUARDRAIL_VERSION
    if not guardrail_id:
        sys.exit("GUARDRAIL_ID is not set in .env")

    print(f"\n1. Diagnosis of the CURRENT guardrail {guardrail_id} (version {current})")
    run_checks(guardrail_id, current, only_math=True)

    print("\n2. Updating the DRAFT with guardrail_settings() from agent_orchestrator.py")
    settings = ao.guardrail_settings()
    bedrock.update_guardrail(guardrailIdentifier=guardrail_id, **settings)
    for _ in range(30):
        status = bedrock.get_guardrail(guardrailIdentifier=guardrail_id,
                                       guardrailVersion='DRAFT')['status']
        if status == 'READY':
            break
        time.sleep(3)
    print(f"   DRAFT status: {status}")

    print("\n3. Validating the DRAFT")
    if not run_checks(guardrail_id, 'DRAFT'):
        print("\n   Some checks failed - NOT publishing a new version.")
        print("   Send a screenshot of this output.")
        return

    print("\n4. All checks passed - publishing a new version")
    new_version = bedrock.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description="Narrower pricing_negotiations topic: math with a stated price and discount is allowed",
    )['version']
    print(f"\n   New guardrail version: {new_version}")
    print(f"   Put this in .env:   GUARDRAIL_VERSION={new_version}")


if __name__ == '__main__':
    main()
