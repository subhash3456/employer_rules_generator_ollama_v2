import json
import streamlit as st
from pipeline import generate_rules, OLLAMA_BASE_URL, CHAT_MODEL, EMBED_MODEL, ROOT

st.set_page_config(page_title='Recruiter Policy → Java Rules', page_icon='🛡️', layout='wide')
st.title('Recruiter Policy → Existing Java rules.json')
st.caption('Local PDF extraction + Ollama. Generated rules are drafts and must be reviewed before deployment.')

with st.sidebar:
    st.subheader('Local Ollama')
    st.code(f'{OLLAMA_BASE_URL}\nChat: {CHAT_MODEL}\nEmbeddings: {EMBED_MODEL}')
    st.markdown(f'```bash\nollama pull {CHAT_MODEL}\nollama pull {EMBED_MODEL}\n```')
    st.caption('If your Ollama version/model does not support /api/embed, the app falls back to lexical retrieval.')
    st.divider()
    st.subheader('Loaded rule engine contract')
    try:
        ref = json.loads((ROOT / 'reference' / 'existing_rules_reference.json').read_text(encoding='utf-8'))
        cat = json.loads((ROOT / 'config' / 'rule_catalog.json').read_text(encoding='utf-8'))
        st.metric('Existing reference rules', len(ref))
        st.metric('Supported search types', len(cat['searchTypes']))
        st.metric('Supported operators', len(cat['allowedOperators']))
    except Exception as exc:
        st.error(str(exc))

st.warning('Employment screening is high impact. This tool does not make hiring decisions. Do not deploy generated rules without legal/compliance review and human approval. Never use protected characteristics or proxies.')
upload = st.file_uploader('Upload recruiter policy PDF', type=['pdf'])
if upload and st.button('Understand policy and generate draft rules', type='primary'):
    try:
        with st.spinner('Extracting PDF, retrieving policy passages and compiling with local Ollama...'):
            result = generate_rules(upload.getvalue(), upload.name)
        gen = result['generation']
        st.subheader("Generation Debug")

        st.write("Generated rules:", len(gen.get("rules", [])))

        st.write(
            "Unmapped policy items:",
            gen.get("unmappedPolicyItems", [])
            )

        st.write(
            "Assumptions:",
            gen.get("assumptions", [])
            )

        st.write(
            "Review notes:",
            gen.get("reviewNotes", [])
            )

        st.write(
            "Validation errors:",
            gen.get("validationErrors", [])
            )

        st.write(
            "Evidence errors:",
            gen.get("evidenceErrors", [])
            )

        st.json(gen)

        rules_json = json.dumps(gen['rules'], indent=2, ensure_ascii=False)
        evidence_json = json.dumps({k: gen[k] for k in ['sourceEvidence','unmappedPolicyItems','assumptions','reviewNotes','validationErrors','evidenceErrors']}, indent=2, ensure_ascii=False)
        if gen['exportAllowed']:
            st.success(f'Generated {len(gen["rules"])} draft rule(s). Structural validation and source-evidence checks passed. Human approval is still required.')
        else:
            st.error('Validation failed. Fix the issues before exporting or loading these rules into Java.')
        if gen['validationErrors']:
            st.subheader('Rule validation errors')
            for err in gen['validationErrors']: st.error(err)
        if gen['evidenceErrors']:
            st.subheader('Source evidence errors')
            for err in gen['evidenceErrors']: st.error(err)
        st.subheader('Generated rules.json — JSON array')
        st.code(rules_json or '[]', language='json')
        if gen['exportAllowed']:
            st.download_button('Download rules.json', rules_json, 'rules.json', 'application/json')
        else:
            st.download_button('Download draft for debugging (NOT validated)', rules_json, 'rules.draft.json', 'application/json')
        st.download_button('Download generation review report', evidence_json, 'generation_review.json', 'application/json')
        a,b=st.columns(2)
        with a:
            with st.expander('Source evidence quotes'):
                st.json(gen['sourceEvidence'])
            with st.expander('Unmapped policy requirements'):
                st.write(gen['unmappedPolicyItems'] or 'None')
        with b:
            with st.expander('Human review notes and assumptions'):
                st.write({'reviewNotes':gen['reviewNotes'],'assumptions':gen['assumptions']})
            with st.expander('Retrieved PDF passages'):
                for item in result['retrievedChunks']:
                    st.markdown(f'**PDF page {item["page"]}**')
                    st.write(item['text'])
        with st.expander('Full extracted PDF text'):
            st.text(result['fullText'])
    except Exception as exc:
        st.exception(exc)
        st.info('Confirm Ollama is running, the configured models are pulled, and the PDF contains selectable text.')
else:
    st.info('Start with samples/sample_recruiter_policy.pdf. The project includes your supplied 47-rule baseline and a redacted candidate-report example.')
