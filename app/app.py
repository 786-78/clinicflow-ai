import gradio as gr
import torch

def chat(message, history):
    conversation = ""
    for human, assistant in history:
        conversation += f"<|start_header_id|>user<|end_header_id|>\n{human}<|eot_id|>"
        conversation += f"<|start_header_id|>assistant<|end_header_id|>\n{assistant}<|eot_id|>"

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are ClinicFlow AI, a responsible clinic triage
and patient intake assistant.
You are calm, professional, empathetic and concise.
Never diagnose any disease. Never give medical advice
or treatment recommendations.
Always ask only one or two questions at a time.
At the end of every conversation, provide the
Structured Summary and strongly recommend
consulting a qualified doctor.<|eot_id|>
{conversation}<|start_header_id|>user<|end_header_id|>

{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()
    return response

css = """
footer { display: none !important; }
* { box-sizing: border-box; }

body {
    background: #f8fafc !important;
    font-family: 'Segoe UI', sans-serif !important;
}

.gradio-container {
    max-width: 860px !important;
    margin: 16px auto !important;
    padding: 0 !important;
    background: transparent !important;
    min-height: unset !important;
}

.wrap { gap: 0 !important; }

.card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
}

.hdr {
    background: #1e40af;
    padding: 11px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.hdr-left {
    display: flex;
    align-items: center;
    gap: 10px;
}
.hdr-logo {
    width: 32px; height: 32px;
    background: rgba(255,255,255,0.15);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.hdr-name {
    color: white;
    font-size: 14px;
    font-weight: 600;
    line-height: 1.2;
}
.hdr-owner {
    color: rgba(255,255,255,0.65);
    font-size: 11px;
}
.hdr-badge {
    font-size: 11px;
    color: rgba(255,255,255,0.85);
    display: flex;
    align-items: center;
    gap: 5px;
}
.hdr-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4ade80;
}

#cf-bot {
    border: none !important;
    border-radius: 0 !important;
    background: white !important;
}

#cf-bot .wrap {
    padding: 14px 20px !important;
    gap: 10px !important;
    background: white !important;
}

#cf-bot button { display: none !important; }

#cf-bot .user {
    background: #1e40af !important;
    color: white !important;
    border-radius: 16px 16px 4px 16px !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    padding: 9px 14px !important;
    max-width: 65% !important;
    min-width: 60px !important;
    width: auto !important;
    white-space: normal !important;
    word-break: break-word !important;
}

#cf-bot .bot {
    background: #f1f5f9 !important;
    color: #1e293b !important;
    border-radius: 4px 16px 16px 16px !important;
    border: 1px solid #e2e8f0 !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    padding: 9px 14px !important;
    max-width: 72% !important;
    width: auto !important;
    word-break: break-word !important;
}

.mid {
    border-top: 1px solid #e2e8f0;
    padding: 10px 20px 12px;
    background: white;
}

#cf-input textarea {
    font-size: 13px !important;
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    background: white !important;
    color: #1e293b !important;
    resize: none !important;
    line-height: 1.4 !important;
}

#cf-input textarea:focus {
    border-color: #1e40af !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(30,64,175,0.08) !important;
}

#cf-input textarea::placeholder {
    color: #94a3b8 !important;
}

#cf-send {
    background: #1e40af !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    min-width: 80px !important;
    height: 40px !important;
}

#cf-send:hover { background: #1e3a8a !important; }

#cf-clear {
    background: none !important;
    border: none !important;
    color: #94a3b8 !important;
    font-size: 11px !important;
    padding: 4px 0 !important;
    text-align: left !important;
    box-shadow: none !important;
}

#cf-clear:hover { color: #64748b !important; }

.ftr {
    border-top: 1px solid #e2e8f0;
    padding: 8px 20px;
    background: #f8fafc;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.ftr-left { font-size: 10px; color: #94a3b8; }
.ftr-right { display: flex; gap: 12px; }
.ftr-link { font-size: 10px; color: #94a3b8; text-decoration: none; }
.ftr-link:hover { color: #1e40af; }
"""

with gr.Blocks(css=css, title="ClinicFlow AI") as demo:

    with gr.Column(elem_classes="card"):

        gr.HTML("""
        <div class="hdr">
            <div class="hdr-left">
                <div class="hdr-logo">🏥</div>
                <div>
                    <div class="hdr-name">ClinicFlow AI</div>
                    <div class="hdr-owner">by Bilal Chawdhary</div>
                </div>
            </div>
            <div class="hdr-badge">
                <div class="hdr-dot"></div>
                Online
            </div>
        </div>
        """)

        chatbot = gr.Chatbot(
            elem_id="cf-bot",
            show_label=False,
            bubble_full_width=False,
            height=320,
        )

        with gr.Column(elem_classes="mid"):
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Describe your symptoms...",
                    show_label=False,
                    elem_id="cf-input",
                    scale=5,
                    lines=1,
                    max_lines=3,
                )
                send = gr.Button("Send ➤", elem_id="cf-send", scale=1)
            clear = gr.Button("🗑️ Clear", elem_id="cf-clear")

        gr.HTML("""
        <div class="ftr">
            <span class="ftr-left">
                ⚠️ Not a substitute for professional medical advice
            </span>
            <div class="ftr-right">
                <a class="ftr-link" href="https://huggingface.co/bilalchawdhary/clinic_flow" target="_blank">🤗 Model</a>
                <a class="ftr-link" href="https://github.com/bilalchawdhary" target="_blank">GitHub</a>
                <a class="ftr-link" href="https://linkedin.com/in/bilalchawdhary" target="_blank">LinkedIn</a>
            </div>
        </div>
        """)

    def respond(message, chat_history):
        if not message.strip():
            return "", chat_history
        response = chat(message, chat_history)
        chat_history.append((message, response))
        return "", chat_history

    send.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear.click(lambda: [], outputs=[chatbot])

demo.launch(share=True, debug=False)
