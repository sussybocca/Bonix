require.config({ paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.39.0/min/vs' }});
require(['vs/editor/editor.main'], function() {
    const editor = monaco.editor.create(document.getElementById('editor'), {
        value: `from flask import Flask\napp = Flask(__name__)\n\n@app.route("/")\ndef home():\n    return "Hello from Flask!"\n\nif __name__ == "__main__":\n    app.run(port=5000)`,
        language: 'python',
        theme: 'vs-dark',
    });

    document.getElementById('runFlaskBtn').addEventListener('click', async () => {
        const code = editor.getValue();
        const formData = new FormData();
        formData.append('code', code);

        const response = await fetch('/api/run_flask_app', { method: 'POST', body: formData });
        const data = await response.json();
        document.getElementById('flaskUrl').textContent = data.url;
    });
});