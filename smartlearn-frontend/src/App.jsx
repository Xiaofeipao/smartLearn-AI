import { useState } from "react";
import { uploadPDF, askQuestion } from "./api";

export default function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  async function handleUpload() {
    if (!file) return;
    setStatus("Uploading…");
    setError(null);
    try {
      const data = await uploadPDF(file);
      setUpload(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setStatus(null);
    }
  }

  async function handleAsk() {
    const q = message.trim();
    if (!q) return;
    setStatus("Asking…");
    setError(null);
    try {
      const data = await askQuestion(q);
      setAnswer(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setStatus(null);
    }
  }

  return (
    <main className="app">
      <h1>SmartLearn Lite</h1>

      <section className="upload-zone">
        <label htmlFor="file-input">PDF file</label>
        <input
          id="file-input"
          type="file"
          accept=".pdf"
          onChange={(e) => setFile(e.target.files[0])}
        />
        {file && <span className="file-name">{file.name}</span>}
        <button type="button" onClick={handleUpload} disabled={!file || status}>
          Upload
        </button>
      </section>

      {upload && (
        <p className="upload-info">
          {upload.filename} — {upload.pages} pages, {upload.characters}{" "}
          characters
        </p>
      )}

      <section className="ask-zone">
        <label htmlFor="message">Question</label>
        <textarea
          id="message"
          rows={3}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
        />
        <button
          type="button"
          onClick={handleAsk}
          disabled={!upload || !message.trim() || status}
        >
          Ask
        </button>
      </section>

      {status && (
        <p className="status-text" role="status">
          {status}
        </p>
      )}
      {error && (
        <p className="error-banner" role="alert">
          {error}
        </p>
      )}

      {answer && (
        <section className="answer-card">
          <p className="answer-text">{answer.answer}</p>
          {answer.citations.length > 0 && (
            <ul className="citations">
              {answer.citations.map((page) => (
                <li className="citation-chip" key={page}>
                  Page {page}
                </li>
              ))}
            </ul>
          )}
        </section>
      )}
    </main>
  );
}
