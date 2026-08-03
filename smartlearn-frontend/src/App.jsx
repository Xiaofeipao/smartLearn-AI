import { useState } from "react";
import { uploadPDF, getDocumentFileURL } from "./api";
import PdfPreview from "./PdfPreview";
import ChatPanel from "./ChatPanel";

export default function App() {
  const [chatId, setChatId] = useState("day3-demo");
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [activePage, setActivePage] = useState(1);
  const [previewKey, setPreviewKey] = useState(0);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [chatKey, setChatKey] = useState(0);

  async function handleUpload() {
    if (!file || !chatId.trim()) return;
    setStatus("Uploading…");
    setError(null);
    try {
      const data = await uploadPDF(file, chatId.trim());
      setUpload(data);
      setActivePage(1);
      setPreviewKey((k) => k + 1);
      setChatKey((k) => k + 1);
    } catch (e) {
      setError(e.message);
    } finally {
      setStatus(null);
    }
  }

  function handleJumpToPage(page) {
    setActivePage(page);
    setPreviewKey((k) => k + 1);   // force iframe reload to jump to #page=N
  }

  return (
    <main className="app">
      <h1>SmartLearn Lite</h1>

      <div className="upload-zone">
        <label htmlFor="chat-id-input">Chat ID</label>
        <input
          id="chat-id-input"
          type="text"
          value={chatId}
          onChange={(e) => setChatId(e.target.value)}
          placeholder="e.g. day3-demo"
        />
        <label htmlFor="file-input">PDF file</label>
        <input
          id="file-input"
          type="file"
          accept=".pdf"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button
          type="button"
          onClick={handleUpload}
          disabled={!file || !chatId.trim() || status}
        >
          Upload
        </button>
      </div>

      {upload && (
        <p className="upload-info">
          Uploaded: {upload.filename} — {upload.pages} pages,{" "}
          {upload.characters} characters
        </p>
      )}

      {status && <p className="status-text">{status}</p>}
      {error && <div className="error-banner" role="alert">{error}</div>}

      <div className="workspace">
        <div className="workspace-left">
          <PdfPreview
            chatId={chatId.trim()}
            upload={upload}
            activePage={activePage}
            previewKey={previewKey}
          />
        </div>
        <div className="workspace-right">
          <ChatPanel
            key={chatKey}
            chatId={chatId.trim()}
            enabled={!!upload}
            onJumpToPage={handleJumpToPage}
          />
        </div>
      </div>
    </main>
  );
}
