import { useState } from "react";
import { uploadPDF } from "./api";
import PdfPreview from "./PdfPreview";
import ChatPanel from "./ChatPanel";

export default function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [activePage, setActivePage] = useState(1);
  const [previewKey, setPreviewKey] = useState(0);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [chatKey, setChatKey] = useState(0);

  async function handleUpload() {
    if (!file) return;
    setStatus("Uploading…");
    setError(null);
    try {
      const data = await uploadPDF(file);
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
  }

  return (
    <main className="app">
      <h1>SmartLearn Lite</h1>

      <div className="upload-zone">
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
          disabled={!file || status}
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
            upload={upload}
            activePage={activePage}
            previewKey={previewKey}
          />
        </div>
        <div className="workspace-right">
          <ChatPanel
            key={chatKey}
            enabled={!!upload}
            onJumpToPage={handleJumpToPage}
          />
        </div>
      </div>
    </main>
  );
}
