import { getDocumentFileURL } from "./api";

export default function PdfPreview({ chatId, upload, activePage, previewKey }) {
  if (!upload) {
    return (
      <div className="pdf-placeholder">
        <p>Upload a PDF to see a preview here.</p>
      </div>
    );
  }

  return (
    <div className="pdf-preview">
      <div className="pdf-label">
        Page {activePage}
      </div>
      <iframe
        key={previewKey}
        src={getDocumentFileURL(chatId, activePage)}
        title="PDF Preview"
        className="pdf-iframe"
      />
    </div>
  );
}
