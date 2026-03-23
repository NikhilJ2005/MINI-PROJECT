import { useState, useRef, useEffect } from "react";
import "./cssFile/ResumeUpload.css";

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

function ResumeUpload({onFileSelect}) {
  const [fileName, setFileName] = useState("");
  const [fileSize, setFileSize] = useState(0);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const validTypes = ["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"];
  const validExtensions = [".pdf", ".txt", ".docx"];

  const handleFile = (file) => {
    if (!file) return;
    const dotIdx = file.name.lastIndexOf(".");
    const ext = dotIdx >= 0 ? file.name.toLowerCase().slice(dotIdx) : "";

    if (validTypes.includes(file.type) && validExtensions.includes(ext)) {
      setFileName(file.name);
      setFileSize(file.size);
      setError("");
      onFileSelect(file);
    } else {
      setFileName("");
      setFileSize(0);
      setError("Only PDF, DOCX, or TXT files are allowed.");
      onFileSelect(null);
    }
  };

  const removeFile = (e) => {
    e.stopPropagation();
    setFileName("");
    setFileSize(0);
    setError("");
    onFileSelect(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    handleFile(file);
    e.target.value = "";
  };

  // Clipboard paste support
  useEffect(() => {
    const handlePaste = (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.kind === "file") {
          const file = item.getAsFile();
          if (file) {
            handleFile(file);
            break;
          }
        }
      }
    };
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, []);

  const fileIcon = fileName.endsWith(".pdf") ? "\uD83D\uDCC4" : "\uD83D\uDCDD";

  return (
    <div className="resume-upload-container">
      <label htmlFor="resume-upload" className="upload-label">
        Upload Resume
      </label>

      <div
        className={`upload-box ${isDragging ? "dragging" : ""}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleBrowseClick}
        role="button"
        tabIndex={0}
      >
        <span className="upload-icon">{fileName ? fileIcon : "\uD83D\uDCC4"}</span>
        <p className="upload-title">
          <strong>Drag & drop your resume here</strong>
        </p>
        <p className="upload-subtext">or click to browse / paste (Ctrl+V) &mdash; .pdf, .docx, .txt</p>

        {fileName && (
          <p className="selected-file">
            <span className="file-info">
              {fileName}
              <span className="file-size">({formatFileSize(fileSize)})</span>
            </span>
            <button
              type="button"
              className="remove-file-btn"
              onClick={removeFile}
              title="Remove file"
            >
              x
            </button>
          </p>
        )}

        {error && (
          <p className="error-text">
            {error}
          </p>
        )}
      </div>

      <input
        id="resume-upload"
        type="file"
        accept=".pdf,.docx,.txt"
        ref={fileInputRef}
        onChange={handleFileChange}
        hidden
      />
    </div>
  );
}

export default ResumeUpload;
