import { useState, useEffect, useCallback, createContext, useContext, useRef } from "react";
import "./cssFile/Toast.css";

let toastCounter = 0;

// Context-based toast system
const ToastContext = createContext(null);

export function useToast() {
  return useContext(ToastContext);
}

// Imperative API for non-component code — uses a ref to the latest addToast
const addToastRef = { current: null };

export function showToast(message, type = "success", duration = 3000) {
  if (addToastRef.current) {
    addToastRef.current({ id: ++toastCounter, message, type, duration });
  }
}

function Toast({ toast, onRemove }) {
  useEffect(() => {
    const timer = setTimeout(() => onRemove(toast.id), toast.duration);
    return () => clearTimeout(timer);
  }, [toast, onRemove]);

  return (
    <div className={`toast toast-${toast.type}`}>
      <span className="toast-icon">
        {toast.type === "success" ? "✓" : toast.type === "error" ? "✕" : "ℹ"}
      </span>
      <span className="toast-message">{toast.message}</span>
      <button className="toast-close" onClick={() => onRemove(toast.id)}>×</button>
    </div>
  );
}

export default function ToastContainer() {
  const [toasts, setToasts] = useState([]);

  const addToast = useCallback((toast) => {
    setToasts((prev) => [...prev, toast]);
  }, []);

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  // Keep the ref in sync with the latest addToast callback
  useEffect(() => {
    addToastRef.current = addToast;
    return () => { addToastRef.current = null; };
  }, [addToast]);

  return (
    <ToastContext.Provider value={addToast}>
      <div className="toast-container">
        {toasts.map((t) => (
          <Toast key={t.id} toast={t} onRemove={removeToast} />
        ))}
      </div>
    </ToastContext.Provider>
  );
}
