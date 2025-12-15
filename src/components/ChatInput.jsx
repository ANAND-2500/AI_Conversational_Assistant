import { useState, useRef, useEffect } from "react";

function ChatInput({ onSend, disabled }) {
  const [text, setText] = useState("");
  const ref = useRef(null);

  useEffect(() => {
    if (!disabled) ref.current?.focus();
  }, [disabled]);

  function autoResize() {
    const el = ref.current;
    el.style.height = "auto";
    el.style.height = el.scrollHeight + "px";
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  function send() {
    if (!text.trim() || disabled) return;
    onSend(text.trim());
    setText("");
    ref.current.style.height = "40px";
  }

  return (
    <div className="input-area">
      <div className="input-box">
        <textarea
          ref={ref}
          value={text}
          onChange={e => {
            setText(e.target.value);
            autoResize();
          }}
          onKeyDown={handleKeyDown}
          placeholder={disabled ? "AI is replying…" : "Message AI Assistant…"}
          disabled={disabled}
        />
        <button className="send-btn" onClick={send} disabled={disabled}>
          ➤
        </button>
      </div>
    </div>
  );
}

export default ChatInput;
