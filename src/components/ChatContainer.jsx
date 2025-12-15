import { useState, useEffect, useRef } from "react";
import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";

function ChatContainer({ messages, onSend, onBotReply }) {
  const [typing, setTyping] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typing]);

  async function handleSend(text) {
    // USER MESSAGE
    onSend(text);

    // LOCK INPUT
    setTyping(true);

    // MOCK BOT DELAY
    setTimeout(() => {
      onBotReply(`Mock AI reply: ${text}`);
      setTyping(false);
    }, 1000);
  }

  return (
    <div className="chat-wrapper">
      <div className="messages">
        {messages.map((msg, i) => (
          <ChatMessage key={i} sender={msg.sender} text={msg.text} />
        ))}

        {typing && (
          <div className="message bot">
            <div className="bubble typing">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <ChatInput onSend={handleSend} disabled={typing} />
    </div>
  );
}

export default ChatContainer;
