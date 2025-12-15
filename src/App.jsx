import { useState, useEffect } from "react";
import Sidebar from "./components/sidebar";
import ChatContainer from "./components/ChatContainer";
import "./App.css";

function App() {
  const [chats, setChats] = useState({});
  const [activeId, setActiveId] = useState(null);

  /* Load from localStorage */
  useEffect(() => {
    const saved = localStorage.getItem("chats");
    if (saved) {
      const parsed = JSON.parse(saved);
      setChats(parsed);
      const firstId = Object.keys(parsed)[0] || null;
      setActiveId(firstId);
    } else {
      createInitialChat();
    }
  }, []);

  /* Save to localStorage */
  useEffect(() => {
    localStorage.setItem("chats", JSON.stringify(chats));
  }, [chats]);

  function createInitialChat() {
    const id = Date.now().toString();
    setChats({
      [id]: { id, title: "New Chat", messages: [] }
    });
    setActiveId(id);
  }

  function newChat() {
    const id = Date.now().toString();
    setChats(prev => ({
      [id]: { id, title: "New Chat", messages: [] },
      ...prev
    }));
    setActiveId(id);
  }

  function renameChat(id, title) {
    setChats(prev => ({
      ...prev,
      [id]: { ...prev[id], title }
    }));
  }

  function deleteChat(id) {
    setChats(prev => {
      const copy = { ...prev };
      delete copy[id];
      const nextId = Object.keys(copy)[0] || null;
      setActiveId(nextId);
      return copy;
    });
  }

  function addMessage(chatId, message) {
    setChats(prev => {
      const chat = prev[chatId];
      if (!chat) return prev;

      const text = String(message.text ?? "");

      const shouldRename =
        chat.title === "New Chat" && message.sender === "user";

      return {
        ...prev,
        [chatId]: {
          ...chat,
          title: shouldRename ? text.slice(0, 30) : chat.title,
          messages: [...chat.messages, { ...message, text }]
        }
      };
    });
  }

  return (
    <div className="app">
      <Sidebar
        chats={chats}
        activeId={activeId}
        onSelect={setActiveId}
        onNewChat={newChat}
        onRename={renameChat}
        onDelete={deleteChat}
      />

      {activeId && (
        <ChatContainer
          messages={chats[activeId]?.messages || []}
          onSend={text =>
            addMessage(activeId, { sender: "user", text })
          }
          onBotReply={text =>
            addMessage(activeId, { sender: "bot", text })
          }
        />
      )}
    </div>
  );
}

export default App;
